# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-serving-path. Route: course/route/lessen/systemen.yaml. Checks, het model van de course, het serving-pad en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-serving-path", "nl")
    C = L.checks   # het model van de course, de verzoeken van de les, het serving-pad en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl) en [Je model achter een API](/course/open/sys-serving-api?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_threshold():
    quick_threshold = L.choice(
        "quick-threshold",
        question="Een spamfilter noemt een mail spam zodra de kans op spam 0,9 bereikt. Je verlaagt de decision threshold naar 0,5. Wat gebeurt er?",
        options={
            "same": "Er verandert niets: het model geeft dezelfde kansen",
            "right": "Meer spam wordt gevangen, en ook meer goede mail wordt spam genoemd",
            "fewer": "Er komt minder spam door, en minder goede mail wordt spam genoemd",
            "retrain": "Het filter moet eerst opnieuw getraind worden voor de nieuwe threshold",
        },
        correct="right",
        why={
            "right": "Een lagere threshold noemt meer mails spam: de recall gaat omhoog, en de false positives ook. Het model zelf blijft hetzelfde; alleen de grens erna schuift. In deze les maakt zo'n grens van het getal van een model een beslissing.",
            "same": "De kansen blijven hetzelfde. Maar welke kansen tellen nu als spam? Zie [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
            "fewer": "Vanaf 0,5 worden meer mails spam genoemd. Kunnen er dan minder goede mails spam genoemd worden? Zie [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
            "retrain": "De threshold komt na het model. Moet het model veranderen als de grens schuift? Zie [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
        },
    )
    quick_threshold
    return (quick_threshold,)


@app.cell(hide_code=True)
def quick_check_ok():
    quick_ok = L.choice(
        "quick-ok",
        question="Een server beantwoordt een verzoek om een voorspelling met status 200 en {\"bikes\": 4855}. Wat zegt de 200?",
        options={
            "busy": "Dat het een druk uur wordt",
            "checked": "Dat het model het getal vergeleek met de data van de training",
            "right": "Dat de server kon antwoorden, niet dat het getal klopt",
            "right-number": "Dat het getal klopt, want een fout getal krijgt status 400",
        },
        correct="right",
        why={
            "right": "200 betekent dat de server het verzoek kon lezen, het model kon uitvoeren en kon antwoorden. Een temperatuur van 26,2 in graden, waar de data 0,64 heeft, krijgt ook 200 en 4.855 fietsen. De waarde van het getal hangt af van wat er voor en na het model gebeurt: het onderwerp van deze les.",
            "busy": "De statuscode gaat over het verzoek, niet over de fietsen. Wat zegt 200 over hoe het verzoek ging? Zie [Je model achter een API](/course/open/sys-serving-api?lang=nl).",
            "checked": "De server controleerde de velden van een verzoek, niet het getal van het model. Wat zegt 200 dan? Zie [Je model achter een API](/course/open/sys-serving-api?lang=nl).",
            "right-number": "Een 400 is voor een verzoek waar de server niets mee kan, zoals een ontbrekend veld. Krijgt een goed gevormd verzoek met een temperatuur in graden er een? Zie [Je model achter een API](/course/open/sys-serving-api?lang=nl).",
        },
    )
    quick_ok
    return (quick_ok,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Ook een model met goede cijfers in de validatie kan in productie slecht werken, zonder één foutmelding. In [Je model achter een API](/course/open/sys-serving-api?lang=nl) lagen de antwoorden voor 3:00 en 4:00 's nachts onder 0 fietsen, tot −11, en kreeg een temperatuur in graden 4.855 fietsen, allemaal met status 200. Het model zit op de validatiedagen gemiddeld 34 fietsen per uur ernaast; de fout zat niet in het model, maar eromheen.

    Naar het model toe moet elke input precies zo voorbereid worden als bij de training: dezelfde kolommen, in dezelfde volgorde, in dezelfde eenheden, geschaald met dezelfde getallen. Na het model moet een getal iets worden waar mensen mee aan de slag kunnen: een heel aantal fietsen, nooit onder 0, en een beslissing, zoals of er een extra crew met een vrachtwagen op pad gaat om de stations te vullen. In *Rules of Machine Learning* schreef Martin Zinkevich op wat engineers bij Google leerden van jaren met modellen in productie. Regel 32 zegt: gebruik dezelfde code in de trainingspipeline en in de serving-pipeline; regel 37 zegt: meet het verschil tussen hoe een model het doet bij de training en in productie.

    **Een voorbeeld uit het dagelijks leven.** Een recept voor een oven in Celsius zegt 180 graden. Een oven in Fahrenheit is bij 180 graden amper warm, en de taart komt er rauw uit. Het recept klopt en de oven werkt, en er piept niets: samen werken recept en oven alleen als allebei met een getal hetzelfde bedoelen.
    """)
    return


@app.cell(hide_code=True)
def predict_batch():
    guess_batch = L.guess(
        "batch",
        question="Het model van het project leerde van features die gestandaardiseerd waren met het gemiddelde en de standaardafwijking van de trainingsuren. Een nieuwe nachtelijke batch bereidt de 24 uren van een dag zelf voor: de batch standaardiseert de uren met het gemiddelde en de standaardafwijking van alleen deze 24 uren. Wat gebeurt er volgens jou met de voorspelling voor 8:00 op donderdag 14 juli 2011, een heldere zomerdag?",
        options={
            "error": "De code stopt met een fout: de schaal past niet meer bij het model",
            "same": "Er verandert niets: hetzelfde uur, met hetzelfde weer",
            "more": "Veel meer fietsen, want de ochtend lijkt warm",
            "right": "Veel minder fietsen, en niets stopt met een fout",
        },
        reveal="**Veel minder fietsen, en niets stopt met een fout.** Met de schaal van de training krijgt 8:00 431 fietsen, en er werden er 486 verhuurd. Met de schaal van de batch krijgt 8:00 er 169. In juli is elk uur warm, dus is ook het gemiddelde van de batch warm: temp 0,64 om 8:00 ligt onder het eigen gemiddelde van de dag, 0,68, en wordt −0,77 in plaats van +0,76. Voor het model lijkt een warme ochtend nu een koele. De weekdag, de maand en het seizoen hebben de hele dag één waarde, dus worden alle drie 0, en dat betekent voor het model een gemiddelde weekdag, maand en seizoen. Over alle 73 validatiedagen zit zo'n batch gemiddeld 73 fietsen per uur ernaast, waar de schaal van de training 34 ernaast zit. De code liep, elk uur kreeg een getal, en de getallen waren fout.",
    )
    guess_batch
    return (guess_batch,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Het serving-pad.** Tussen een verzoek en een antwoord is het model één stap van meerdere. Het *serving-pad* van deze les heeft er vijf: het verzoek; de controle uit [Je model achter een API](/course/open/sys-serving-api?lang=nl), met 400 als antwoord op een fout verzoek; *prepare*, om van het verzoek de 34 inputs te maken waarvan het model leerde; het model; en *decide*, om van het getal een antwoord te maken waar het bedrijf iets mee kan. Alles naar het model toe heeft één taak: het model de input geven zoals bij de training. Na het model beslist het bedrijf.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **Training-serving skew.** Als de service een input anders voorbereidt dan de training deed, krijgt het model getallen waarvan het nooit leerde, en geeft het getallen die ernaast zitten, zonder foutmelding. Dat verschil is **training-serving skew**. Het komt in veel vormen voor:

        - *opnieuw berekende statistieken*: een scaler met het gemiddelde en de standaardafwijking van de data van de dag, in plaats van de getallen uit de training, zoals in de gok;
        - *eenheden*: een temperatuur in graden, waar de data van de training graden gedeeld door 41 heeft, zoals bij de 4.855 fietsen;
        - *tijd*: uren in UTC op de ene plek en in lokale tijd op een andere, zodat 8:00 in Washington, D.C., 12:00 of 13:00 wordt;
        - *volgorde*: dezelfde 10 features in een andere volgorde, zodat elke weight de verkeerde kolom vermenigvuldigt.

        De belangrijkste verdediging: één functie `prepare` die de training en de service allebei aanroepen, met het gemiddelde en de standaardafwijking één keer berekend, op de trainingsuren, en opgeslagen in config.json naast de weights. Dat is stap 4 van het recept van [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl): een nieuw voorbeeld wordt gestandaardiseerd met dezelfde `mean` en `std`. Je vangt wat er toch nog doorheen glipt met een controle van de eenheden en de tijdzone van elk verzoek, met een test die dezelfde uren door beide paden stuurt en de inputs vergelijkt, en met een log van de inputs van de service.

        **Bedrijfsregel.** Na het model wordt een getal een antwoord. Een **bedrijfsregel** is een vaste regel van de organisatie, toegepast door een systeem rond een model, wat het model ook zegt: nooit minder dan 0 fietsen, alleen hele fietsen, geen crew tussen middernacht en 6:00. Zulke regels komen niet uit data, en een model hoort de regels niet te leren: mensen bepalen de regels, de regels veranderen zonder nieuwe training, en in de code zijn de regels voor iedereen te lezen.

        **Een decision threshold uit kosten.** In een druk uur, met minstens 400 verhuringen, sturen de planners van het fietsdeelbedrijf een extra crew met een vrachtwagen om de stations te vullen. Het model geeft een getal; een decision threshold, zoals in [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl), maakt er ja of nee van. De plek van de threshold hangt af van wat elke vergissing kost. Een rit van de crew voor niets kost één eenheid; een druk uur zonder crew kost er drie, in lege stations en ritten die nooit gebeuren. De threshold hoeft dus niet bij de 400 van een druk uur te liggen. Je kiest de threshold op de validation set, in Oefening 3.

        **Op grote schaal.** Grote teams houden de voorbereiding op één plek. TensorFlow Transform berekent de statistieken één keer over de trainingsdata en schrijft de statistieken in de graph die de server uitvoert. TensorFlow Data Validation vergelijkt de inputs van de training met de inputs in productie, en meldt skew. Een *feature store* berekent elke feature één keer, en levert dezelfde waarde aan de training en aan het model in productie. Het idee blijft hetzelfde: één definitie van elke input, aan beide kanten gebruikt.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **training-serving skew** | | een verschil tussen hoe inputs bij de training en in productie worden voorbereid, waardoor een goed model foute getallen geeft zonder foutmelding | |
        | **bedrijfsregel** | | een vaste regel van de organisatie, toegepast door een systeem rond een model, wat het model ook zegt | |
        """),
        L.watch_out("Goede cijfers in de validatie zeggen niets over skew: de validatie bereidde de uren voor met de code van de training. Skew blijkt pas in productie, dus log de inputs die een service krijgt, en vergelijk de inputs met de inputs van de training."),
        L.watch_out("Train een model niet om nooit minder dan 0 te geven, of om 's nachts stil te blijven: dan mengt het model wat het uit data leerde met wat mensen besloten, en kan niemand de regel veranderen zonder opnieuw te trainen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Het serving-pad rond een model bouwen:

    1. Schrijf de voorbereiding één keer, als functie die de training en de service allebei aanroepen.
    2. Bewaar de statistieken van de training, zoals het gemiddelde en de standaardafwijking van elke feature, naast de weights, en laad de statistieken bij het starten van de service; bereken de statistieken nooit opnieuw op de data in productie.
    3. Controleer eerst elk verzoek, en houd de eenheden, de tijdzone en de volgorde van de features zoals bij de training.
    4. Maak van het getal van het model een antwoord met bedrijfsregels: afgerond, nooit onder 0, en wat de organisatie verder besloot.
    5. Kies de decision threshold op de validation set, uit wat elke vergissing kost.
    6. Test dat beide paden voor dezelfde uren dezelfde inputs geven, en log de inputs die de service krijgt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.prepare(request)` is de voorbereiding van de course, met het gemiddelde en de standaardafwijking van de trainingsuren uit `C.CONFIG`. `C.expect(x)` voert het model uit op de 34 inputs en geeft de fietsen als gewoon getal, en `C.decide(pred)` maakt van dat getal fietsen en een beslissing. De cel stuurt twee uren van 14 juli 2011 door het pad: 8:00, en 4:00 's nachts. De eerste 24 inputs zijn het uur; de cel print de 10 daarna, afgerond, in de volgorde van `C.FEATURES`: weekday, workingday, holiday, weathersit, temp, atemp, hum, windspeed, season en mnth. Voer de cel uit.
    """)
    return


@app.cell
def example_path():
    for _request in (C.DAY[8], C.DAY[4]):
        _x = C.prepare(_request)
        _pred = C.expect(_x)
        print("uur", _request["hr"], "features:", [round(v, 2) for v in _x[24:].tolist()])
        print("  model:", round(_pred, 1), " decide:", C.decide(_pred))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Om 8:00 gaat temp 0,64 erin als 0,76, het vijfde getal: iets warmer dan het gemiddelde trainingsuur. Het model maakt 431,2, en `decide` maakt er 431 fietsen van en stuurt de extra crew; er werden er 486 verhuurd. Om 4:00 maakt het model −10,8, een getal waar geen planner iets mee kan, en `decide` maakt er 0 fietsen van. In Oefening 1 schrijf je de regel waarmee `prepare` standaardiseert; in Oefening 2 de bedrijfsregel van `decide`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_prepare():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Zoals bij de training

        `prepare(request)` krijgt de body van een verzoek als dict, met `hr` en de 10 features, en geeft de 34 inputs van het model terug als één tensor. `one_hot` maakt `hour`, de 24 getallen van het uur. `values` bevat de 10 features in een tensor, gemaakt met `torch.tensor`: `float(request[f])` voor elke naam `f` in `FEATURES`, in de volgorde van de training. `CONFIG` is wat config.json van de training bewaart: een dict met een list van 10 gemiddelden onder `"mean"` en een list van 10 standaardafwijkingen onder `"std"`, allebei gewone Python-lists. `torch.cat` zet `hour` en `scaled` achter elkaar. Vervang de `...` zodat `scaled` gelijk is aan `values`, gestandaardiseerd met de getallen van de training.

        De check roept `prepare(C.DAY[8])` aan, en vergelijkt de 34 inputs met de inputs van de training. Daarna 17:00 en 3:00 van dezelfde dag, en 17:00 op 7 december 2011, bij lichte regen.
        """),
        L.task("Maak van `scaled` de features, gestandaardiseerd met het gemiddelde en de standaardafwijking van de training."),
    ])
    return


@app.cell(hide_code=True)
def exercise_prepare():
    box_prepare = L.code_box("prepare", lines=5)
    box_prepare
    return (box_prepare,)


@app.cell(hide_code=True)
def check_prepare(box_prepare):
    result_prepare = L.check("prepare", box_prepare.value)
    result_prepare
    return (result_prepare,)


@app.cell(hide_code=True)
def hints_prepare():
    L.hints(
        "prepare",
        [
            "Stap 2 van het recept: standaardiseer met de getallen uit de training, nooit met getallen van het verzoek zelf. Trek van `values` de 10 gemiddelden af, en deel daarna door de 10 standaardafwijkingen. Beide zijn lists in `CONFIG`, onder `\"mean\"` en `\"std\"`, en `torch.tensor` maakt van een list een tensor. Het resultaat gaat in `scaled`.",
            "Zoals in [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl): `(new - mean) / std`. Python deelt voordat het aftrekt, dus heeft de aftrekking haakjes nodig. Wat komt er na het minteken, als tensor?",
            "`    scaled = (values - ___) / torch.tensor(CONFIG[\"std\"])`",
        ],
        why="`scaled = (values - torch.tensor(CONFIG[\"mean\"])) / torch.tensor(CONFIG[\"std\"])`: elke feature min het gemiddelde over de trainingsuren, gedeeld door de standaardafwijking over de trainingsuren. Om 8:00 wordt temp 0,64 dan 0,76, zoals bij de training, en het model maakt weer 431,2 fietsen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_decide():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een bedrijfsregel

        `decide(pred)` krijgt het getal dat het model voor één uur maakt, een gewoon Python-getal zoals 431,2 of −10,8, en geeft het antwoord van de service terug: een dict met de fietsen en de beslissing. `THRESHOLD` is 400: vanaf zoveel fietsen is `extra_crew` gelijk aan `True`, en sturen de planners de extra crew. Vervang de `...` zodat `bikes` gelijk is aan `pred`, afgerond op een heel aantal fietsen, en nooit onder 0.

        De check roept `decide(-10.8)` aan: het getal van 4:00 's nachts. Daarna de getallen van 8:00, 19:00, 17:00 en 3:00 van dezelfde dag, en een getal net onder 400.
        """),
        L.task("Maak van `bikes` een heel aantal fietsen, nooit onder 0."),
    ])
    return


@app.cell(hide_code=True)
def exercise_decide():
    box_decide = L.code_box("decide", lines=4)
    box_decide
    return (box_decide,)


@app.cell(hide_code=True)
def check_decide(box_decide):
    result_decide = L.check("decide", box_decide.value)
    result_decide
    return (result_decide,)


@app.cell(hide_code=True)
def hints_decide():
    L.hints(
        "decide",
        [
            "Stap 4 van het recept: rond `pred` af met `round`, en zorg dat het resultaat niet onder 0 komt, zodat de −10,8 van het model 0 fietsen wordt. Het antwoord gaat in `bikes`.",
            "Python heeft een functie die de grootste van twee getallen geeft. Met 0 als een van de twee komt er nooit iets onder 0 uit. Welke functie is het?",
            "`    bikes = ___(0, round(pred))`",
        ],
        why="`bikes = max(0, round(pred))`: `round` maakt van 431,2 het hele getal 431, en `max` geeft de grootste van 0 en dat getal, dus wordt −10,8 dan 0 fietsen. Het model blijft zoals het is; de regel erna maakt van het getal een antwoord waar de planners iets mee kunnen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_threshold():
    mo.md(r"""
    ### Oefening 3. Waar trek je de grens?

    De planners tellen de beslissingen van `decide` op de 1.733 uren van de 73 validatiedagen, voor vier thresholds. Van deze uren waren er 115 druk, met minstens 400 verhuringen. Een rit van de crew voor niets kost 1; een druk uur zonder crew kost 3.

    | Threshold | Crew in een druk uur | Crew voor niets | Druk uur, geen crew |
    |---|---|---|---|
    | 300 | 100 | 110 | 15 |
    | 350 | 84 | 47 | 31 |
    | 400 | 62 | 12 | 53 |
    | 450 | 34 | 3 | 81 |
    """)
    return


@app.cell(hide_code=True)
def choice_threshold_cell():
    choice_threshold = L.choice(
        "threshold",
        question="Welke threshold kost de planners het minst?",
        options={
            "300": "300 fietsen: de crew mist alleen 15 drukke uren",
            "right": "350 fietsen: 47 ritten voor niets en 31 gemiste uren",
            "400": "400 fietsen: dezelfde grens als een druk uur",
            "450": "450 fietsen: de crew gaat bijna nooit voor niets op pad",
        },
        correct="right",
        why={
            "right": "350 kost 47 + 3 × 31 = 140, het minst van de vier: 300 kost 110 + 3 × 15 = 155, 400 kost 12 + 3 × 53 = 171, en 450 kost 3 + 3 × 81 = 246. Een gemist uur kost drie keer zoveel als een rit voor niets, dus komt de grens onder de 400 van een druk uur: de crew gaat iets vaker op pad, en mist minder drukke uren. Met andere kosten wint een andere threshold: de kosten zijn een keuze van het bedrijf, niet van het model.",
            "300": "Alleen 15 gemiste uren, maar 110 ritten voor niets. Wat kost dat samen, met 1 per rit en 3 per gemist uur?",
            "400": "Hetzelfde getal als een druk uur voelt goed. Wat kost het dan, met 53 gemiste uren van 3 elk?",
            "450": "Bijna geen ritten voor niets, maar hoeveel drukke uren mist de crew, met 3 per gemist uur?",
        },
    )
    choice_threshold
    return (choice_threshold,)


@app.cell(hide_code=True)
def text_exercise_sides():
    mo.md(r"""
    ### Oefening 4. Voor of na het model?

    Acht stappen van het serving-pad van het fietsdeelbedrijf. Zet elke stap aan de goede kant: naar het model toe, waar elke stap doet wat de training deed, of na het model, waar bedrijfsregels en een threshold van het getal een antwoord maken.
    """)
    return


@app.cell(hide_code=True)
def sort_sides_cell():
    sort_sides = L.sort("sides")
    sort_sides
    return (sort_sides,)


@app.cell(hide_code=True)
def check_sides(sort_sides):
    result_sides = L.check("sides", sort_sides)
    result_sides
    return (result_sides,)


@app.cell(hide_code=True)
def hints_sides():
    L.hints(
        "sides",
        [
            "Vraag bij elke stap: heeft het model de stap nodig om de input te lezen zoals bij de training, of verandert de stap wat eruit komt?",
            "Vier stappen gaan over wat erin gaat: de schaal, het uur, de eenheden en de volgorde van de features. Welke vier gaan over het getal dat eruit komt?",
            "Naar het model toe: standaardiseren, het uur, \\_\\_\\_ en \\_\\_\\_. Na het model: onder 0, de crew, \\_\\_\\_ en \\_\\_\\_.",
        ],
        why="**Naar het model toe:** standaardiseren, het uur, graden en de volgorde, want het model geeft alleen goede getallen voor input die zo is voorbereid als bij de training. **Na het model:** onder 0, de crew, 's nachts en hele fietsen, want het bedrijf bepaalt wat het getal betekent en wat nooit mag gebeuren.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### De nachtelijke batch

    Een nachtelijke batch bereidt de 24 uren van één dag van tevoren voor, en de planners handelen 's ochtends naar de beslissingen. Kies een dag uit de validation set, en waar de batch het gemiddelde en de standaardafwijking vandaan haalt: uit de trainingsuren, zoals in config.json; uit de 24 uren van de batch zelf; of nergens vandaan, zonder schaling. De afbeelding toont de verhuringen, wat het model maakt met de schaal van de training en met de schaal die je koos, en de uren waarin het gekozen pad de crew stuurt, bij de threshold van 350 uit Oefening 3.
    """)
    return


@app.cell(hide_code=True)
def demo_form():
    _w = C.DEMO_TEXT[L.lang]
    demo_day = mo.ui.dropdown(options={_w["days"][k]: k for k in C.DAYS}, value=_w["days"]["thu"], label="Dag")
    demo_source = mo.ui.radio(options={_w["sources"][k]: k for k in C.SOURCES}, value=_w["sources"]["training"],
                              label="Gemiddelde en standaardafwijking uit")
    mo.vstack([
        demo_day,
        demo_source,
        L.task("Kies voor elke dag de batch zelf, en daarna geen schaling. Op welke dag stuurt de batch helemaal geen crew? En hoe ver zit het pad zonder schaling over alle 73 dagen ernaast?"),
    ])
    return demo_day, demo_source


@app.cell(hide_code=True)
def demo_picture(demo_day, demo_source):
    mo.Html(C.day_picture(demo_day.value, demo_source.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Op donderdag 14 juli stuurt de schaal van de training de crew om 8:00, 17:00, 18:00 en 19:00: precies de vier drukke uren. De batch stuurt de crew alleen om 17:00 en 18:00, en voorspelt 169 fietsen voor 8:00, terwijl er 486 werden verhuurd. Op woensdag 3 augustus stuurt de batch helemaal geen crew, hoewel drie uren druk waren. Zonder schaling zit het model op een eigen manier ernaast, en stuurt het de crew in uren die niet druk zijn. Over alle 73 dagen zit de batch gemiddeld 73 fietsen per uur ernaast en het pad zonder schaling 120, waar de schaal van de training 34 ernaast zit. Geen van deze runs stopte met een fout.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Je model achter een API](/course/open/sys-serving-api?lang=nl) gaf de server door wat het model ook zei, zelfs een getal onder 0 fietsen, of 4.855 voor een temperatuur in graden. Deze les zette een pad rond het model. Naar het model toe wordt de input voorbereid zoals bij de training, gestandaardiseerd zoals in [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl) met het gemiddelde en de standaardafwijking uit de trainingsuren. Na het model maken bedrijfsregels en een decision threshold, gekozen uit kosten zoals in [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl), van het getal een antwoord. In de volgende les, *Een draaiend model in de gaten houden: metrics, logs en alerts*, houdt een service zichzelf in de gaten: wat de service krijgt, wat de service antwoordt, en wanneer iemand moet kijken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_threshold, result_decide, result_prepare, result_sides):
    L.wrap_up(result_prepare, result_decide, choice_threshold, result_sides)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
