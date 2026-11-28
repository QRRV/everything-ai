# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-code-tests. Route: course/route/lessen/systemen.yaml. Checks, de versies met een bug, de runs van de course op 16 uren, de demo en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-code-tests", "nl")
    C = L.checks   # de versies met een bug, de runs van de course op 16 uren, de checks van je tests, de demo en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Project: een model van de vraag, bijgehouden en ingepakt](/course/open/sys-project-tracked-model?lang=nl) en [Foutmeldingen lezen en bugs vinden](/course/open/py-errors?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_days():
    quick_days = L.choice(
        "quick-days",
        question="In je project ops bestaat de validation set uit elke vijfde dag van 2011, met alle uren van zo'n dag. Waarom hele dagen, en niet 1.733 willekeurige uren?",
        options={
            "faster": "Hele dagen maken de validation set kleiner, dus kost meten na elke epoch minder tijd",
            "right": "De uren van één dag delen weer en drukke momenten, dus moet een validatiedag nieuw zijn voor het model",
            "pandas": "Een mask op de dag is het enige soort mask dat pandas op een tabel met uren kan maken",
            "night": "Hele dagen houden de nachturen uit de training set, want midden in de nacht huurt bijna niemand een fiets",
        },
        correct="right",
        why={
            "right": "Met willekeurige uren leert het model van de andere uren van dezelfde dagen, dus zijn de validatie-uren makkelijker dan een nieuwe dag: de RMSE kwam ongeveer 7 fietsen lager uit.",
            "faster": "73 hele dagen hebben 1.733 uren, net zoveel als 1.733 willekeurige uren: wat delen de uren van één dag? Zie [Project: een model van de vraag, bijgehouden en ingepakt](/course/open/sys-project-tracked-model?lang=nl).",
            "pandas": "Een mask kan rijen kiezen op elke kolom, zoals het uur: wat delen de uren van één dag, waardoor splitsen per dag beter is? Zie [Project: een model van de vraag, bijgehouden en ingepakt](/course/open/sys-project-tracked-model?lang=nl).",
            "night": "Elke validatiedag houdt alle uren, ook de nachturen, en elke trainingsdag ook: wat delen de uren van één dag? Zie [Project: een model van de vraag, bijgehouden en ingepakt](/course/open/sys-project-tracked-model?lang=nl).",
        },
    )
    quick_days
    return (quick_days,)


@app.cell(hide_code=True)
def quick_check_errors():
    quick_errors = L.choice(
        "quick-errors",
        question="Een functie die je schreef draait zonder foutmelding. Wat zegt dat?",
        options={
            "answer": "Dat de antwoorden kloppen, want Python stopt bij elke fout in een functie met een foutmelding",
            "checked": "Dat de functie getest is, want Python probeert elke nieuwe functie eerst op een paar eigen inputs uit, nog voordat de functie draait",
            "right": "Alleen dat Python de functie kon uitvoeren: of de antwoorden kloppen, zeggen inputs waarvan je het antwoord kent",
            "types": "Dat elke waarde erin het goede type heeft, dus kunnen alleen de getallen er nog een beetje naast zitten",
        },
        correct="right",
        why={
            "right": "Een bug zonder foutmelding geeft stil een fout antwoord, zoals 0 als grootste getal van `[-3, -7]`, waar −3 bedoeld was. Met inputs waarvan je het antwoord kent, zie je de bug.",
            "answer": "Python stopt alleen waar Python niet verder kan, zoals bij een naam die Python niet kent: en een functie die draait en een fout getal geeft? Zie [Foutmeldingen lezen en bugs vinden](/course/open/py-errors?lang=nl).",
            "checked": "Python voert uit wat je schreef en probeert zelf niets: wie probeert de functie op inputs waarvan het antwoord bekend is? Zie [Foutmeldingen lezen en bugs vinden](/course/open/py-errors?lang=nl).",
            "types": "Python controleert geen types voordat een functie draait, en een fout antwoord kan het goede type hebben: hoe vind je een bug zonder foutmelding? Zie [Foutmeldingen lezen en bugs vinden](/course/open/py-errors?lang=nl).",
        },
    )
    quick_errors
    return (quick_errors,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Project: een model van de vraag, bijgehouden en ingepakt](/course/open/sys-project-tracked-model?lang=nl) trainde een versie van je model in een paar seconden. Een echt model traint uren of dagen, op machines die per uur geld kosten. De meeste bugs in trainingscode stoppen de run niet met een foutmelding: de code draait, de loss daalt een beetje, en het model komt slechter uit dan mogelijk is. Dat merk je aan het eind, als je het al merkt.

    Tests vinden zulke bugs in seconden, nog voordat de lange run begint. Engineers bij Google zetten in *The ML Test Score* (Breck en anderen, 2017) 28 tests en controles op een rij voor modellen in productie, van de data tot het model tot hoe het model draait. Deze les neemt drie tests in dezelfde geest, voor de code van je project *ops*: een test van één functie op één rij, een test dat een netwerk een paar uren uit het hoofd kan leren, en een controle die op elke voorspelling draait zodra het model in gebruik is.

    **Een voorbeeld uit het dagelijks leven.** Een bakker probeert een nieuwe oven eerst uit met één klein broodje, en bakt pas daarna honderd broden. Het broodje kost tien minuten, en komt het rauw of verbrand uit de oven, dan weet de bakker dat er iets mis is met de oven, zonder een dag meel weg te gooien.
    """)
    return


@app.cell(hide_code=True)
def predict_column():
    guess_column = L.guess(
        "column",
        question="In deze les traint een nieuw netwerk van 577 getallen 400 stappen op alleen 16 uren van 2011. Stel dat een bug in de trainingscode `y` verandert in een rij van 16 getallen, terwijl het netwerk een kolom van 16 geeft. Wat doet de loss op deze 16 uren, denk je?",
        options={
            "zero": "De loss daalt nog steeds bijna tot 0: het netwerk leert de 16 uren evengoed uit het hoofd",
            "right": "De loss daalt een tijdje, stopt dan ver boven 0 en blijft daar",
            "grow": "De loss groeit bij elke stap, tot ver boven de loss van het begin",
            "same": "De loss blijft op de beginwaarde, want het netwerk kan helemaal niet leren",
        },
        reveal="**De loss stopt ver boven 0.** De loss daalt binnen de eerste 100 stappen van 36.930 naar ongeveer 21.091 en blijft daar, terwijl de goede code de loss naar 0,99 brengt. Met `y` als rij geeft `model(x) - y` 16 × 16 verschillen: elke voorspelling tegen elk van de 16 uren. Het gemiddelde is het ene getal dat het dichtst bij alle 16 uren tegelijk ligt, dus leert het netwerk voor elk uur 125,9 fietsen te zeggen, en 21.091 is de variantie van de 16 uren: de gemiddelde gekwadrateerde afstand tot het gemiddelde. De run geeft geen foutmelding, en de loss daalt zelfs met meer dan een derde.",
    )
    guess_column
    return (guess_column,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een test is een functie met `assert`.** `assert` met een voorwaarde erachter doet niets als de voorwaarde True is, en stopt met een `AssertionError` als de voorwaarde False is. Een test is een functie met een naam die met `test` begint: de functie roept je code aan op een voorbeeld, en legt met `assert` vast wat eruit moet komen. pytest, de gebruikelijke tool voor tests in Python, vindt al zulke functies in de bestanden waarvan de naam met `test_` begint, en draait alle tests in één keer; dat doe je na elke wijziging, en in deze les draait de course je tests voor je. Een test slaagt als de functie zonder foutmelding eindigt, en faalt als een `assert`, of een andere fout, de functie stopt.

        **Een unit test: één functie, één voorbeeld dat je met de hand kunt uitrekenen.** Net als in [Foutmeldingen lezen en bugs vinden](/course/open/py-errors?lang=nl) test je een klein stuk met een input waarvan je het antwoord kent. Voor `inputs` uit je project is één rij genoeg: uur 8 moet een 1 geven in kolom 8 van de 24 uurkolommen, en een feature precies op het gemiddelde moet 0 geven. Je kiest de rij zo dat je elk van de 34 getallen kent.

        **Een goede test faalt als de code fout is.** Slagen voor de goede code is pas de helft: een test moet ook falen voor code met een bug. De checks van deze les proberen je tests op de versie van de course en op versies met bugs die in echte projecten voorkomen, en laten per versie zien of je test slaagt of faalt.

        **Shapes en bereiken.** Veel bugs in trainingscode zie je aan de shape van een tensor of aan een waarde die niet kan: een rij waar een kolom bedoeld was, een NaN, een negatief aantal fietsen. Een regel zoals `assert pred.shape == y.shape` net boven de loss kost niets en vangt de bug van de gok hierboven, zoals het uitgewerkte voorbeeld laat zien.

        **Overfit one batch.** Een netwerk van 577 getallen kan 16 uren uit het hoofd leren. Train je het netwerk op alleen deze uren, met niets anders om te leren, dan moet de loss dicht bij 0 komen. Lukt dat niet, dan kan het netwerk zelfs het makkelijkste niet leren, en de oorzaak is bijna altijd een bug: een verkeerde shape, een stap die nooit gebeurt, gradients die zich opstapelen. De test duurt een seconde, terwijl een lange run op alle data de bug verbergt achter getallen die geloofwaardig lijken. Andrej Karpathy maakte deze test bekend met *A Recipe for Training Neural Networks* (2019).
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        De afbeelding laat de gok zien: met de goede code daalt de loss tot ongeveer 1, en met `y` als rij stopt de loss bij 21.091. Op een schaal waarop elke stap omhoog tien keer zoveel is, passen beide lijnen in één grafiek.

        **Een regression test bewaakt een score.** Je ingepakte model heeft op de validatiedagen een RMSE van 53,04 fietsen per uur. Een regression test legt na elke wijziging vast dat de RMSE onder een vaste grens blijft: maakt een wijziging het model slechter, dan faalt de test. Een regression is hier iets wat werkte en weer kapotging, niet de taak om een getal te voorspellen.

        **Een model assertion draait terwijl het model in gebruik is.** Tests draaien voordat het model de deur uit gaat; een model assertion kijkt naar elke voorspelling zodra het model in gebruik is, en markeert de voorspellingen die niet kunnen kloppen. Het model van de course in je project verwacht −10,8 fietsen om 4 uur 's nachts op 14 juli 2011, en minder dan 0 fietsen in 235 van de 8.645 uren van 2011, allemaal tussen middernacht en 5 uur 's nachts. Afronden naar 0 verbergt deze uren; een model assertion telt de uren, zodat iemand gaat kijken, en de gemarkeerde uren zijn later goede voorbeelden om van te leren. Daniel Kang en anderen gaven het idee in 2020 deze naam.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **unit test** | | een kleine test van één functie, op een voorbeeld waarvan je het antwoord kent | `def test_one_row():` |
        | `assert` | | doet niets als de voorwaarde erachter True is; stopt met een `AssertionError` als de voorwaarde False is | `assert x.shape == (1, 34)` |
        | **overfit one batch** | | een nieuw netwerk op alleen een paar voorbeelden trainen: de loss moet dicht bij 0 komen, anders zit er een bug in de code | `train_epoch(model, opt, x, y, 16)` |
        | **regression test** | | een test dat iets wat eerst werkte na een wijziging nog werkt, zoals een score die niet slechter mag worden | `assert rmse <= 55` |
        | **model assertion** | | een controle tijdens het gebruik van het model: markeert elke voorspelling die niet kan kloppen | `impossible(pred)` |
        | **NaN** | nan | not a number: wat `torch.tensor(0.0) / 0` geeft, en hoe een tabel een ontbrekende waarde markeert; NaN is aan niets gelijk, zelfs niet aan NaN | `float("nan")` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Tests voor ML-code:

    1. Kies voor elke functie een voorbeeld waarvan je het antwoord met de hand kunt uitrekenen, zoals één rij, en leg met `assert` vast wat eruit moet komen: een unit test.
    2. Zorg dat elke test faalt voor code met een bug: probeer de test op een versie met een bug die je kent, en zie de test falen.
    3. Leg shapes en bereiken met `assert` vast, nog voordat een lange run begint: voorspellingen en targets met dezelfde shape, geen NaN, geen waarde die niet kan.
    4. Overfit één kleine batch: train een nieuw netwerk op alleen een paar voorbeelden, en de loss moet dicht bij 0 komen.
    5. Bewaak de score met een regression test: een vaste grens iets boven de score die je nu hebt.
    6. Markeer met een model assertion elke voorspelling die niet kan kloppen zodra het model in gebruik is, en tel de gemarkeerde voorspellingen.
    7. Draai alle tests na elke wijziging.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Vier uren, net als in het uitgewerkte voorbeeld van [Project: een model van de vraag, bijgehouden en ingepakt](/course/open/sys-project-tracked-model?lang=nl): het netwerk zegt 30, 40, 220 en 110 fietsen, als een kolom met shape (4, 1), en er waren er 10, 40, 250 en 100. Is `y` ook een kolom, dan heeft `pred - y` de shape (4, 1): de verschillen 20, 0, −30 en 10, de kwadraten 400, 0, 900 en 100, en de loss is het gemiddelde van de kwadraten, 350.

    Is `y` een rij met shape (4,), dan maakt broadcasting van `pred - y` een tabel met shape (4, 4): elk van de 4 voorspellingen min elk van de 4 aantallen fietsen, 16 verschillen, en de loss is het gemiddelde van de 16 kwadraten: 14.300. Nu beloont de loss het verkeerde. Zegt een netwerk precies 10, 40, 250 en 100, dan krijgt het netwerk 17.100, en zegt een netwerk voor elk uur 100, het gemiddelde, dan krijgt het netwerk 8.550: het laagste wat een netwerk kan krijgen, de variantie van de vier uren. Dus trekt de training elke voorspelling naar het gemiddelde, net als in de gok.

    De regel `assert pred.shape == y.shape` ziet (4, 1) en (4,), en faalt nog voordat de eerste stap begint.
    """)
    return


@app.cell(hide_code=True)
def text_few():
    mo.md(r"""
    ### De 16 uren

    De test van oefening 3 traint op 16 uren van 2011: `.sample(16, random_state=0)` kiest de uren uit alle 8.645, elke keer dezelfde 16. De cel leest de uren met `read_hours` uit je project, en print met `.to_string(index=False)` de datum, het uur en de fietsen van elk van de 16, en dan met `.min()` de minste fietsen, met `.max()` de meeste en met `.mean()` het gemiddelde. Voer de cel uit.
    """)
    return


@app.cell
def example_few():
    from ops.demand import read_hours

    hours = read_hours()
    few = hours.sample(16, random_state=0)   # elke keer dezelfde 16 uren
    print(few[["dteday", "hr", "cnt"]].to_string(index=False))
    print("van", few["cnt"].min(), "tot", few["cnt"].max(), "fietsen, gemiddeld", few["cnt"].mean())
    return


@app.cell(hide_code=True)
def text_exercise_one_row():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een unit test van één rij

        `test_one_row()` test `inputs` uit je project, dat uit `ops.demand` komt, samen met `FEATURES`, de namen van de 10 features. De regels van de test staan er al, en de test importeert `pandas` als `pd`. `scale` bevat voor elke feature een verzonnen gemiddelde en standaardafwijking, in de volgorde van `FEATURES`. `zip` koppelt elke naam aan het gemiddelde, en `dict` maakt er een rij van: elke feature precies op het gemiddelde. Dan wordt `row["temp"]` 0,75, één standaardafwijking van 0,25 boven het gemiddelde van 0,5, en wordt het uur `row["hr"]` 8. `pd.DataFrame([row])` maakt een tabel van deze ene rij, en `inputs` geeft `x`, een tensor met shape (1, 34). De laatste regel legt `x.shape` vast en slaagt voor elke versie van `inputs` die de check probeert.

        Zet er regels met `assert` onder, net zo ver ingesprongen als de andere, en leg zo vast wat er in twee kolommen van `x` moet staan. `x[0, 8]` is het getal in kolom 8 van de eerste rij. De eerste 24 kolommen zijn de uren 0 tot en met 23, one-hot; daarna komen de 10 features in de volgorde van `FEATURES`: weekday, workingday, holiday, weathersit, temp, atemp, hum, windspeed, season en mnth, elk gestandaardiseerd: min het gemiddelde, gedeeld door de standaardafwijking.

        De check draait `test_one_row()` op de goede `inputs`, waar de test moet slagen, en op vier versies met een bug, waar de test moet falen: uur 8 in kolom 9, de features eerst en dan de uren, het gemiddelde niet eraf, en het gemiddelde van de gegeven uren eraf in plaats van het gemiddelde in `scale`. Onder de check zie je hoe je test het op elke versie doet. Is de test goed, dan bewaart de check je test als `work/projects/ops/test_inputs.py`.
        """),
        L.task("Voeg regels met `assert` toe, zodat je test alle vier versies met een bug vangt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_one_row():
    box_one_row = L.code_box("one-row", lines=14)
    box_one_row
    return (box_one_row,)


@app.cell(hide_code=True)
def check_one_row(box_one_row):
    result_one_row = L.check("one-row", box_one_row.value)
    result_one_row
    return (result_one_row,)


@app.cell(hide_code=True)
def hints_one_row():
    L.hints(
        "one-row",
        [
            "Uur 8 is one-hot: één van de 24 uurkolommen van `x` bevat 1, de andere 23 bevatten 0. De features komen na de 24 uurkolommen, in de volgorde van `FEATURES`. Een regel met `assert` en `==` legt één getal van `x` vast.",
            "temp is de vijfde feature in `FEATURES`, op index 4. Voor deze rij staat elke feature behalve temp precies op het gemiddelde: (waarde − gemiddelde) / standaardafwijking = 0. temp geeft (0,75 − 0,5) / 0,25. De kolommen van de features zijn 24 tot en met 33.",
            "`    assert x[0, ___] == 1`",
        ],
        why="`assert x[0, 8] == 1` en `assert x[0, 28] == 1`, allebei ingesprongen in `test_one_row`: uur 8 in kolom 8, en temp, (0,75 − 0,5) / 0,25 = 1, in kolom 24 + 4 = 28.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_impossible():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een model assertion

        `impossible(pred)` krijgt een tensor met voorspellingen, met een willekeurige shape, en geeft een mask met dezelfde shape terug: True voor elke voorspelling die niet kan kloppen, False voor de andere. Twee soorten kunnen niet kloppen: minder dan 0 fietsen, en NaN, wat een uur geeft als een van de inputs ontbreekt, zoals de luchtvochtigheid. 0 fietsen kan wel, laat in de nacht. Een vergelijking zoals `pred < 0` geeft een mask. `torch.isnan(t)` is True waar de tensor `t` NaN bevat; `==` kan NaN niet vinden, want NaN is aan niets gelijk. `|` combineert twee masks: True waar minstens één van de twee True is. Python doet `|` eerder dan `<`, dus geef elke vergelijking eigen haakjes. Vervang de `...`.

        De check roept `impossible(PRED)` aan, met `PRED` de 24 voorspellingen van het model van de course voor donderdag 14 juli 2011; het mask moet True zijn op de plekken 3 en 4, de uren waarvoor het model −6,3 en −10,8 fietsen verwacht. Daarna probeert de check drie eigen tensors: één met een NaN, een kolom, en één met nullen. Is de functie goed, dan bewaart de check je functie als `work/projects/ops/assertions.py`.
        """),
        L.task("Maak `impossible` True voor elke voorspelling onder 0 en voor elke NaN."),
    ])
    return


@app.cell(hide_code=True)
def exercise_impossible():
    box_impossible = L.code_box("impossible", lines=5)
    box_impossible
    return (box_impossible,)


@app.cell(hide_code=True)
def check_impossible(box_impossible):
    result_impossible = L.check("impossible", box_impossible.value)
    result_impossible
    return (result_impossible,)


@app.cell(hide_code=True)
def hints_impossible():
    L.hints(
        "impossible",
        [
            "Twee masks, elk uit één vergelijking op `pred`: één voor minder dan 0 fietsen, en één uit `torch.isnan`. `|` zet de twee samen: True waar een van de twee True is. Elke vergelijking komt tussen eigen haakjes.",
            "NaN is zelfs niet gelijk aan NaN: `float(\"nan\") == float(\"nan\")` is False, dus vindt `==` NaN nooit. In `PRED` liggen alleen de uren 3 en 4 onder 0, en `PRED` bevat geen NaN; de check probeert ook een tensor met een NaN.",
            "`    return (pred < 0) | ___`",
        ],
        why="`return (pred < 0) | torch.isnan(pred)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_overfit():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Overfit one batch

        `test_overfit_one_batch()` test `train_epoch` uit je project, en importeert `train_epoch` met de andere onderdelen uit `ops.demand`. De regels van de test staan er al. `read_hours` leest de uren van 2011 in `hours`, en `.sample(16, random_state=0)` zet de 16 uren van de cel hierboven in `few`. `inputs` en `targets` maken er `x` en `y` van, met `scaling` op alle uren. `torch.manual_seed(0)` maakt het begin elke keer hetzelfde, `DemandNet(16)` is een nieuw netwerk van 577 getallen, en `torch.optim.Adam` met `lr=0.03`, een learning rate van 0,03, is de optimizer, die de getallen van het netwerk uit `model.parameters()` krijgt, net als in [Project: een model van de vraag, bijgehouden en ingepakt](/course/open/sys-project-tracked-model?lang=nl). `first` is de loss van het netwerk dat nog niets geleerd heeft: de mean squared error, met `.mean()` over de gekwadrateerde verschillen, en `.item()` maakt er een Python-getal van. `for _ in range(400)` voert `train_epoch` 400 keer uit; met een batch van 16 is elke keer één stap op alle 16 uren. `last` is de loss daarna.

        Vervang de `...` na `assert` door een voorwaarde op `first` en `last`. De goede `train_epoch` brengt de loss van 36.930 naar onder de 1. De check draait je test op deze versie, waar de test moet slagen, en op drie versies met een bug, waar de test moet falen: één zonder `opt.step()`, één die `model(x)` met `y` als rij vergelijkt, en één zonder `opt.zero_grad()`. Elke run duurt een fractie van een seconde. Is de test goed, dan bewaart de check je test als `work/projects/ops/test_training.py`.
        """),
        L.task("Laat de `assert` falen, tenzij de loss dicht bij 0 komt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_overfit():
    box_overfit = L.code_box("overfit", lines=16)
    box_overfit
    return (box_overfit,)


@app.cell(hide_code=True)
def check_overfit(box_overfit):
    result_overfit = L.check("overfit", box_overfit.value)
    result_overfit
    return (result_overfit,)


@app.cell(hide_code=True)
def hints_overfit():
    L.hints(
        "overfit",
        [
            "`first` en `last` zijn getallen: de loss aan het begin en aan het eind. Een voorwaarde zoals `last < first` vraagt alleen dat de loss daalt, en van een netwerk dat 16 uren uit het hoofd kan leren, mag je veel meer verwachten. Vermenigvuldig `first` met een klein getal om een grens te maken.",
            "Met `y` als rij stopt de loss bij 21.091, 57% van het begin, en zonder `opt.zero_grad()` eindigt de loss op 28.134. De goede versie eindigt op 0,99, minder dan een duizendste van 36.930.",
            "`    assert last < ___ * first`",
        ],
        why="`assert last < 0.01 * first`: onder 1% van het begin. Elke grens van ongeveer 0,0001 tot 0,5 keer `first` slaagt hier voor de goede versie en faalt voor alle drie de bugs.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Drie versies van train_epoch, en je test op elke versie

    De knop traint drie keer een nieuw netwerk op de 16 uren, elke keer 400 stappen, met drie versies van `train_epoch`: de goede, de versie met `y` als rij, en de versie zonder `opt.zero_grad()`. De grafiek toont de loss van de versie die nu traint, met de versies die al klaar zijn als dunne lijnen van streepjes. Ernaast zie je wat het netwerk voor elk van de 16 uren zegt, van het rustigste naar het drukste, tegen de fietsen die echt gehuurd zijn. Daarna draait je test uit oefening 3 op elke versie. Is de run klaar, dan kun je door de stappen terugbladeren.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    run_button = mo.ui.run_button(label="Train drie versies op 16 uren")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def demo_run(result_overfit, run_button):
    result_overfit   # opnieuw met je test zodra de check de test bewaart
    if not run_button.value:
        _out = mo.md("*Druk op de knop om de drie versies te trainen.*")
    else:
        _now = {}

        def _show(_name, _step, _loss, _preds, _done):
            if _now.get("name") != _name:   # een nieuwe versie: een eigen grafiek, met de versies die al klaar zijn
                _chart = L.live(C.live_labels(_name, L.lang), C.STEPS, role="loss", low=C.LOW, high=C.HIGH, window=1)
                for _other, _points in _done:
                    _chart.reference(_points, C.version_name(_other, L.lang), role="loss")
                _chart.watch(lambda: C.now_picture(_now, L.lang))
                _now["chart"], _now["name"], _now["preds"] = _chart, _name, []
            _now["step"] = _step
            if _preds is not None:
                _now["preds"] = _preds
            _now["chart"].add(_step, _loss, C.progress_line(_name, _step, _loss, L.lang))

        _r = C.demo(C.saved_test(), show=_show)
        if isinstance(_r, str):
            _note = C.slowed(L.lang)
            _out = _now["chart"].stopped(_note) if "chart" in _now else mo.md(_note)
        else:
            _out = mo.vstack([
                L.explore("rewind", "steps", lambda _v: C.rewind_frame(_r, int(_v), L.lang), start=C.STEPS,
                          low=0, high=C.STEPS, step=10, role="",
                          task="Ga naar stap 50, dan naar 100 en naar 400. Waar staan de stippen van elke versie bij stap 400, en tussen welke stappen gaan de goede versie en de versie zonder `opt.zero_grad()` uiteen?"),
                mo.md(C.verdicts(_r, L.lang)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.vstack([
        mo.md(r"""
        De goede versie brengt de loss van 36.930 naar 0,99: na 400 stappen staat bijna elke stip op de staaf, een netwerk dat de 16 uren uit het hoofd kent. Met `y` als rij schuiven alle stippen naar één hoogte, 125,9 fietsen, het gemiddelde van de 16 uren, en blijft de loss op 21.091. Zonder `opt.zero_grad()` lijken de eerste 50 stappen op de goede, 11.725 tegen 11.481; dan stapelen de gradients van alle eerdere stappen zich op, duwt elke stap verder in de richting van de eerdere stappen, en springt de loss tussen 5.439 en 28.134. Met een test die alleen naar stap 50 keek, was deze bug je ontgaan.

        Geen van de drie versies stopte met een foutmelding. De check van oefening 3 probeert ook de versie zonder `opt.step()`; deze versie zou een vlakke lijn tekenen: de loss blijft op 36.930.
        """),
        L.watch_out("Een batch overfitten laat zien dat de code kan leren, niet dat het model goed is. Over een nieuwe dag zegt een netwerk dat 16 uren uit het hoofd kent niets: dat meet de validation set."),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_regression():
    mo.md(r"""
    ### Oefening 4. Een regression test voor de score

    Je ingepakte model heeft op de validatiedagen een RMSE van 53,04 fietsen per uur. Na elke wijziging aan de code traint een regression test het model opnieuw en legt met `assert` iets vast over de nieuwe `rmse`.
    """)
    return


@app.cell(hide_code=True)
def choice_regression_cell():
    choice_regression = L.choice(
        "regression",
        question="Welke `assert` bewaakt dat de score niet slechter wordt, zonder te falen als er niets mis is?",
        options={
            "exact": "`assert rmse == 53.04`: de score moet precies blijven wat de score nu is",
            "baseline": "`assert rmse <= 132.8`: niet slechter dan een model dat altijd het gemiddelde van de trainingsuren zegt",
            "right": "`assert rmse <= 55`: een vaste grens iets boven de score die je nu hebt",
            "better": "`assert rmse < last_rmse`: elke nieuwe run moet de vorige run verslaan",
        },
        correct="right",
        why={
            "right": "53,04 schuift een beetje zonder dat er iets mis is: met de seeds 1 en 2 eindigt dezelfde versie op 52,8 en 53,5. Een grens van 55 laat dat door, en faalt als een wijziging het model duidelijk slechter maakt, zoals een wijziging die de score op 79,1 brengt, niet beter dan het gemiddelde van hetzelfde uur op hetzelfde soort dag.",
            "exact": "Een andere seed, of een onschuldige wijziging aan de code, verschuift 53,04 een paar tienden: de test zou falen zonder dat er iets mis is. Wat moet een test van een score toelaten?",
            "baseline": "Ook een model dat 100 fietsen ernaast zit slaagt dan nog: de test vangt alleen een model dat slechter is dan voor elk uur dat gemiddelde zeggen. Hoe ver mag de score stijgen voordat er iets mis is?",
            "better": "Twee runs zonder fout verschillen een beetje, de ene kant op of de andere, dus faalt deze test op ongeveer de helft van de runs. Onder welk vast getal moet de score blijven?",
        },
    )
    choice_regression
    return (choice_regression,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Foutmeldingen lezen en bugs vinden](/course/open/py-errors?lang=nl) testte je kleine stukken met inputs waarvan je het antwoord kende; hier werden zulke tests functies die pytest kan draaien, en elke test moet ook falen voor code met een bug. Je project *ops* uit [Project: een model van de vraag, bijgehouden en ingepakt](/course/open/sys-project-tracked-model?lang=nl) heeft er nu drie: `test_inputs.py`, `test_training.py` en de model assertion in `assertions.py`. Tests van code controleren wat de code doet; de volgende les, *Doet het model wat het moet? Behavioral tests*, controleert wat het model doet, zoals meer fietsen bij mooi weer. In *Project: tests en een model card* krijgt je model van de vraag allebei, en in *Project: je model als service, bewaakt op drift* beantwoordt het model verzoeken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_regression, result_impossible, result_one_row, result_overfit):
    L.wrap_up(result_one_row, result_impossible, result_overfit, choice_regression)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
