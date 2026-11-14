# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson data-synthetic. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("data-synthetic", "nl")
    C = L.checks   # de 10.000 metingen van AI4I, het logboek en de test, de gesimuleerde metingen, k-NN, de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Verbeter de data, niet het model](/course/open/data-centric-ai?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_fixed():
    quick_fixed = L.choice(
        "quick-fixed",
        question="Data-centric AI houdt het ene hetzelfde en verbetert het andere. Wat is wat?",
        options={
            "model": "De data blijft hetzelfde, en je verbetert het model",
            "test": "Model en data blijven hetzelfde; alleen de test set verandert",
            "right": "Het model blijft hetzelfde, en je verbetert de data",
        },
        correct="right",
        why={
            "model": "Dat is de modelgerichte aanpak: bijvoorbeeld een andere k kiezen. Zie *Het idee* in [Verbeter de data, niet het model](/course/open/data-centric-ai?lang=nl).",
            "test": "De test set blijft hetzelfde, zodat je de getallen ervoor en erna kunt vergelijken. Zie *Het idee* in [Verbeter de data, niet het model](/course/open/data-centric-ai?lang=nl).",
            "right": "Goed: labels verbeteren maakte de data beter. Vandaag maak je haar beter door voorbeelden toe te voegen die ontbraken.",
        },
    )
    quick_fixed
    return (quick_fixed,)


@app.cell(hide_code=True)
def quick_check_vote():
    quick_vote = L.choice(
        "quick-vote",
        question="Met 41 van de 274 labels van de pinguïns verkeerd zat k = 1 bij 14 testpinguïns fout en k = 9 bij maar 3. Waarom heeft k = 9 er minder last van?",
        options={
            "right": "De meeste van de 9 dichtstbijzijnde hebben het goede label, en die stemmen een verkeerd label weg",
            "fixed": "k = 9 verbetert de verkeerde labels tijdens het trainen",
            "fewer": "k = 9 bewaart minder pinguïns, en dus minder verkeerde labels",
        },
        correct="right",
        why={
            "fixed": "k-NN verandert geen enkel label: het stemt alleen met de labels die het heeft. Zie *Zelf doen* in [Verbeter de data, niet het model](/course/open/data-centric-ai?lang=nl).",
            "fewer": "Elke k bewaart alle pinguïns uit de training set; k zegt alleen hoeveel ervan stemmen. Zie [Verbeter de data, niet het model](/course/open/data-centric-ai?lang=nl).",
            "right": "Goed: in een stemming van 9 valt een verkeerd label weg tussen de goede. Vandaag stemt k = 5 tussen twee klassen, storing of niet: met een oneven k eindigt een stemming nooit gelijk.",
        },
    )
    quick_vote
    return (quick_vote,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Sommige voorbeelden zijn zeldzaam omdat ze slecht nieuws zijn: een machine die uitvalt, fraude, een zeldzame ziekte. Je kunt niet wachten op honderd storingen voordat je er een model op traint. Als er te weinig data is, kun je er meer maken.

    Deze les gebruikt AI4I 2020: 10.000 metingen van een freesmachine, met temperaturen, toerental, koppel en storingen. AI4I is zelf gemaakt: Stephan Matzka (2020) bouwde het met een simulatie die op echte onderhoudsdata lijkt, en schreef de regels op waarmee de machine uitvalt. Hier doe je alsof de metingen van een echte machine komen. De eerste 3000 metingen zijn je logboek, en bij geen enkele viel de machine uit door hitte. De volgende 2000, in een warmere periode, bevatten 115 hittestoringen.

    **Een voorbeeld uit het dagelijks leven.** Piloten oefenen een motorbrand in een vluchtsimulator. Wachten op een echte zou een slechte manier zijn om het te leren.
    """)
    return


@app.cell(hide_code=True)
def predict_missed_cell():
    guess_missed = L.guess(
        "missed-guess",
        question="Je voegt 1000 gesimuleerde metingen toe aan het logboek, gelabeld met een regel voor hittestoringen, en traint k-NN met k = 5 op het logboek en de gesimuleerde metingen samen. Hoeveel van de 115 echte hittestoringen in de volgende 2000 metingen mist het?",
        options={
            "all": "Bijna alle 115: verzonnen metingen leren niets over de echte machine",
            "right": "Ongeveer 10",
            "half": "Ongeveer de helft",
        },
        reveal="**Ongeveer 10.** Met de gesimuleerde metingen van de les mist k-NN er 10 van de 115 en geeft het 9 keer vals alarm; met andere gesimuleerde metingen, willekeurig getrokken, mist het er 2 tot 16. Met alleen het logboek mist het alle 115: daarin viel de machine nooit uit door hitte.",
    )
    guess_missed
    return (guess_missed,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Drie manieren om data te maken.** Verander voorbeelden die je hebt: een foto van een kat, een beetje gedraaid of kleiner uitgesneden, is nog steeds een kat; een latere les doet dat met beelden. Doe het proces na dat de data maakt: een **simulatie** is een programma dat nadoet hoe de data ontstaat, zodat het voorbeelden kan maken, ook van gevallen die je nooit hebt gemeten. Of laat een model voorbeelden en hun labels schrijven: een language model kan vragen en antwoorden schrijven, zoals het domein over taal laat zien. Wat je zo maakt, is **synthetische data**. Vandaag simuleer je: een meting van de machine is een temperatuurverschil en een toerental.

        **Een label van een regel.** Een gesimuleerde meting heeft een label nodig, en krijgt dat van een regel: de regel die de maker van AI4I opschreef voor een hittestoring. Als het verschil tussen proces- en luchttemperatuur onder 8,6 K ligt en het toerental onder 1380 rpm, kan de warmte niet weg, en valt de machine uit. Een label dat een regel geeft, is een **synthetisch label**. Het is maar zo goed als de regel.

        **Zoals de echte data waar het ertoe doet.** Gesimuleerde metingen helpen alleen als ze op de punten die ertoe doen op echte lijken. Hier nemen ze hun toerental uit het logboek: toerentallen waarop de machine echt draait. Alleen het temperatuurverschil gaat waar het logboek nooit kwam: tussen 7,5 en 9,5 K, terwijl het logboek boven 8,6 K blijft.

        **Test alleen op echte data.** Een test set van gesimuleerde metingen krijgt zijn labels van dezelfde regel. Die toetst of het model de regel heeft geleerd, niet of de regel klopt. Alleen echte metingen, niet gebruikt om te trainen of te simuleren, zeggen hoe het model het doet op de machine.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **synthetische data** | data die een programma maakte, in plaats van data uit de wereld | `_sim_diff` |
        | **simulatie** | een programma dat het proces nadoet dat de data maakt | |
        | **synthetisch label** | een label dat een regel of een programma aan een voorbeeld geeft, in plaats van de echte uitkomst | `_sim_heat` |
        | `start + width * rng.random(n)` | `n` willekeurige getallen van `start` tot `start + width`: `rng.random(n)`, kort voor `rng.random(size=n)`, geeft getallen tussen 0 en 1, vermenigvuldigen rekt ze op tot de breedte, en `start` erbij optellen schuift ze omhoog | `7.5 + 2 * _rng.random(1000)` |
        | een `if` binnen een `if` | de binnenste `if` doet alleen mee als de buitenste voorwaarde klopt, dus moeten ze allebei waar zijn | `if _sim_diff[_i] < 8.6:` met daaronder `if _sim_speed[_i] < 1380:` |
        """),
        L.watch_out("De regel werkt hier zo goed omdat het de regel is waarmee AI4I gemaakt is. Een regel van een echte ingenieur is een gok, dichtbij of er een beetje naast, en een synthetisch label is nooit beter dan zijn regel."),
        mo.md(r"""
        Met de slider kies je de toerentalgrens van de regel: de 1000 gesimuleerde metingen krijgen er hun labels van, en k-NN traint erop en op het logboek. Links de hittestoringen die k-NN mist en hoe vaak het vals alarm geeft op de volgende 2000 echte metingen; rechts hetzelfde op 1000 andere gesimuleerde metingen, met labels van dezelfde regel. Schuif de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_limit_cell():
    def _frame(limit):
        said = C.said(limit, L.lang)
        alt = ("Links: de hittestoringen die k-NN mist en hoe vaak het vals alarm geeft op de volgende 2000 echte metingen, voor elke "
               "toerentalgrens van de regel. Rechts: hetzelfde op 1000 andere gesimuleerde metingen, met labels van dezelfde "
               "regel. " + said)
        return (C.picture(limit, C.words(L.lang) | {"alt": alt}, L.lang), said)

    explore_limit = L.explore("limit", "toerentalgrens (rpm)", _frame, start=1380, values=C.LIMITS, role="hyper",
                              task="Schuif de slider van 1260 naar 1500. Bij welke grens doet k-NN het goed op de echte metingen, en bij welke op de gesimuleerde?")
    explore_limit
    return


@app.cell(hide_code=True)
def text_after_limit():
    mo.md(r"""
    Op de gesimuleerde testmetingen maakt k-NN bij elke grens 5 tot 15 fouten: het heeft de regel geleerd, wat die ook zegt. Op de echte metingen werkt alleen 1380 rpm, de grens waarmee AI4I gemaakt is: 10 gemist, 9 keer vals alarm. Bij 40 rpm lager maakt de simulatie te weinig storingen, en mist k-NN er 76 van de 115; bij 40 rpm hoger noemt het metingen die goed gingen een storing, met 83 keer vals alarm. Een test op gesimuleerde metingen had elk van deze regels goed genoemd.

    Waarom gebruik je niet gewoon de regel? Dat kan hier: met twee features en één soort storing doet de regel het werk alleen. Gesimuleerde metingen verdienen hun plek als een model ook moet leren wat het logboek laat zien en geen regel die je kent beschrijft. Bij een echte machine is dat het meeste; in AI4I kwam zelfs dat uit regels, dus houdt deze les het bij één soort storing, zodat je ziet wat de simulatie doet.

    ### Recept

    Als je data te weinig voorbeelden heeft van wat ertoe doet:

    1. Schrijf op wat je weet over hoe die data ontstaat: hier de regel voor een hittestoring.
    2. Trek willekeurig metingen waar de machine kan komen, ook waar je logboek er geen heeft, en neem uit het logboek over wat het goed heeft gemeten, zoals de toerentallen.
    3. Geef elke gesimuleerde meting een label met de regel: synthetische labels.
    4. Train op je echte data en de gesimuleerde metingen samen.
    5. Test alleen op echte data, die niet diende voor de training en niet voor de simulatie.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel leest AI4I, neemt de eerste 3000 metingen als logboek en de volgende 2000 als test, en traint k-NN met k = 5 twee keer: op alleen het logboek, en op het logboek met 1000 gesimuleerde metingen. Elke keer print ze hoeveel van de 115 hittestoringen k-NN mist en hoe vaak het vals alarm geeft. Voer de cel uit.
    """)
    return


@app.cell
def example_simulate():
    import numpy as np
    import pandas as pd

    _df = pd.read_csv("data/ai4i-2020.csv.gz")
    _temps = np.array(_df[["Process temperature [K]", "Air temperature [K]"]])
    _diff = _temps[:, 0] - _temps[:, 1]              # procestemperatuur min luchttemperatuur, in K
    _speed = np.array(_df["Rotational speed [rpm]"])
    _heat = np.array(_df["HDF"])                     # 1 bij een hittestoring


    def _runs(diff, speed):
        # één rij voor elke meting: het temperatuurverschil en het toerental
        runs = np.zeros((len(diff), 2))
        runs[:, 0] = diff
        runs[:, 1] = speed
        return runs


    def _knn(train, labels, runs, k):
        # k-NN op gestandaardiseerde waarden: 1 als de meeste van de k dichtstbijzijnde uitvielen
        mean, std = train.mean(axis=0), train.std(axis=0)
        out = []
        for x in runs:
            dists = np.linalg.norm((train - mean) / std - (x - mean) / std, axis=1)
            nearest = np.argsort(dists, kind="stable")[:k]
            out.append(int(np.mean(labels[nearest]) > 0.5))
        return np.array(out)


    _log = _runs(_diff[:3000], _speed[:3000])
    _test, _real = _runs(_diff[3000:5000], _speed[3000:5000]), _heat[3000:5000]

    # 1. alleen het logboek: 3000 metingen, geen enkele hittestoring
    _pred = _knn(_log, _heat[:3000], _test, 5)
    print("alleen logboek:", np.sum(_real[_pred == 0]), "gemist,", np.sum(_pred[_real == 0]), "keer vals alarm")

    # 2. simuleer 1000 metingen: een temperatuurverschil tussen 7,5 en 9,5 K, een toerental uit het logboek
    _rng = np.random.default_rng(4)
    _sim_diff = 7.5 + 2 * _rng.random(1000)
    _sim_speed = _speed[:3000][_rng.integers(0, 3000, 1000)]

    # 3. geef ze een label met de regel: synthetische labels
    _sim_heat = np.zeros(1000)
    for _i in range(1000):
        if _sim_diff[_i] < 8.6:
            if _sim_speed[_i] < 1380:
                _sim_heat[_i] = 1
    print("gesimuleerde hittestoringen:", int(np.sum(_sim_heat)))

    # 4. train op het logboek en de gesimuleerde metingen samen, test op de volgende 2000 echte metingen
    _train = np.concatenate([_log, _runs(_sim_diff, _sim_speed)])
    _labels = np.concatenate([_heat[:3000], _sim_heat])
    _pred = _knn(_train, _labels, _test, 5)
    print("logboek en simulatie:", np.sum(_real[_pred == 0]), "gemist,", np.sum(_pred[_real == 0]), "keer vals alarm")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met alleen het logboek mist k-NN alle 115: bij geen enkele meting daarin viel de machine uit door hitte, dus stemt geen buur ooit voor een storing. Met 1000 gesimuleerde metingen, waarvan er 76 volgens de regel uitvallen, mist k-NN er 10 en geeft het 9 keer vals alarm. Alle 19 fouten liggen aan de rand van de regel, met een temperatuurverschil bij 8,6 K of een toerental bij 1380 rpm, waar metingen die goed gingen en metingen met een storing buren zijn.

    Waarom vanaf 7,5 K? Dat is een gok hoe ver een warme dag het verschil omlaag kan drukken. De precieze ondergrens doet er minder toe dan de regel: vanaf 6,5 of 8,0 K mist k-NN over twintig willekeurige simulaties 2 tot 26 van de 115, tegenover 2 tot 16 vanaf 7,5 K.

    `_speed[:3000][_rng.integers(0, 3000, 1000)]` pakt 1000 toerentallen uit het logboek, op willekeurige posities; hetzelfde toerental kan twee keer gekozen worden.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_power():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een regel voor vermogensstoringen

        De maker van AI4I schreef voor elke soort storing een regel. Een vermogensstoring: het vermogen dat het gereedschap nodig heeft, koppel maal toerental, ligt onder 3500 W of boven 9000 W. `power_failure(torque, speed)` rekent het vermogen in watt al uit. Vervang de `...` door de regel: 1 bij een vermogensstoring, en anders 0. De check roept `power_failure(20.0, 1500)` aan, wat 1 moet geven, en nog zeven; daaronder geeft je regel alle 10.000 metingen van AI4I een label.
        """),
        L.task("Laat `power_failure` 1 geven onder 3500 W of boven 9000 W, en anders 0."),
    ])
    return


@app.cell(hide_code=True)
def exercise_power():
    box_power = L.code_box("power", lines=10)
    box_power
    return (box_power,)


@app.cell(hide_code=True)
def check_power(box_power):
    result_power = L.check("power", box_power.value)
    result_power
    return (result_power,)


@app.cell(hide_code=True)
def hints_power():
    L.hints(
        "power",
        [
            "Er zijn twee manieren om uit te vallen. Wat geeft je functie in elk daarvan terug, en wat aan het einde, als geen van beide klopt?",
            "Een `if` met een `return` erin stopt de functie meteen. Na twee zulke `if`s zijn alleen de metingen zonder vermogensstoring over.",
            "`if power < ___:` met daaronder `return 1`, hetzelfde voor `power > ___`, en aan het einde `return ___`.",
        ],
        why="Bij het voorbeeld is het vermogen 20,0 × 1500 × 2π / 60 ≈ 3142 W, onder 3500 W: de eerste `if` geeft 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_between():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Getallen tussen twee grenzen

        `between(rng, low, high, n)` moet `n` willekeurige getallen tussen `low` en `high` geven, uit `rng.random(n)`, zoals het uitgewerkte voorbeeld zijn temperatuurverschillen tussen 7,5 en 9,5 K trekt. Vervang de `...`. De check roept `between(np.random.default_rng(1), 7.5, 9.5, 3)` aan en nog vijf, elke keer met een nieuwe generator, zodat ze jouw getallen met de hare kan vergelijken.
        """),
        L.task("Laat `between` `n` getallen tussen `low` en `high` geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_between():
    box_between = L.code_box("between", lines=5)
    box_between
    return (box_between,)


@app.cell(hide_code=True)
def check_between(box_between):
    result_between = L.check("between", box_between.value)
    result_between
    return (result_between,)


@app.cell(hide_code=True)
def hints_between():
    L.hints(
        "between",
        [
            "`rng.random(n)` geeft `n` getallen tussen 0 en 1. Hoe breed moeten ze worden, en waar moeten ze beginnen?",
            "Vermenigvuldigen rekt ze op, tot tussen 0 en `high - low`. Schuif ze daarna omhoog, zodat ze bij `low` beginnen.",
            "`return ___ + (___ - ___) * rng.random(n)`.",
        ],
        why="Bij `low` 7,5 en `high` 9,5 is de breedte 2: `rng.random(n)` keer 2 ligt tussen 0 en 2, en 7,5 plus dat tussen 7,5 en 9,5.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_test_on():
    mo.md(r"""
    ### Oefening 3. Welke test vertrouw je?
    """)
    return


@app.cell(hide_code=True)
def choice_test_on_cell():
    choice_test_on = L.choice(
        "test-on",
        question="Volgens de regel van een collega komt een hittestoring voor onder 1460 rpm. Getraind op het logboek en 1000 metingen die met die regel gesimuleerd zijn, mist k-NN 8 van de 207 storingen in 1000 andere gesimuleerde metingen, op dezelfde manier gelabeld, en geeft het 3 keer vals alarm. Wat zegt dat over de echte machine?",
        options={
            "good": "Het model zal het goed doen: 11 fouten op 1000 metingen",
            "right": "Weinig: de labels van de test komen van dezelfde regel, dus toetst de test de regel aan zichzelf",
            "strict": "Het zal veel echte storingen missen: 1460 rpm is te streng",
        },
        correct="right",
        why={
            "good": "De gesimuleerde testmetingen kregen hun labels van de regel van de collega. Op de volgende 2000 echte metingen geeft hetzelfde model 176 keer vals alarm.",
            "strict": "1460 rpm is ruimer dan de 1380 van AI4I, niet strenger: meer metingen tellen als storing. Op de echte metingen mist het model er 6 en geeft het 176 keer vals alarm.",
            "right": "Goed: het model leerde de regel, en de test legt het langs diezelfde regel. Op de volgende 2000 echte metingen mist het er 6 van de 115 en geeft het 176 keer vals alarm: de regel is te ruim.",
        },
    )
    choice_test_on
    return (choice_test_on,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Verbeter de data, niet het model](/course/open/data-centric-ai?lang=nl) maakte de data beter door labels te verbeteren; deze les maakt haar beter door de voorbeelden toe te voegen die ontbraken. k-NN op gestandaardiseerde features komt uit [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl), en een test set die nergens anders aan meedeed uit [Werkt het op nieuwe data? Training set en test set](/course/open/ml-train-test-split?lang=nl). Voorbeelden veranderen die je al hebt, komt terug bij beelden; een model dat zelf voorbeelden schrijft, in het domein over taal.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_test_on, result_between, result_power):
    L.wrap_up(result_power, result_between, choice_test_on)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
