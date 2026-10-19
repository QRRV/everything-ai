# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-baselines. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-baselines", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [De loss waarop je traint en de metric die je meldt](/course/open/ml-loss-vs-metric?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_accuracy():
    quick_accuracy = L.choice(
        "quick-accuracy",
        question="Een model geeft drie foto's de kansen `[0.2, 0.8]`, `[0.6, 0.4]` en `[0.3, 0.7]`, en alle drie hebben het label 1. Wat is zijn accuracy?",
        options={
            "mean": "Ongeveer 0,63, de gemiddelde kans op klasse 1",
            "one": "1, omdat alle drie de labels hetzelfde zijn",
            "right": "Ongeveer 0,67",
        },
        correct="right",
        why={
            "mean": "Dat is hoe zeker het model gemiddeld is van klasse 1. Welke klasse voorspelt het voor elke foto? Zie het recept in [De loss waarop je traint en de metric die je meldt](/course/open/ml-loss-vs-metric?lang=nl).",
            "one": "Accuracy vergelijkt elke voorspelling met zijn label. Wat voorspelt het model voor de tweede foto? Zie het recept in [De loss waarop je traint en de metric die je meldt](/course/open/ml-loss-vs-metric?lang=nl).",
            "right": "De waarschijnlijkste klassen zijn 1, 0 en 1: twee van de drie zijn goed.",
        },
    )
    quick_accuracy
    return (quick_accuracy,)


@app.cell(hide_code=True)
def quick_check_train():
    quick_train = L.choice(
        "quick-train",
        question="Op welk getal traint gradient descent, en welk getal meld je?",
        options={
            "right": "Het traint op de loss, en je meldt een metric zoals accuracy",
            "metric": "Het traint op de accuracy, en je meldt de loss",
            "same": "Het is allebei hetzelfde getal",
        },
        correct="right",
        why={
            "metric": "Accuracy is een trap zonder hellingen. Zie *Het idee* in [De loss waarop je traint en de metric die je meldt](/course/open/ml-loss-vs-metric?lang=nl).",
            "same": "Het zijn verschillende getallen die uit elkaar kunnen lopen. Zie *Het idee* in [De loss waarop je traint en de metric die je meldt](/course/open/ml-loss-vs-metric?lang=nl).",
            "right": "De loss heeft hellingen om te volgen; de metric zegt waar mensen om geven.",
        },
    )
    quick_train
    return (quick_train,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model haalt een accuracy van 0,90 op zijn test set. Is dat goed? Op zich zegt het getal niets: als negen van de tien voorbeelden bij één klasse horen, haal je met altijd die klasse gokken ook 0,90. Voordat je een score vertrouwt, vergelijk je hem met de eenvoudigste voorspelling die je kunt doen zonder iets van de features te leren. Verslaat het model die niet, dan laat zijn score niets zien van wat het geleerd heeft.

    **Een voorbeeld uit het dagelijks leven.** Een weersverwachting van "morgen wordt het zoals vandaag" klopt verrassend vaak. Een weerdienst is pas het geld waard als die het beter doet dan dat.
    """)
    return


@app.cell(hide_code=True)
def predict_good():
    guess_good = L.guess(
        "good",
        question="Een lijn voorspelt de prijzen van 8 testhuizen met een MSE van ongeveer 648. Wat denk je dat je op dezelfde huizen haalt als je voor elk huis de gemiddelde trainingsprijs voorspelt?",
        options={
            "lower": "Minder dan 648: het gemiddelde is de beste vaste voorspelling",
            "same": "Ongeveer evenveel: de huizen liggen toch dicht bij de lijn",
            "higher": "Veel meer, een paar keer 648",
        },
        reveal="**Veel meer: ongeveer 3088**, bijna vijf keer de MSE van de lijn. De gemiddelde trainingsprijs, ongeveer 248 duizend euro, zit ver naast de goedkope en de dure huizen, dus de lijn heeft wel iets van de oppervlakte geleerd. Zonder die 3088 ernaast zegt 648 weinig: een MSE is in duizend euro in het kwadraat en heeft geen vaste schaal.",
    )
    guess_good
    return (guess_good,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **baseline** is een eenvoudige voorspelling om een model mee te vergelijken, zoals een regel die de features negeert. De gebruikelijkste zijn:

        - bij classificatie: voorspel altijd de **meest voorkomende klasse** van de training set;
        - bij regressie: voorspel altijd het **gemiddelde label** van de training set.

        De baseline leert alleen van de training set, net als het model, en je meet ze allebei op dezelfde test set met dezelfde metric. Een model dat zijn baseline niet verslaat, heeft met die metric niet laten zien dat het iets van de features geleerd heeft, hoe goed zijn score er ook uitziet. Hoe groter het verschil, hoe meer het model toevoegt. Zolang je nog kiest, vergelijk je met de baseline op de validation set; aan het eind meld je ze allebei op de test set.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **baseline** | | een eenvoudige voorspelling om een model mee te vergelijken, zoals altijd de meest voorkomende klasse of het gemiddelde label van de training set | `counts.argmax()`, `y_train.mean()` |
        """),
        L.watch_out("Haal de baseline uit de training set, nooit uit de test set. Het gemiddelde van de testlabels is de beste vaste voorspelling voor precies die voorbeelden, dus beter dan elk getal dat je had kunnen kiezen voordat je ze zag: dat is geen eerlijke baseline."),
        mo.md(r"""
        De huizen uit [Werkt het op nieuwe data? Training set en test set](/course/open/ml-train-test-split?lang=nl): 32 huizen in de training set (gevuld) en 8 in de test set (open). De doorgetrokken <span class="c-pred">lijn</span> is geleerd van de training set; de gestreepte lijn is de baseline, de gemiddelde trainingsprijs voor elk huis.
        """),
        mo.Html(C.picture({"title": "Een lijn en zijn baseline", "x": "oppervlakte (honderden m²)", "y": "prijs (duizend euro)", "baseline": "baseline",
                           "alt": "Veertig huizen naar oppervlakte en prijs, 32 gevuld en 8 open. Een stijgende rechte lijn gaat erdoorheen; een vlakke gestreepte lijn bij ongeveer 248 duizend euro is de baseline, ver van de goedkope en de dure huizen."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo beoordeel je een score:

    1. Kies een baseline: de meest voorkomende klasse, of het gemiddelde label, van de training set.
    2. Meet de baseline op de test set met dezelfde metric als het model.
    3. Vergelijk: pas het verschil tussen het model en zijn baseline laat zien wat het model geleerd heeft.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Allebei de gevallen uit het recept. Eerst classificatie: tien mails in de training set en tien in de test set, waarbij 1 spam is, en de klassen die een spamfilter voor de testmails voorspelt. Dan regressie: de huizen uit de figuur, met de lijn die van de training set geleerd is. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_baselines():
    import numpy as np

    # classificatie: tien mails in de training set en tien in de test set, 1 is spam
    _mails_train = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    _mails_test = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0])
    _predicted = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0])   # wat het filter voor de testmails voorspelt
    # 1. de baseline: de meest voorkomende klasse van de training set
    _counts = np.eye(2)[_mails_train].sum(axis=0)
    _most_common = _counts.argmax()
    print(_counts, _most_common)
    # 2. allebei op dezelfde test set, met de accuracy
    print(np.mean(_predicted == _mails_test), np.mean(_mails_test == _most_common))

    # regressie: de huizen uit de figuur
    _rng = np.random.default_rng(18)
    _sizes = 0.5 + 1.5 * _rng.random(size=40)
    _prices = 60 + 150 * _sizes + 80 * (_rng.random(size=40) - 0.5)
    _order = _rng.permutation(40)
    _x_train, _y_train = _sizes[_order[:32]], _prices[_order[:32]]
    _x_test, _y_test = _sizes[_order[32:]], _prices[_order[32:]]
    _w, _b = 152.82, 52.32   # de lijn die van de training set geleerd is

    # 1. de baseline: de gemiddelde trainingsprijs
    _baseline = _y_train.mean()
    print(_baseline)
    # 2. allebei op dezelfde test set, met de MSE
    print(np.mean((_y_test - (_w * _x_test + _b)) ** 2))
    print(np.mean((_y_test - _baseline) ** 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Acht van de tien mails in de training set zijn geen spam, dus de baseline voorspelt altijd klasse 0. Op de testmails haalt het filter een accuracy van 0,8, en de baseline ook: de 0,8 van het filter laat niets zien van wat het geleerd heeft. De gemiddelde trainingsprijs is ongeveer 248 duizend euro. Op de 8 testhuizen heeft de lijn een MSE van ongeveer 648, de baseline ongeveer 3088: bijna vijf keer zoveel, dus de oppervlakte helpt echt om de prijs te voorspellen. Stap 3 is de vergelijking zelf.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_disease():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een zeldzame ziekte
    """)
    return


@app.cell(hide_code=True)
def choice_disease_cell():
    choice_disease = L.choice(
        "rare-disease",
        question="Twee op de honderd mensen hebben een ziekte. Een model dat die moet herkennen, haalt een accuracy van 0,97 op de test set. Wat zegt dat?",
        options={
            "good": "Het werkt goed: 97 van de 100 kloppen",
            "right": "Het is slechter dan altijd 'gezond' zeggen, wat 0,98 haalt",
            "nothing": "Niets: accuracy kun je niet gebruiken voor een ziekte",
        },
        correct="right",
        why={
            "good": "Wat haalt de eenvoudigste regel hier, altijd de meest voorkomende klasse? Zie *Het idee*.",
            "nothing": "Accuracy kun je gebruiken; hij heeft alleen iets nodig om mee te vergelijken. Zie *Het idee*.",
            "right": "Altijd de meest voorkomende klasse voorspellen, gezond, heeft er 98 van de 100 goed. Het model verslaat dat niet eens, dus zijn 0,97 laat niets zien van wat het geleerd heeft. Of het de zieke mensen vindt, zegt de accuracy niet.",
        },
    )
    choice_disease
    return (choice_disease,)


@app.cell(hide_code=True)
def text_exercise_majority():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De meest voorkomende klasse

        Twaalf trainingslabels en acht testlabels, van drie klassen. `counts` bevat al hoe vaak elke klasse in de training set voorkomt, met one-hot-vectoren zoals in [One-hot-vectoren voor klassen](/course/open/ml-one-hot?lang=nl). Zet de meest voorkomende klasse in `most_common`, en in `accuracy` de accuracy op de test set als je die klasse altijd voorspelt. Vervang de twee `...`. De laatste regel is `accuracy`, zodat de check het ziet. De check probeert ook andere trainingslabels.
        """),
        L.task("Zet in `accuracy` de accuracy op de test set als je altijd de meest voorkomende klasse van de training set voorspelt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_majority():
    box_majority = L.code_box("majority", lines=8)
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
            "Stap 1 en 2 van het recept voor classificatie.",
            "Welke methode geeft de plek van het grootste getal in `counts`? Vergelijk daarna elk testlabel met die klasse, zoals bij accuracy.",
            "`most_common = counts.___()` en `accuracy = np.mean(y_test == ___)`.",
        ],
        why="De tellingen zijn 3, 7 en 2, dus klasse 1 komt het meest voor. Van de 8 testlabels zijn er 5 gelijk aan 1: een accuracy van 0,625.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mean():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De gemiddelde prijs

        Zes trainingsprijzen en drie testprijzen. Zet de gemiddelde trainingsprijs in `baseline`, en in `mse` de MSE op de test set als je die prijs voor elk huis voorspelt. Vervang de twee `...`. De laatste regel is `mse`, zodat de check het ziet. De check probeert ook andere trainingsprijzen.
        """),
        L.task("Zet in `mse` de MSE op de test set als je voor elk huis de gemiddelde trainingsprijs voorspelt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mean():
    box_mean = L.code_box("mean-baseline", lines=7)
    box_mean
    return (box_mean,)


@app.cell(hide_code=True)
def check_mean(box_mean):
    result_mean = L.check("mean-baseline", box_mean.value)
    result_mean
    return (result_mean,)


@app.cell(hide_code=True)
def hints_mean():
    L.hints(
        "mean-baseline",
        [
            "Stap 1 en 2 van het recept voor regressie.",
            "Welke set geeft de baseline, en op welke set meet je hem? De voorspelling is voor elk huis hetzelfde getal.",
            "`baseline = ___.mean()` en `mse = np.mean((___ - baseline) ** 2)`.",
        ],
        why="De gemiddelde trainingsprijs is 240. De testprijzen liggen er 20, 40 en 50 vanaf; hun kwadraten, 400, 1600 en 2500, hebben een gemiddelde van 1500.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [De loss waarop je traint en de metric die je meldt](/course/open/ml-loss-vs-metric?lang=nl) koos je de metric om te melden; een baseline geeft dat getal betekenis. In [Werkt het op nieuwe data? Training set en test set](/course/open/ml-train-test-split?lang=nl) werd de lijn gemeten op nieuwe huizen; nu weet je ook dat hij de eenvoudigste regel verslaat. In de volgende les, *Project: een lineaire classifier voor handgeschreven cijfers*, train je een model op echte plaatjes, en het eerste wat je controleert, is of het beter doet dan altijd het meest voorkomende cijfer gokken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_disease, result_majority, result_mean):
    L.wrap_up(choice_disease, result_majority, result_mean)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
