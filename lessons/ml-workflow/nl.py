# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-workflow. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-workflow", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Drie korte vragen over [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl), [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl) en [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_baseline():
    quick_baseline = L.choice(
        "quick-baseline",
        question="Een model voorspelt huizenprijzen met een MSE van 900 op de test set. Altijd de gemiddelde prijs van de training set voorspellen haalt een MSE van 850 op dezelfde huizen. Wat zegt dat?",
        options={
            "small": "Het model is goed: een MSE van 900 is klein voor prijzen van een paar ton",
            "right": "Het model heeft niet laten zien dat het iets van de features geleerd heeft",
            "cheat": "De baseline speelt vals: die is op de test set uitgerekend",
        },
        correct="right",
        why={
            "small": "Een MSE heeft geen vaste schaal: of 900 klein is, zegt alleen een vergelijking. Welk getal is lager? Zie [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl).",
            "cheat": "Het gemiddelde komt uit de training set, net als het model; alleen het meten gebeurt op de test set. Zie [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl).",
            "right": "Bij een MSE is lager beter, en het gemiddelde alleen haalt 850: met deze metric hielpen de features het model niet.",
        },
    )
    quick_baseline
    return (quick_baseline,)


@app.cell(hide_code=True)
def quick_check_precision():
    quick_precision = L.choice(
        "quick-precision",
        question="Een filter verplaatst 8 mails naar de spammap, en 6 daarvan zijn spam. Er waren in totaal 10 spammails. Wat zijn de precision en de recall?",
        options={
            "right": "Precision 0,75, recall 0,6",
            "swapped": "Precision 0,6, recall 0,75",
            "same": "Allebei 0,6",
        },
        correct="right",
        why={
            "swapped": "Die zijn omgedraaid. Welke van de twee kijkt alleen naar de mails die het filter verplaatste? Zie *Het idee* in [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
            "same": "Dat is de recall. Waardoor deel je voor de precision? Zie *Het idee* in [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
            "right": "Van de 8 mails die het verplaatste, zijn er 6 spam: precision 0,75. Van de 10 spammails vond het er 6: recall 0,6.",
        },
    )
    quick_precision
    return (quick_precision,)


@app.cell(hide_code=True)
def quick_check_license():
    quick_license = L.choice(
        "quick-license",
        question="Je vindt een set foto's op een website, zonder licentie. Wat mag je ermee?",
        options={
            "free": "Vrij gebruiken: het staat op internet",
            "credit": "Gebruiken en delen, als je de website waar het vandaan komt maar noemt",
            "right": "Ervan uitgaan dat je het niet mag kopiëren of delen, en eerst de maker vragen",
        },
        correct="right",
        why={
            "free": "Dat iets op internet staat, maakt het nog niet vrij te gebruiken. Wat betekent geen licentie? Zie [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).",
            "credit": "De bron noemen is wat sommige licenties vragen; hier is er geen. Zie [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).",
            "right": "Geen licentie betekent geen toestemming: vraag het de maker voordat je het deelt of erop voortbouwt.",
        },
    )
    quick_license
    return (quick_license,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je kent nu de onderdelen van een project: een metric, een training set, validation set en test set, een baseline, een model, en een confusion matrix die laat zien waar het misgaat. Een echt project is meer dan zijn onderdelen. Het begint vóór de eerste regel code, met een vraag, en het houdt niet op bij het eerste model. Ook de volgorde telt: een doel dat je kiest nadat je de uitkomsten zag, buigt mee naar die uitkomsten, net als een testscore als je met de test set kiest.

    **Een voorbeeld uit het dagelijks leven.** Een gerecht dat je voor het eerst kookt, is zelden meteen goed. Je proeft, verandert één ding, en kookt het opnieuw.
    """)
    return


@app.cell(hide_code=True)
def predict_first():
    guess_first = L.guess(
        "first",
        question="Een webshop vraagt je om een model dat voorspelt welke klanten weggaan. Wat moet je volgens jou als eerste doen?",
        options={
            "data": "Zoveel mogelijk data over de klanten verzamelen, uit elk systeem van de winkel",
            "model": "Een eerste model trainen, om te zien hoe ver je komt",
            "goal": "Afspreken wat weggaan is, en hoe je een model beoordeelt",
            "method": "De beste soort model voor de klus kiezen",
        },
        reveal="**Afspreken wat weggaan is, en hoe je een model beoordeelt.** Is een klant die een halfjaar niets bestelt weg? En welke fout kost meer: een trouwe klant aanzien voor iemand die weggaat, of een vertrekkende klant missen? Zolang dat niet is afgesproken, kun je elke uitkomst achteraf mooi laten lijken. Misschien blijkt zelfs dat een eenvoudige regel het werk doet, en dan heb je helemaal geen model nodig.",
    )
    guess_first
    return (guess_first,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De **workflow van machine learning** is de volgorde van de stappen van een project, en de weg terug:

        1. **Vraag.** Bij welke beslissing moet het model helpen? Kijk eerst of een vaste regel dat al doet: dan heb je geen machine learning nodig. Zo niet, schrijf het als een taak: wat zijn de voorbeelden, de features en het label?
        2. **Metric.** Hoe beoordeel je een model, en hoe goed is goed genoeg? Dat volgt uit wat elke soort fout kost. Spreek het af vóór je traint.
        3. **Data.** Waar komt de data vandaan, en mag je die gebruiken? Splits de data in een training set, validation set en test set voordat je goed kijkt.
        4. **Baseline.** Wat haalt de eenvoudigste voorspelling op de validation set: de meest voorkomende klasse, het gemiddelde, of een eenvoudige regel?
        5. **Model.** Train op de training set, en kies op de validation set, met de metric van stap 2.
        6. **Nog een ronde.** Kijk naar de voorbeelden die het fout heeft, en ga terug naar de stap die het tegenhoudt: meer of betere data, andere features, een ander model, en soms een betere vraag.

        Pas als alles gekozen is, meet je het model en de baseline één keer op de test set, en zet je die twee getallen naast elkaar.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **workflow van machine learning** | | de stappen van een project in hun volgorde: vraag, metric, data, baseline, model, en nog een ronde tot het goed genoeg is | |
        """),
        L.watch_out("Spreek de metric en het doel af voordat je uitkomsten ziet. Kies je ze achteraf, dan buigen ze mee naar het model dat je toevallig hebt, net als een test set waarmee je kiest."),
        mo.Html(C.picture([("1. Vraag", "wat moet het beslissen?"), ("2. Metric", "hoe goed is goed genoeg?"), ("3. Data", "waarvandaan? splitsen"),
                           ("4. Baseline", "de eenvoudigste voorspelling"), ("5. Model", "trainen, kiezen met de metric"), ("6. Nog een ronde", "waar gaat het mis?")],
                          "terug naar de stap die het tegenhoudt", "De workflow van machine learning",
                          "Zes vakken in een lus: 1 vraag, 2 metric en 3 data links omlaag, dan opzij naar 4 baseline, en 5 model en 6 nog een ronde rechts omhoog. Een gestreepte pijl loopt van 6 terug naar 3, waar de ronde opnieuw begint: terug naar de stap die het tegenhoudt.")),
        mo.md(r"""
        Stap 1 en 2 komen eerst, en de test set gaat één keer apart. Daarna gaan stap 3 tot en met 6 rond: van stap 6 terug naar de stap die het model tegenhoudt, meestal de data of het model, tot het model goed genoeg is, of tot duidelijk is dat het dat niet wordt.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo doorloop je een project met machine learning:

    1. Schrijf de vraag op, en kijk of geen vaste regel hem beantwoordt.
    2. Spreek de metric en het doel af, vanuit wat elke fout kost.
    3. Zoek uit waar de data vandaan komt en of je die mag gebruiken, en splits de data.
    4. Meet een baseline op de validation set.
    5. Train een model, en kies op de validation set met de metric.
    6. Kijk wat het fout doet, ga terug naar de stap die het tegenhoudt, en herhaal. Meet aan het eind één keer op de test set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een spamfilter, van stap 1 tot stap 6. Er is al een filter dat elke mail een kans op spam geeft, zoals in [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl); in stap 5 kies je alleen nog zijn threshold. Hier zijn 20 mails uit de validation set, met hun kans en of ze spam zijn. `(_probs >= _t) * 1` maakt van `True` en `False` 1 en 0: de voorspellingen van het filter. En `_probs[_y == 0]` kiest uit `_probs` met een mask van de labels: de kansen van de echte mails. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_workflow():
    import numpy as np

    # 1. de vraag: spam naar een eigen map; een echte mail mag daar bijna nooit terechtkomen
    # 2. de metric: een precision van minstens 0.9, en dan de hoogste recall
    # 3. de data: mails die hiervoor gebruikt mogen worden, gesplitst voordat iemand goed keek;
    #    hier 20 mails uit de validation set, hun kans op spam, en 1 voor spam
    _probs = np.array([0.97, 0.12, 0.83, 0.66, 0.04, 0.35, 0.92, 0.28, 0.58, 0.09,
                       0.45, 0.62, 0.02, 0.81, 0.22, 0.88, 0.06, 0.41, 0.74, 0.18])
    _y = np.array([1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0])
    # 4. de baseline: elke mail spam noemen; de precision is het aandeel spam
    print(np.mean(_y))
    # 5. het model: een threshold, gekozen met de metric; elke regel: threshold, precision, recall
    for _t in [0.3, 0.5, 0.7, 0.8, 0.85, 0.9]:
        _pred = (_probs >= _t) * 1
        _tp = np.sum(_pred * _y)
        print(_t, _tp / np.sum(_pred), _tp / np.sum(_y))
    # 6. nog een ronde: de echte mails met een hoge kans, en de spam onder 0.85
    _ham = _probs[_y == 0]
    _spam = _probs[_y == 1]
    print(_ham[_ham >= 0.5])
    print(_spam[_spam < 0.85])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    8 van de 20 mails zijn spam, dus elke mail spam noemen geeft een precision van 0,4: ver onder het doel van 0,9, en niets spam noemen vindt helemaal geen spam. Met het filter stijgt de precision meestal met de threshold: bij 0,8 zakt hij even, omdat er een spammail onder valt. Pas vanaf 0,85 haalt hij 0,9. Daar is de recall 0,375: het filter verplaatst 3 van de 8 spammails. Dat haalt het doel, maar het helpt nauwelijks, en daar is stap 6 voor. Twee echte mails krijgen 0,83 en 0,62, en vijf spammails krijgen tussen 0,45 en 0,81. Wie die zeven mails leest, ziet waar het filter de fout ingaat. Misschien heeft het andere features nodig, meer voorbeelden van dit soort spam, of een ander model. Dan maak je nog een ronde. De test set wacht tot het eind.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_need():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Is machine learning nodig?
    """)
    return


@app.cell(hide_code=True)
def choice_need_cell():
    choice_need = L.choice(
        "need-ml",
        question="Een bibliotheek wil een herinnering sturen voor elk boek dat meer dan 21 dagen te laat is. Wat is daarvoor nodig?",
        options={
            "model": "Een classifier, getraind op oude uitleningen met hun inleverdata en herinneringen",
            "right": "Geen machine learning: de vaste regel, meer dan 21 dagen te laat, doet het",
            "baseline": "Eerst een baseline, en dan een model dat die verslaat",
        },
        correct="right",
        why={
            "model": "Wat zou een model hier leren dat de bibliotheek niet al precies weet? Zie stap 1 in *Het idee*.",
            "baseline": "Een baseline is voor een taak die een model moet leren. Moet hier iets geleerd worden? Zie stap 1 in *Het idee*.",
            "right": "De regel is bekend en precies: je schrijft hem als code. Machine learning is voor beslissingen waarvan je de regel niet kunt opschrijven, zoals spam of handschrift.",
        },
    )
    choice_need
    return (choice_need,)


@app.cell(hide_code=True)
def text_exercise_steps():
    mo.md(r"""
    ### Oefening 2. Welke stap?

    Een team bouwt een spamfilter. Zet alles wat het team doet bij de stap van de workflow waar het hoort.
    """)
    return


@app.cell(hide_code=True)
def sort_steps_cell():
    sort_steps = L.sort("steps")
    sort_steps
    return (sort_steps,)


@app.cell(hide_code=True)
def check_steps(sort_steps):
    result_steps = L.check("steps", sort_steps)
    result_steps
    return (result_steps,)


@app.cell(hide_code=True)
def hints_steps():
    L.hints(
        "steps",
        [
            "Vraag bij elk: gebeurt het voordat er voorbeelden zijn, met de voorbeelden, of met een model?",
            "Twee gaan over de voorbeelden. Welke komt pas als er een model is, en kijkt waar het misgaat?",
            "Vraag: waar het voor is. Metric: \\_\\_\\_. Data: toestemming, en \\_\\_\\_. Baseline: nooit spam. Model: trainen. Nog een ronde: \\_\\_\\_.",
        ],
        why="Waar het voor is, is de vraag; hooguit 1 echte mail op de 20 in de spammap is de metric; toestemming en de test set apart zetten zijn de data; nooit spam is de baseline; trainen en de threshold kiezen is het model; het lezen van de valse alarmen begint de volgende ronde.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Laat de metric kiezen

        Vijf spamfilters zijn gemeten op de validation set: `names`, `precision` en `recall` bevatten hun namen en scores, in dezelfde volgorde. Het doel is dat van het uitgewerkte voorbeeld: een precision van minstens 0,9, en dan de hoogste recall. Vervang de `...` door de keuze: als een filter een precision van minstens 0,9 haalt, en zijn recall hoger is dan `best_recall`, wordt het `best`, en zijn recall de nieuwe `best_recall`. De laatste regel is `best`, zodat de check het ziet. De check probeert ook andere filters.
        """),
        L.task("Zet in `best` de naam van het filter met de hoogste recall van de filters met een precision van minstens 0,9."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pick():
    box_pick = L.code_box("pick-model", lines=12)
    box_pick
    return (box_pick,)


@app.cell(hide_code=True)
def check_pick(box_pick):
    result_pick = L.check("pick-model", box_pick.value)
    result_pick
    return (result_pick,)


@app.cell(hide_code=True)
def hints_pick():
    L.hints(
        "pick-model",
        [
            "Twee vragen per filter, in deze volgorde: haalt het het doel voor de precision, en vindt het meer spam dan het beste tot nu toe?",
            "Een `if` in een `if`: de buitenste vraagt naar `precision[i]`, de binnenste naar `recall[i]`. Wat moet er veranderen als beide waar zijn?",
            "`if precision[i] >= 0.9:`, daarin `if recall[i] > best_recall:`, en daarin `best = ___` en `best_recall = ___`.",
        ],
        why="Vier filters halen een precision van minstens 0,9: B, C, D en E, met een recall van 0,40, 0,75, 0,60 en 0,30. Daarvan vindt C de meeste spam. A heeft de hoogste recall van allemaal, 0,90, maar haalt met een precision van 0,85 het doel niet.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl) gaf je stap 4, [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl) de metrics van stap 2 en de blik op fouten van stap 6, en [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl) de vragen van stap 3. De workflow zet ze op volgorde, en maakt er een lus van. In de volgende les, *Project: voorspel iets echts, van begin tot eind*, doorloop je de hele workflow op een echte tabel, de Palmer-pinguïns, met scikit-learn.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_need, result_pick, result_steps):
    L.wrap_up(choice_need, result_steps, result_pick)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
