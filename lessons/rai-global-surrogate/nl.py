# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-global-surrogate. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-global-surrogate", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Hoe een voorspelling meebeweegt met een feature](/course/open/rai-pdp-ice?lang=nl) en [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_average():
    quick_average = L.choice(
        "quick-average",
        question="In het uitgewerkte voorbeeld van Hoe een voorspelling meebeweegt met een feature stijgt de partial dependence van het forest op leeftijd met 0,25 van 25 tot 45 jaar, terwijl de lijnen van de mensen met een opleiding van 13 of meer gemiddeld met 0,476 stijgen. Wat verbergt het gemiddelde?",
        options={
            "ignores": "Dat het forest opleiding buiten beschouwing laat",
            "right": "Dat leeftijd zwaarder telt met een diploma",
            "few": "Dat er te weinig mensen van 45 in de tabel staan",
        },
        correct="right",
        why={
            "right": "Goed. De lijnen van mensen met een opleiding tot 9 stijgen met 0,154, die van mensen met 13 of meer met 0,476: wat leeftijd doet, hangt af van opleiding, en één gemiddelde lijn kan dat niet laten zien. Deze les zoekt één beeld van het hele forest, met al zijn features tegelijk.",
            "ignores": "Het forest gebruikt opleiding: de lijnen stijgen drie keer zo hard bij mensen met 13 of meer als bij mensen met een opleiding tot 9. Het gemiddelde over iedereen verbergt dat verschil. Zie [Hoe een voorspelling meebeweegt met een feature](/course/open/rai-pdp-ice?lang=nl).",
            "few": "Bij 45 jaar wordt het forest gevraagd naar alle 1000 mensen, wat hun eigen leeftijd ook is. Wat het gemiddelde verbergt, is dat de lijnen per opleiding verschillen: 0,154 bij een opleiding tot 9, 0,476 bij 13 of meer. Zie [Hoe een voorspelling meebeweegt met een feature](/course/open/rai-pdp-ice?lang=nl).",
        },
    )
    quick_average
    return (quick_average,)


@app.cell(hide_code=True)
def quick_check_pure():
    quick_pure = L.choice(
        "quick-pure",
        question="Een beslisboom groeit door tot elk deel zuiver is. Hoeveel van zijn voorbeelden uit de training set heeft hij goed?",
        options={
            "right": "Allemaal",
            "validation": "Ongeveer evenveel als op de validation set",
            "common": "Alleen die van de klasse die het vaakst voorkomt",
        },
        correct="right",
        why={
            "right": "Goed: elk blad bevat voorbeelden van één klasse en zegt die klasse, dus elk voorbeeld uit de training set is goed, en de boom overfit. In deze les doet een boom de antwoorden van een ander model na, en een kopie die zo ver doorgroeit, is het eens met elke rij waarvan hij leerde. Dat zegt weinig; het gaat om de rijen waarvan hij niet leerde.",
            "validation": "Op de validation set doet hij het slechter: een boom die doorgroeit tot elk deel zuiver is, heeft elk voorbeeld uit de training set goed, omdat elk blad één klasse bevat. Dat is overfitting. Zie [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl).",
            "common": "Elk blad zegt de klasse van de voorbeelden erin, en als elk blad zuiver is, is dat voor elk van hen de goede klasse: alle voorbeelden uit de training set zijn goed. Zie [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl).",
        },
    )
    quick_pure
    return (quick_pure,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In de vorige twee lessen mat de toezichthouder van de bank op welke features het forest leunt, en tekende hoe zijn ja meebeweegt met de leeftijd; net als eerder is een voorspelling van meer dan 50.000 dollar verdienen het ja van de bank. Allebei kijken ze naar één feature tegelijk. Nu vraagt de toezichthouder om het hele forest in één keer, als regels die een mens kan lezen. Maar het forest bestaat uit 100 bomen met samen 28.295 bladeren, en dat leest niemand. Doe het dus na: laat één kleine boom groeien, niet op wie echt meer verdient, maar op wat het forest zegt. Tel dan hoe vaak de kopie hetzelfde antwoord geeft als het forest. Waar dat zo is, beschrijven zijn paar vragen het forest; waar het niet zo is, beschrijven ze iets anders.

    **Een voorbeeld uit het dagelijks leven.** Een nieuwe medewerker van een bakkerij kijkt elke avond hoe de chef-bakker beslist welke broden voor de halve prijs weggaan, en schrijft drie regels op een kaartje. Of het kaartje deugt, toets je aan de beslissingen van de chef-bakker zelf, niet tegen wat er verkocht is: klopt het kaartje op 19 van de 20 avonden, dan vat het de chef-bakker goed samen. Waarom de chef-bakker zo beslist, zegt het niet, en op de twintigste avond was er een reden die het kaartje niet kent.
    """)
    return


@app.cell(hide_code=True)
def predict_baseline():
    guess_baseline = L.guess(
        "baseline",
        question="Op de testhelft zegt het forest tegen 14,8% van de mensen ja. Neem een kopie die tegen iedereen nee zegt. Hoe vaak is die het eens met het forest?",
        options={
            "low": "Bij ongeveer 15% van de mensen",
            "half": "Bij ongeveer de helft van de mensen",
            "high": "Bij ongeveer 85% van de mensen",
        },
        reveal="**Ongeveer 85%: bij 85,2% van de testhelft.** Hij is het eens waar het forest nee zegt, en dat zijn de meeste mensen. De fidelity van een kopie zegt dus weinig zolang je hem niet met dit getal vergelijkt. Een kopie met één vraag over opleiding haalt 88,1%; de kopie met diepte 3 uit het uitgewerkte voorbeeld verderop 95,7%.",
    )
    guess_baseline
    return (guess_baseline,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Doe de antwoorden na, niet de wereld.** Mark Craven en Jude Shavlik (1996) lieten een beslisboom groeien uit een getraind neural network, met een methode die ze TREPAN noemden. Het network was de leraar: het beantwoordde de vragen van TREPAN, dat het network alleen als black box gebruikte. Daarna maten ze de boom op twee manieren. Accuracy is bij hen het percentage voorbeelden uit de test set dat goed wordt ingedeeld; fidelity het percentage voorbeelden uit de test set waarop de boom hetzelfde indeelt als zijn neural network. Met hoogstens 15 splitsingen in een boom waren de bomen het op vier datasets eens met hun network bij 85,7 tot 95,9 procent van de voorbeelden uit de test set. Op data over hartziekten had het network 84,5 procent goed, de boom 81,8 procent, en waren de twee het eens bij 94,1 procent.

        **Een global surrogate.** Christoph Molnar noemt het een global surrogate: een global surrogate model is een model dat je kunt lezen en dat getraind is om de voorspellingen van een black box na te bootsen. Het recept: neem een dataset, vraag de black box om zijn voorspellingen, kies een soort model dat je kunt lezen, train het op de dataset met die voorspellingen, meet hoe goed het ze nadoet, en lees het. Global, zoals in [Soorten uitleg](/course/open/rai-kinds-of-explanation?lang=nl): de kopie beschrijft het model voor alle rijen tegelijk, niet voor één persoon.

        **Fidelity is geen accuracy.** Accuracy vergelijkt de kopie met de werkelijkheid, fidelity met het model dat hij nadoet. Waar de kopie het eens is met het model, heeft hij het goed precies wanneer het model het goed heeft; zijn accuracy kan dus hoogstens zoveel van die van het model afwijken als het deel van de rijen waar de twee het oneens zijn. Himabindu Lakkaraju, Ece Kamar, Rich Caruana en Jure Leskovec (2017) meten fidelity als de agreement rate: het deel van de rijen in de data waar de benadering hetzelfde label geeft als de voorspelling van de black box. Ze waarschuwen ook dat fidelity en leesbaarheid elkaar tegenwerken: fidelity vraagt om details en nuances, leesbaarheid om eenvoud. Een diepere boom is het vaker eens, en lastiger te lezen. Molnar meet fidelity in plaats daarvan met R², dat past bij antwoorden die getallen zijn, zoals voorspelde kansen.

        **Vergeleken met wat?** In UCI Adult verdienen de meeste mensen 50.000 dollar of minder, en dus zegt het forest tegen de meesten nee. Een kopie die altijd nee zegt, is het dan eens met het forest overal waar het forest nee zegt. Vergelijk elke fidelity met dat getal. En er is geen afgesproken grens: Molnar schrijft over R² dat niet duidelijk is vanaf welke waarde je erop kunt vertrouwen dat de surrogate dicht genoeg bij de black box zit, en vraagt of dat 80%, 50% of 99% verklaarde variantie moet zijn.

        **Dichtbij voor de een, ver weg voor de ander.** Een fidelity over iedereen kan een groep verbergen. Het kan volgens Molnar gebeuren dat het model dat je kunt lezen voor het ene deel van de dataset heel dichtbij zit, maar voor een ander deel ver ernaast. Meet het dus ook per groep. De schuifbalk hieronder laat zien waar kopieën zoals in het uitgewerkte voorbeeld het oneens zijn met het forest.

        **Dezelfde antwoorden, andere redenen.** Cynthia Rudin (2019): een uitleg die het in 90% van de gevallen eens is met het oorspronkelijke model, legt dat model meestal uit, maar zit er in 10% van de gevallen naast. Erger nog, een kopie kan het eens zijn om andere redenen: zelfs een uitleg die bijna precies hetzelfde doet als een black box, kan heel andere features gebruiken, en is dan niet trouw aan wat de black box uitrekent. Rudin raadt aan zulke kopieën helemaal geen uitleg meer te noemen. En de kopie beschrijft het model, niet de wereld; Molnar: je trekt conclusies over het model en niet over de data, omdat de surrogate de echte uitkomst nooit ziet.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **black box** | een model dat je alleen van buiten ziet: je geeft het rijen, en het geeft antwoorden |
        | **global surrogate** | een model dat je kunt lezen, getraind op de antwoorden van een black box, zodat het zo vaak mogelijk dezelfde antwoorden geeft; ook surrogate model genoemd |
        | **fidelity** | hoe goed de antwoorden van de surrogate kloppen met die van het model dat hij nadoet: voor ja en nee de agreement rate, voor getallen vaak R². Meet het op rijen waarvan hij niet leerde, anders lijkt het beter dan het is |
        | **agreement rate** | het deel van de rijen waar de surrogate hetzelfde antwoord geeft als het model |
        | `export_text` | in scikit-learn: de vragen en bladeren van een boom als tekst, elk op een eigen regel |
        """),
        L.watch_out("Fidelity is geen accuracy, en geen reden. Een kopie die het bij 96% van de mensen eens is met het forest, zit voor de andere 4% naast het forest, kan voor één groep veel slechter zijn, en kan dezelfde antwoorden geven met andere features dan het forest gebruikt. Lees de kopie als een samenvatting van de antwoorden van het forest, niet van hoe het forest ze uitrekent."),
    ])
    return


@app.cell(hide_code=True)
def explore_depth_cell():
    explore_depth = L.explore("depth", "max_depth", lambda v: C.depth_picture(v, "nl"), start=3,
                              values=C.DEPTHS, role="data",
                              task="Schuif van diepte 1 naar 8. Bij welke opleidingen zit de kopie naast het forest, en hoeveel stijgt de fidelity van de ene diepte naar de volgende?")
    mo.vstack([
        mo.md("**Waar de kopie het oneens is met het forest, per diepte**"),
        explore_depth,
        mo.md(r"""
        Kopieën zoals in het uitgewerkte voorbeeld verderop: bomen met diepte 1 tot 8, gegroeid op de antwoorden van het forest op de helft om op te trainen. Elke staaf telt de mensen van de testhelft met die education_num bij wie de kopie het oneens is met het forest; gevuld waar de kopie ja zegt en het forest nee, open waar de kopie nee zegt en het forest ja. education_num 9 is een diploma van de middelbare school, 13 een bachelordiploma.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een model na te doen met een model dat je kunt lezen:

    1. Neem rijen waarop het model antwoord kan geven, en splits ze: rijen om de kopie op te laten groeien, en rijen om hem op te meten.
    2. Vraag het model om zijn antwoorden op allemaal.
    3. Laat een klein model dat je kunt lezen groeien, zoals een ondiepe boom, op de eerste rijen en de antwoorden van het model, niet op de echte labels.
    4. Meet de fidelity op de andere rijen, en vergelijk hem met een kopie die altijd het antwoord geeft dat het model het vaakst geeft.
    5. Meet de fidelity ook per groep, en lees de kopie alleen waar hij het eens is.
    6. Lees de kopie als een samenvatting van de antwoorden van het model: niet van de wereld, en niet van hoe het model ze uitrekent.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het forest uit [Hoe een voorspelling meebeweegt met een feature](/course/open/rai-pdp-ice?lang=nl), met dezelfde kolommen en helften. `_forest.estimators_` bevat zijn 100 bomen, en `.get_n_leaves()` telt de bladeren van één boom. De kopie is een boom met diepte 3, gegroeid op `_answers`, wat het forest zegt voor de helft om op te trainen, en `export_text(tree, feature_names=...)` geeft zijn vragen en bladeren als tekst, met de namen van de kolommen. `_same = _copy_says == _forest_says` is `True` voor elke persoon bij wie de twee het eens zijn, en `.mean()` van een array met `True` en `False` is het deel dat `True` is: `_same.mean()` is de fidelity. `np.full(n, False)` maakt een array met `n` keer `False`: een kopie die tegen iedereen nee zegt. Een boom zonder `max_depth` groeit door tot elk deel zuiver is. Voer de cel uit. Dat duurt een paar seconden.
    """)
    return


@app.cell
def example_surrogate():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier, export_text

    # 1. het forest uit de vorige les, gegroeid op de helft om op te trainen
    _people = pd.read_csv("data/adult.csv.gz")
    _X = _people[["age", "education_num", "hours_per_week"]].to_numpy().astype(float)
    _y = (_people["income"] == ">50K").to_numpy()
    _order = np.random.default_rng(0).permutation(len(_y))
    _train, _test = _order[: len(_y) // 2], _order[len(_y) // 2:]
    _forest = RandomForestClassifier(min_samples_leaf=20, random_state=0).fit(_X[_train], _y[_train])
    _leaves = 0
    for _tree in _forest.estimators_:
        _leaves = _leaves + _tree.get_n_leaves()
    print("bladeren in het forest:", _leaves)

    # 2 en 3. de antwoorden van het forest op de helft om op te trainen, en een boom met diepte 3 erop
    _answers = _forest.predict(_X[_train])
    _copy = DecisionTreeClassifier(max_depth=3, random_state=0).fit(_X[_train], _answers)
    print(export_text(_copy, feature_names=["age", "education_num", "hours_per_week"]))

    # 4. fidelity op de testhelft, naast een kopie die tegen iedereen nee zegt, en hoe vaak elk het goed heeft
    _forest_says = _forest.predict(_X[_test])
    _copy_says = _copy.predict(_X[_test])
    _same = _copy_says == _forest_says
    print("fidelity:", _same.mean().round(3))
    print("tegen iedereen nee zeggen:", (np.full(len(_test), False) == _forest_says).mean().round(3))
    print("goed, de kopie:", (_copy_says == _y[_test]).mean().round(3))
    print("goed, het forest:", (_forest_says == _y[_test]).mean().round(3))

    # 5. fidelity per opleiding
    _degree = _X[_test][:, 1] >= 13
    _rest = _X[_test][:, 1] <= 12
    print("opleiding 13 of meer:", _degree.sum(), "mensen, fidelity", _same[_degree].mean().round(3))
    print("opleiding tot 12:", _rest.sum(), "mensen, fidelity", _same[_rest].mean().round(3))

    # 6. een boom met diepte 3 op de echte labels, en een kopie die doorgroeit tot elk deel zuiver is
    _labels = DecisionTreeClassifier(max_depth=3, random_state=0).fit(_X[_train], _y[_train])
    print("op de echte labels, fidelity:", (_labels.predict(_X[_test]) == _forest_says).mean().round(3))
    _full = DecisionTreeClassifier(random_state=0).fit(_X[_train], _answers)
    print("doorgegroeid tot zuiver:", _full.get_n_leaves(), "bladeren")
    print("  fidelity op de helft om op te trainen:", (_full.predict(_X[_train]) == _answers).mean().round(3))
    print("  fidelity op de testhelft:", (_full.predict(_X[_test]) == _forest_says).mean().round(3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het forest heeft 28.295 bladeren; de kopie heeft er 8, en maar één daarvan zegt ja: education_num boven 12,5, age boven 31,5 en hours_per_week boven 39,5. Zijn andere vragen splitsen een deel in twee delen die allebei nee zeggen, omdat scikit-learn een splitsing kiest op Gini impurity, die kan dalen terwijl het antwoord hetzelfde blijft. De hele kopie laat zich dus lezen als één zin: ja voor mensen met minstens een bachelordiploma die 32 of ouder zijn en 40 uur of meer werken, en nee voor alle anderen. Hij is het eens met het forest bij 95,7% van de testhelft, en heeft het ongeveer even vaak goed als het forest: 79,6% tegen 79,8%. Maar een kopie die tegen iedereen nee zegt, is het al eens bij 85,2%, en de kopie past niet overal even goed: 98,8% bij de 18.374 mensen met een opleiding tot 12, en 86,3% bij de 6047 met 13 of meer, waar het forest de meeste van zijn ja's zegt. Een boom die op de echte labels groeide, is het minder vaak eens, 93,1%: hij leerde de wereld, niet het forest. En een kopie die doorgroeit tot elk deel zuiver is, is het eens bij 100% van de helft om op te trainen, de rijen waarvan hij leerde, en bij 99,8% van de testhelft, maar met 116 bladeren is hij geen samenvatting meer die iemand leest.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fidelity():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Hoe vaak zijn ze het eens?

        Schrijf `fidelity(copy, original)`: `copy` en `original` zijn lijsten met `True` en `False` van dezelfde lengte, één antwoord per persoon, en de functie geeft het deel van de plaatsen terug waar `copy` hetzelfde antwoord geeft als `original`. De check roept eerst `fidelity([True, False, False, True], [True, False, True, True])` aan, en daarna andere lijsten. De tekening voert je functie uit op de antwoorden van het forest en van de kopieën met diepte 1 tot 8, voor de 24.421 mensen van de testhelft.
        """),
        L.task("Schrijf `fidelity(copy, original)`: tel in `same` de plaatsen `i` waar `copy[i]` gelijk is aan `original[i]`, en geef dat aantal gedeeld door het aantal plaatsen terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_fidelity():
    box_fidelity = L.code_box("fidelity", lines=8)
    box_fidelity
    return (box_fidelity,)


@app.cell(hide_code=True)
def check_fidelity(box_fidelity):
    result_fidelity = L.check("fidelity", box_fidelity.value)
    result_fidelity
    return (result_fidelity,)


@app.cell(hide_code=True)
def hints_fidelity():
    L.hints(
        "fidelity",
        [
            "Tel de plaatsen waar beide lijsten hetzelfde zeggen, en deel door het aantal plaatsen. Plaatsen waar allebei `False` zeggen, tellen ook mee.",
            "Begin met `same = 0` en loop langs de plaatsen met `for i in range(len(copy)):`. Als de twee antwoorden op plaats `i` gelijk zijn, tel je 1 op bij `same`. Geef na de loop het deel terug.",
            "`return same / ___`",
        ],
        why="`same = 0`, dan `for i in range(len(copy)):` met `if copy[i] == original[i]:` en `same = same + 1`. Na de loop `return same / len(copy)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_stump():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De beste losse vraag

        De kleinste kopie stelt één vraag. Het vak bevat drie lijsten voor de helft om op te trainen, met één getal voor elke education_num van 1 tot 16: `people`, hoeveel mensen die hebben; `forest_yes`, tegen hoeveel van hen het forest ja zegt; en `true_yes`, hoeveel van hen echt meer dan 50.000 dollar verdienen. Een kopie met één vraag zegt ja als education_num minstens `t` is. Vul `scores` met zijn fidelity voor elke `t` van 1 tot 17: bij `t` = 1 zegt hij tegen iedereen ja, bij 17 tegen niemand. Loop binnen de loop over `t` langs de niveaus met `e` van 0 tot 15, met `same` als telling van de mensen voor wie de kopie zegt wat het forest zegt: niveau `e` is education_num `e + 1`. `sum(people)` telt een lijst op. De tekening laat je scores zien naast de antwoorden van het forest.
        """),
        L.task("Vervang de `...`: tel voor elk niveau bij `same` de mensen op voor wie de kopie zegt wat het forest zegt; voeg na de niveaus het deel van iedereen toe aan `scores`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_stump():
    box_stump = L.code_box("stump", lines=17)
    box_stump
    return (box_stump,)


@app.cell(hide_code=True)
def check_stump(box_stump):
    result_stump = L.check("stump", box_stump.value)
    result_stump
    return (result_stump,)


@app.cell(hide_code=True)
def hints_stump():
    L.hints(
        "stump",
        [
            "Voor één `t` is de kopie het eens met het forest bij elke persoon met education_num `t` of meer tegen wie het forest ja zegt, en bij elke persoon onder `t` tegen wie het forest nee zegt. Tel beide, en deel door iedereen.",
            "Begin binnen `for t in range(1, 18):` met `same = 0`, en dan `for e in range(16):`. Als `e + 1 >= t`, tel je de ja's van het forest op niveau `e` op; anders tel je zijn nee's op, de mensen min de ja's. Voeg na de loop over `e` het deel toe aan `scores`.",
            "`scores.append(same / ___)`",
        ],
        why="`same = 0`, dan `for e in range(16):` met `if e + 1 >= t:` en `same = same + forest_yes[e]`, `else:` en `same = same + people[e] - forest_yes[e]`. Na die loop `scores.append(same / sum(people))`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_between():
    mo.md(r"""
    ### Oefening 3. Het eens zijn en het goed hebben
    """)
    return


@app.cell(hide_code=True)
def choice_between_cell():
    choice_between = L.choice(
        "between",
        question="Een kopie is het bij 95% van de testhelft eens met het forest, en het forest heeft ongeveer 80% ervan goed. Hoe vaak kan de kopie het goed hebben?",
        options={
            "same": "80%, precies zoals het forest",
            "right": "Ergens tussen 75% en 85%",
            "fidelity": "95%, zijn fidelity",
        },
        correct="right",
        why={
            "right": "Goed. Waar de kopie het eens is met het forest, heeft hij het goed precies wanneer het forest het goed heeft. Op de andere 5% kan hij het goed hebben waar het forest het fout heeft, of fout waar het forest het goed heeft, dus zijn accuracy ligt hoogstens 5 procentpunt van die van het forest af. In het uitgewerkte voorbeeld: fidelity 95,7%, de kopie heeft 79,6% goed, het forest 79,8%.",
            "same": "Alleen waar de twee het eens zijn, heeft de kopie het goed precies wanneer het forest het goed heeft. Op de andere 5% kan hij het goed hebben waar het forest het fout heeft, of andersom, dus hij kan tot 5 procentpunt beter of slechter zijn. Zie *Fidelity is geen accuracy* in *Het idee*.",
            "fidelity": "Fidelity vergelijkt de kopie met het forest, niet met de werkelijkheid: een kopie die het eens is met het forest, zit fout waar het forest fout zit. Zijn accuracy ligt hoogstens 5 procentpunt van de 80% van het forest af. Zie *Fidelity is geen accuracy* in *Het idee*.",
        },
    )
    choice_between
    return (choice_between,)


@app.cell(hide_code=True)
def text_exercise_reasons():
    mo.md(r"""
    ### Oefening 4. Dezelfde antwoorden
    """)
    return


@app.cell(hide_code=True)
def choice_reasons_cell():
    choice_reasons = L.choice(
        "reasons",
        question="Een kopie is het bij 99% van de testhelft eens met het forest. Wat zegt dat over hoe het forest beslist?",
        options={
            "same": "Dat het forest dezelfde vragen in dezelfde volgorde stelt",
            "right": "Weinig: andere wegen geven dezelfde antwoorden",
            "all": "Alles: de kopie is het forest in het klein",
        },
        correct="right",
        why={
            "right": "Goed. Volgens Rudin kan een kopie die bijna hetzelfde doet heel andere features gebruiken, en is hij dan niet trouw aan wat de black box uitrekent. En 99% over iedereen kan een groep verbergen: in het uitgewerkte voorbeeld is de kopie met diepte 3 het eens bij 98,8% van de mensen met een opleiding tot 12 en bij 86,3% van wie 13 of meer heeft.",
            "same": "De kopie groeit op de antwoorden van het forest, niet op zijn bomen: hij kan dezelfde antwoorden bereiken met andere vragen, of andere features. Zie *Dezelfde antwoorden, andere redenen* in *Het idee*.",
            "all": "De kopie beschrijft de antwoorden van het forest, niet hoe het forest ze uitrekent, en de 1% die hij fout heeft kan allemaal in één groep zitten. Zie *Dezelfde antwoorden, andere redenen* in *Het idee*.",
        },
    )
    choice_reasons
    return (choice_reasons,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Welke features doen ertoe?](/course/open/rai-permutation-importance?lang=nl) en [Hoe een voorspelling meebeweegt met een feature](/course/open/rai-pdp-ice?lang=nl) bekeken het forest één feature tegelijk; deze les doet het helemaal na met één boom, zoals in [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl), maar gegroeid op de antwoorden van het forest. Net als die twee is een global surrogate een global explanation, en model-agnostic: hij heeft alleen de antwoorden van het model nodig, in de indeling van [Soorten uitleg](/course/open/rai-kinds-of-explanation?lang=nl). De volgende les, [LIME: een eenvoudig model rond één voorspelling](/course/open/rai-lime?lang=nl), doet het model ook na, maar alleen dicht bij één persoon, om één beslissing uit te leggen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_between, choice_reasons, result_fidelity, result_stump):
    L.wrap_up(result_fidelity, result_stump, choice_between, choice_reasons)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
