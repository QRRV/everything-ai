# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-data-leakage. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-data-leakage", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Betrouwbaarder kiezen: cross-validation](/course/open/ml-cross-validation?lang=nl) en [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_fold():
    quick_fold = L.choice(
        "quick-fold",
        question="5-fold cross-validation op 50 voorbeelden, 10 in elke fold. Op welke voorbeelden traint het model in de ronde van fold 3?",
        options={
            "all": "Op alle 50",
            "right": "Op de 40 voorbeelden van de andere vier folds",
            "fold": "Op de 10 voorbeelden van fold 3",
        },
        correct="right",
        why={
            "all": "Dan zou het ook trainen op de 10 voorbeelden waarop het in deze ronde gemeten wordt. Zie het recept in [Betrouwbaarder kiezen: cross-validation](/course/open/ml-cross-validation?lang=nl).",
            "fold": "Op die voorbeelden wordt het in deze ronde gemeten. Zie het recept in [Betrouwbaarder kiezen: cross-validation](/course/open/ml-cross-validation?lang=nl).",
            "right": "Het traint op de andere folds en wordt gemeten op fold 3, dus elk voorbeeld wordt één keer gecontroleerd door een model dat er niet op trainde.",
        },
    )
    quick_fold
    return (quick_fold,)


@app.cell(hide_code=True)
def quick_check_scale():
    quick_scale = L.choice(
        "quick-scale",
        question="Een model trainde op gestandaardiseerde features. Met welk gemiddelde en welke standaardafwijking standaardiseer je een nieuw voorbeeld?",
        options={
            "own": "Met die van het voorbeeld zelf",
            "right": "Met die van de training set",
            "all": "Met die van de training set en de nieuwe voorbeelden samen",
        },
        correct="right",
        why={
            "own": "Eén voorbeeld heeft geen eigen spreiding, en het model leerde zijn weights voor de schaal van de training set. Zie stap 4 van het recept in [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl).",
            "all": "Het model leerde zijn weights op de schaal van de training set, en die schaal hoort niet te veranderen met de voorbeelden die je toevallig voorspelt. Zie stap 4 van het recept in [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl).",
            "right": "Het gemiddelde en de standaardafwijking leer je van de training set, net als de weights, en ze blijven zoals ze zijn voor elk nieuw voorbeeld.",
        },
    )
    quick_scale
    return (quick_scale,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Cross-validation controleert elk voorbeeld één keer, met een model dat er niet op trainde. Dat werkt alleen als er niets over die voorbeelden op een andere manier in het model terechtkwam. Neem 50 mensen en geef elk van hen 1000 getallen, willekeurig getrokken tussen 0 en 1, en een label 0 of 1, ook willekeurig getrokken, de helft van elk. Niets in die getallen kan het label voorspellen: op nieuwe mensen heeft elk model ongeveer de helft goed. Met 1000 kolommen en maar 50 mensen lijkt het verstandig om eerst de nuttigste kolommen te houden: de 20 waarvan het gemiddelde het meest verschilt tussen de mensen met label 1 en de mensen met label 0. Daarna 5-fold cross-validation, met de 5 dichtstbijzijnde buren uit [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl) op die 20 kolommen.

    **Een voorbeeld uit het dagelijks leven.** Rond 1904 leek in Berlijn een paard, dat bekend werd als de slimme Hans, te kunnen rekenen: vroeg je hoeveel 3 plus 2 is, dan tikte hij vijf keer met zijn hoef. Eind 1904 ontdekte de psycholoog Oskar Pfungst dat Hans het alleen goed had als degene die de vraag stelde het antwoord wist, en Hans die persoon kon zien; in 1907 beschreef hij het in een boek. Hans hield op met tikken als die zich ontspande, zonder dat de vraagsteller merkte dat die het antwoord verklapte. Het antwoord lekte via degene die de vraag stelde.
    """)
    return


@app.cell(hide_code=True)
def predict_noise():
    guess_noise = L.guess(
        "noise-score",
        question="Welk deel van de 50 mensen heeft de cross-validation goed, op deze willekeurige getallen?",
        options={
            "half": "Ongeveer 0,5: er valt niets te leren",
            "some": "Ongeveer 0,7",
            "right": "Bijna allemaal",
        },
        reveal="**0,96**, 48 van de 50 mensen. Toch hebben dezelfde 20 kolommen en 5 buren van 1000 nieuwe mensen er 52,1% goed: gokken. De 20 kolommen zijn gekozen met de labels van alle 50 mensen, dus ook met de labels van de 10 mensen waarop elke fold gemeten wordt. Het antwoord lekte erin, net als bij Hans.",
    )
    guess_noise
    return (guess_noise,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Data leakage** is informatie die in de training van een model terechtkomt en die het model niet heeft als het echt voorspelt: meestal informatie over de voorbeelden waarop het gemeten wordt. De score meet dan iets wat nieuwe voorbeelden niet meebrengen, en valt te goed uit. Het gebeurt meestal op een van drie manieren.

        1. **Een keuze gemaakt met de voorbeelden waarop je meet.** Hier is dat **feature selection**, kiezen welke features een model krijgt: de 20 kolommen zijn gekozen met de labels van alle 50 mensen. Hetzelfde geldt voor het gemiddelde en de standaardafwijking om te standaardiseren, en voor de waarden die gaten vullen: alles wat van data geleerd wordt, hoort bij de training, dus je leert het alleen van de trainingsvoorbeelden, en bij cross-validation in elke fold opnieuw. Ook hyperparameters: de score waarmee je ze koos, valt iets te goed uit, en daarom blijft de test set tot het allerlaatst apart.
        2. **Target leakage**: een feature die pas na het label bekend is, of eruit gemaakt is, zoals of een patiënt het medicijn kreeg tegen precies de ziekte die het model moet voorspellen.
        3. **Hetzelfde voorbeeld, of dezelfde persoon, aan beide kanten**: een rij die twee keer in de tabel staat, of meer foto's van één patiënt, een paar om op te trainen en een paar om op te meten. Een **group split** houdt elke groep heel, aan één kant: verdeel per patiënt, niet per foto.

        Waarom werkt het zo goed op ruis? Van 1000 kolommen willekeurige getallen verschillen er een paar puur door toeval tussen de twee labels, en de beste 20 verschillen het meest. Gekozen op alle 50 mensen verschillen die kolommen ook tussen de labels van de 10 mensen waarop een fold gemeten wordt, door datzelfde toeval, en de buren vinden dat. Gekozen op de 40 trainingsmensen van de fold zeggen ze niets over de andere 10.

        Sayash Kapoor en Arvind Narayanan (2023) zetten op een rij wat anderen vonden: data leakage in 294 gepubliceerde papers, in 17 vakgebieden die machine learning gebruiken.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **data leakage** | informatie in de training die het model niet heeft als het echt voorspelt, zoals informatie over de voorbeelden waarop het gemeten wordt | |
        | **feature selection** | kiezen welke features een model krijgt | `select(X[train], y[train], 20)` |
        | **target leakage** | een feature die pas na het label bekend is, of eruit gemaakt is | |
        | `KNeighborsClassifier(n_neighbors=5)` | k-nearest neighbors van scikit-learn met k = 5: `fit` bewaart de trainingsvoorbeelden, en `predict` geeft elk nieuw voorbeeld het label dat de meeste van zijn 5 dichtstbijzijnde buren hebben | |
        | **group split** | een verdeling die elke groep voorbeelden die bij elkaar horen, zoals de foto's van één patiënt, aan één kant houdt | `GroupKFold(n_splits=5)` |
        """),
        L.watch_out("Cross-validation beschermt je niet vanzelf. Het controleert het model dat in elke fold traint, niet de stappen vóór het verdelen: een stap die één keer op alle voorbeelden gedaan is, zit in elke fold. Schudden of meer folds helpen ook niet."),
        mo.md(r"""
        Het beeld laat tabellen met ruis zoals deze zien, met 10 tot 10000 kolommen. Links, voor één tabel: de 20 gekozen kolommen, grootste verschil eerst. Een volle stip is het verschil tussen de gemiddelden van de twee labels bij de 50 mensen, wat de selectie zag; een open grijze stip is het verschil in dezelfde kolom bij 1000 nieuwe mensen. Rechts, voor tien van zulke tabellen, de score van cross-validation met de kolommen gekozen op alle 50 mensen (volle stippen) en binnen elke fold (open stippen); de stippellijn is gokken, 0,5, en de grote stippen zijn de tabel links. Verschuif de slider, of sleep over die grafiek.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_columns_cell():
    def _frame(n):
        n = int(n)
        leaky, honest = C.MEASURED[True][n][C.EXAMPLE], C.MEASURED[False][n][C.EXAMPLE]
        seen, new = C.chosen_gaps(n)
        alt = (f"De eerste grafiek: de {min(n, C.KEEP)} gekozen kolommen van een tabel met ruis met {n} kolommen, grootste verschil eerst. Bij de 50 mensen lopen de verschillen tussen de gemiddelden van de twee labels van {L.number(seen[0])} tot {L.number(seen[-1])}; bij 1000 nieuwe mensen liggen ze allemaal onder {L.number(max(new))}. "
               + "De tweede grafiek: voor 10 tot 10000 kolommen de scores van cross-validation van tien tabellen. Met de kolommen gekozen op alle 50 mensen stijgen ze van ongeveer 0,5 tot tussen 0,92 en 0,98; met de kolommen gekozen binnen elke fold blijven ze tussen 0,3 en 0,66, rond gokken. "
               + f"Met {n} kolommen scoort deze tabel {L.number(leaky)} en {L.number(honest)}.")
        said = f"""{n} kolommen: gekozen op alle 50 mensen zegt cross-validation **{L.number(leaky)}**; gekozen binnen elke fold **{L.number(honest)}**. De gekozen kolommen verschillen gemiddeld {L.number(seen.mean(), 3)} tussen de labels van de 50, en {L.number(new.mean(), 3)} bij nieuwe mensen."""
        return (C.picture(n, {"title": "De 20 gekozen kolommen", "rank": "gekozen kolom, grootste verschil eerst", "gap": "verschil tussen de gemiddelden",
                              "every": "Tien tabellen met ruis", "columns": "kolommen", "score": "score van cross-validation",
                              "leaky": "gekozen op alle 50", "honest": "gekozen binnen elke fold", "alt": alt}), said)

    explore_columns = L.explore("columns", "kolommen", _frame, start=20, values=C.COLUMNS, role="",
                                task="Verschuif het aantal kolommen van 10 naar 10000. Wat gebeurt er met de verschillen van de gekozen kolommen, en met de twee scores?")
    explore_columns
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Met 10 of 20 kolommen houdt de selectie ze allemaal, dus zijn beide scores gelijk, rond 0,5. Hoe meer kolommen er zijn om uit te kiezen, hoe groter de verschillen van de beste 20 door toeval: gemiddeld 0,075 bij 20 kolommen, 0,21 bij 1000 en 0,28 bij 10000, terwijl ze bij nieuwe mensen rond 0,01 blijven. De score met de kolommen gekozen op alle 50 mensen stijgt mee, tot tussen 0,82 en 1 vanaf 1000 kolommen. Binnen elke fold gekozen blijven de scores tussen 0,3 en 0,66: gokken, met de spreiding die 50 mensen geven.

    ### Recept

    Zo meet je een model eerlijk met cross-validation:

    1. Verdeel eerst: snijd de voorbeelden in folds, en houd de test set apart, voordat er iets van ze geleerd wordt.
    2. Leer in elke fold elke stap die van data leert alleen van de trainingsvoorbeelden van die fold: de features die je houdt, het gemiddelde en de standaardafwijking om te standaardiseren, de waarden die gaten vullen.
    3. Pas die stappen, zoals ze geleerd zijn, toe op de voorbeelden waarop de fold gemeten wordt, en meet.
    4. Vraag bij elke feature: zou ik dit weten op het moment dat ik voorspel? Zo niet, laat hem weg.
    5. Horen voorbeelden bij elkaar, zoals de foto's van één patiënt, verdeel ze dan per groep: `GroupKFold(n_splits=5)` van scikit-learn geeft de folds, met `split(X, y, groups)`.
    6. Lijkt een score te mooi om waar te zijn? Meet hem op voorbeelden die het model en zijn stappen nooit gezien hebben.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De tabel met ruis uit *Waarom*: 50 mensen en 1000 kolommen, getrokken met seed 6, en 1000 nieuwe mensen uit dezelfde generator. `_select` houdt de k kolommen met het grootste verschil tussen de gemiddelden van de twee labels; `np.argsort` geeft de posities van de verschillen van klein naar groot, dus de laatste k zijn de posities van de grootste. De cross-validation is die uit [Betrouwbaarder kiezen: cross-validation](/course/open/ml-cross-validation?lang=nl), en telt de mensen die hij goed heeft, met `KNeighborsClassifier` van scikit-learn voor de 5 dichtstbijzijnde buren die je in [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl) met de hand telde, maar de kolommen worden één keer gekozen, vóór de folds, met alle 50 mensen: dat breekt stap 2 van het recept. Daarna dezelfde stappen op alle 50 mensen, gemeten op de nieuwe mensen zoals in stap 6. Het commentaar met `#` zegt wat elk stuk doet. Voer de cel uit.
    """)
    return


@app.cell
def example_leak():
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier

    _rng = np.random.default_rng(6)
    _X = _rng.random(size=(50, 1000))
    _y = _rng.permutation([0, 1] * 25)
    _X_new = _rng.random(size=(1000, 1000))
    _y_new = _rng.permutation([0, 1] * 500)


    def _select(X, y, k):
        # de k kolommen waarvan het gemiddelde het meest verschilt tussen label 1 en label 0
        gap = abs(X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0))
        return np.argsort(gap)[-k:]


    # het lek: de kolommen worden één keer gekozen, met de labels van alle 50 mensen
    _cols = _select(_X, _y, 20)
    _rows = np.arange(50)
    _right = 0
    for _f in range(5):
        _val = _rows[10 * _f:10 * (_f + 1)]
        _train = np.concatenate([_rows[:10 * _f], _rows[10 * (_f + 1):]])
        _model = KNeighborsClassifier(n_neighbors=5).fit(_X[_train][:, _cols], _y[_train])
        _right = _right + np.sum(_model.predict(_X[_val][:, _cols]) == _y[_val])
    print(_right / 50)
    # stap 6: dezelfde stappen op alle 50 mensen, gemeten op 1000 nieuwe mensen
    _model = KNeighborsClassifier(n_neighbors=5).fit(_X[:, _cols], _y)
    print(np.mean(_model.predict(_X_new[:, _cols]) == _y_new))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Cross-validation zegt 0,96; de nieuwe mensen krijgen 0,521, een haar boven gokken. `_X[_train][:, _cols]` neemt eerst de rijen van de trainingsmensen, en dan de gekozen kolommen: `[:, _cols]` houdt de kolommen waarvan de positie in `_cols` staat, zoals `X[rows]` rijen houdt. Elk model in de folds traint alleen op zijn eigen 40 mensen, zoals het hoort bij cross-validation; het lek zit in de regel vóór de loop. Met de kolommen gekozen binnen elke fold scoort deze tabel 0,52. In oefening 1 schrijf je dat voor een andere tabel.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_folds():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Binnen de folds

        Het vak trekt een andere tabel met ruis, met seed 2, en heeft de cross-validation uit het uitgewerkte voorbeeld, met `select` als functie. Alleen de kolommen ontbreken nog: vervang de `...` in de loop, zodat elke fold zijn eigen 20 kolommen kiest, uit zijn eigen trainingsmensen. De check voert je code uit en kijkt naar de score op de laatste regel; hij probeert ook een andere tabel.
        """),
        L.task("Kies de kolommen binnen elke fold."),
    ])
    return


@app.cell(hide_code=True)
def exercise_folds():
    box_folds = L.code_box("inside-folds", lines=21)
    box_folds
    return (box_folds,)


@app.cell(hide_code=True)
def check_folds(box_folds):
    result_folds = L.check("inside-folds", box_folds.value)
    result_folds
    return (result_folds,)


@app.cell(hide_code=True)
def hints_folds():
    L.hints(
        "inside-folds",
        [
            "Het uitgewerkte voorbeeld koos de kolommen één keer, vóór de loop, met `_X` en `_y`. Hier kiest elke fold ze opnieuw, binnen de loop: met welke rijen van `X`, en welke labels?",
            "`train` bevat de posities van de 40 mensen waarop deze fold traint. `X[train]` zijn hun rijen; welke labels horen daarbij?",
            "`cols = select(X[___], y[___], 20)`.",
        ],
        why="Binnen de loop zijn `X[train]` en `y[train]` de 40 mensen waarop de fold traint. Kolommen die uit hen gekozen zijn, weten niets over de 10 mensen van `val`, dus de score op die 10 is eerlijk, zoals hij ook voor nieuwe mensen zal zijn.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_known():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Bekend op 1 september?

        Een school wil op 1 september voorspellen welke leerlingen in juni slagen voor hun eindexamen, zodat ze de anderen op tijd kan helpen. Van de leerlingen van eerdere jaren heeft ze de kolommen hieronder, en of ze slaagden. Welke mogen een feature zijn, en welke lekken?
        """),
        L.task("Zet elke kolom in zijn groep."),
    ])
    return


@app.cell(hide_code=True)
def exercise_known():
    sort_known = L.sort("known-then")
    sort_known
    return (sort_known,)


@app.cell(hide_code=True)
def check_known(sort_known):
    result_known = L.check("known-then", sort_known)
    result_known
    return (result_known,)


@app.cell(hide_code=True)
def text_exercise_patient():
    mo.md(r"""
    ### Oefening 3. Tien foto's van elke patiënt
    """)
    return


@app.cell(hide_code=True)
def choice_patient_cell():
    choice_patient = L.choice(
        "same-patient",
        question="Een model leert een huidziekte te herkennen op foto's: 1000 foto's van 100 patiënten, tien van elk. De foto's worden willekeurig gesplitst, 800 om op te trainen en 200 om op te testen, en het model heeft 95% van de testfoto's goed. In een ander ziekenhuis heeft het 70% goed. Wat ging er eerst mis?",
        options={
            "few": "200 testfoto's zijn te weinig om iets te zeggen",
            "right": "Foto's van dezelfde patiënten staan aan beide kanten: het model kan de patiënt herkennen in plaats van de ziekte",
            "folds": "Ze hadden cross-validation moeten gebruiken in plaats van één test set",
        },
        correct="right",
        why={
            "few": "Op 200 foto's is 95% tegen 70% veel meer dan toeval. Wat hebben de testfoto's gemeen met de trainingsfoto's?",
            "folds": "Cross-validation met willekeurig verdeelde foto's heeft in elke fold hetzelfde probleem. Wiens foto's zitten in elke fold?",
            "right": "Met tien foto's van elke patiënt heeft bijna elke testfoto foto's van dezelfde patiënt in de training: dezelfde huid, hetzelfde licht. De test mat patiënten die het model al gezien had. Een group split per patiënt, zoals `GroupKFold`, houdt alle foto's van een patiënt aan één kant.",
        },
    )
    choice_patient
    return (choice_patient,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het lek zat verstopt in de folds uit [Betrouwbaarder kiezen: cross-validation](/course/open/ml-cross-validation?lang=nl). De regel is die uit [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl) voor het gemiddelde en de standaardafwijking, nu voor elke stap die van data leert; en net als in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl) is een score alleen eerlijk over de voorbeelden waarmee niets gekozen is. Language models hebben een eigen lek: testvragen die op internet staan, komen in hun trainingstekst terecht; daarover gaat *Testdata uit de training data houden*. In de volgende les, *Elke stap in elke fold: pipelines*, herhaalt scikit-learn elke stap die van data leert voor je binnen elke fold, zonder eigen loop.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_patient, result_folds, result_known):
    L.wrap_up(result_folds, result_known, choice_patient)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
