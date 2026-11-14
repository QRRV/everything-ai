# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-imbalanced. Route: course/route/lessen/klassiek-ml.yaml. Checks en plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-imbalanced", "nl")
    C = L.checks   # de runs van de machines uit AI4I 2020, de modellen met en zonder weights, en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl) en één over [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_precision():
    quick_f1 = L.choice(
        "quick-f1",
        question="Een model heeft precision 0,8 en recall 0,2. Wat is zijn F1 score?",
        options={
            "mean": "0,5",
            "right": "0,32",
            "product": "0,16",
        },
        correct="right",
        why={
            "mean": "0,5 is het gewone gemiddelde. De F1 score is $2PR / (P + R) = 2 \\cdot 0{,}8 \\cdot 0{,}2 / 1{,}0 = 0{,}32$: die blijft dicht bij de kleinste van de twee. Zie *Het idee* in [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl).",
            "product": "0,16 is het product. De F1 score is twee keer het product, gedeeld door de som: $2 \\cdot 0{,}8 \\cdot 0{,}2 / (0{,}8 + 0{,}2) = 0{,}32$. Zie *Het idee* in [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl).",
            "right": "$2 \\cdot 0{,}8 \\cdot 0{,}2 / (0{,}8 + 0{,}2) = 0{,}32$, dicht bij de kleinste van de twee. Het eerste model van vandaag heeft ook een hoge precision en een lage recall, en je ziet hoe je de een tegen de ander afweegt als storingen zeldzaam zijn.",
        },
    )
    quick_f1
    return (quick_f1,)


@app.cell(hide_code=True)
def quick_check_nearest():
    quick_nearest = L.choice(
        "quick-nearest",
        question="`distances` bevat de afstand van een nieuw voorbeeld tot elk voorbeeld in de training set. Wat geeft de plekken van zijn 5 naaste voorbeelden in de training set?",
        options={
            "values": "`np.sort(distances)[:5]`",
            "far": "`np.argsort(distances)[-5:]`",
            "right": "`np.argsort(distances)[:5]`",
        },
        correct="right",
        why={
            "values": "`np.sort` geeft de 5 kleinste afstanden zelf, niet waar ze staan. `np.argsort` geeft de plekken, van dichtbij naar ver. Zie het uitgewerkte voorbeeld van [Vraag het de buren](/course/open/ml-knn?lang=nl).",
            "far": "`np.argsort` gaat van dichtbij naar ver, dus de laatste 5 plekken zijn de 5 verste. Zie het uitgewerkte voorbeeld van [Vraag het de buren](/course/open/ml-knn?lang=nl).",
            "right": "Het nieuwe voorbeeld zit niet tussen de voorbeelden in de training set. Vandaag wel: dan is het naaste van allemaal het voorbeeld zelf, op afstand 0.",
        },
    )
    quick_nearest
    return (quick_nearest,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een fabriek wil weten welke runs van een machine in een storing eindigen, zodat ze die op tijd kan nakijken. Storingen zijn zeldzaam: in de dataset AI4I 2020 zijn het er 339 van de 10.000 runs, 3,4%. De dataset is synthetisch: Stephan Matzka maakte hem in 2020 zo dat hij lijkt op echte onderhoudsdata, die bedrijven zelden publiceren. Een model dat leert zo vaak mogelijk gelijk te hebben, leert "geen storing" te zeggen, en de storingen, de voorbeelden waar het om gaat, raken zoek. Hetzelfde gebeurt bij fraude, zeldzame ziektes en een kapot onderdeel aan een productielijn: de klasse die je wilt vinden, is de zeldzame.

    **Een voorbeeld uit het dagelijks leven.** Een rookmelder die nooit afgaat, heeft op bijna elke dag van het jaar gelijk. Hij heeft het mis op precies de dag dat het ertoe doet.
    """)
    return


@app.cell(hide_code=True)
def predict_always_cell():
    guess_always = L.guess(
        "always-guess",
        question="Van 10.000 runs eindigen er 339 in een storing. Een model zegt elke keer \"geen storing\". Welk deel van de runs heeft het goed?",
        options={
            "half": "Ongeveer 50%",
            "right": "Ongeveer 97%",
            "few": "Ongeveer 3%",
        },
        reveal="**96,6%.** Het heeft elk van de 9.661 runs zonder storing goed, en vindt geen enkele van de 339 storingen: recall 0%. Een model dat nooit waarschuwt, heeft bijna altijd gelijk, en is nutteloos.",
    )
    guess_always
    return (guess_always,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Meestal goed, weinig gevonden.** De course trainde logistische regressie op 7.000 van de runs, met de vijf getallen van elke run: de temperatuur van de lucht en van het proces, het toerental, het koppel en de slijtage van het gereedschap. Op de andere 3.000 runs, met 110 storingen, heeft het model 97,2% goed, nauwelijks meer dan elke keer "geen storing", 96,3%. Het vindt 31 van de 110 storingen: recall 28,2%. De loss is een gemiddelde over alle 7.000 runs, en de 229 storingen zijn daar 3,3% van. Dus geeft het model de meeste runs een kleine kans op een storing, en komen maar weinig kansen tot 0,5.

        **Class weights.** Geef elk voorbeeld een weight, en laat zijn log loss zo vaak meetellen als die weight zegt. Een **class weight** is de weight van elk voorbeeld van één klasse. Met gebalanceerde class weights weegt elke klasse in totaal even zwaar: een voorbeeld van klasse $c$ krijgt

        $$w_c = \frac{n}{2\,n_c},$$

        met $n$ voorbeelden, waarvan $n_c$ van klasse $c$. Van de 7.000 runs zijn er 229 storingen: elke storing weegt $7.000 / (2 \cdot 229) = 15{,}28$, elke normale run $7.000 / (2 \cdot 6.771) = 0{,}517$, en beide klassen wegen in totaal 3.500. In scikit-learn is het `class_weight="balanced"`. Je schrijft de weights in oefening 1.

        **De aantallen veranderen.** In plaats van weights kun je de training set veranderen. **Undersampling** houdt elke storing en gooit willekeurig normale runs weg, tot er van elk 229 over zijn. **Oversampling** herhaalt willekeurig storingen, tot er van elk 6.771 zijn.

        **SMOTE.** Een storing herhalen geeft kopieën. **SMOTE** (Chawla en anderen, 2002) maakt nieuwe storingen: neem een storing $a$, een van zijn $k$ naaste andere storingen $b$, gevonden zoals in [Vraag het de buren](/course/open/ml-knn?lang=nl), en een willekeurig getal $u$ tussen 0 en 1. De nieuwe storing is

        $$a + u\,(b - a),$$

        een punt op de lijn van $a$ naar $b$: met $u = 0$ is het $a$, met $u = 1$ is het $b$. Je schrijft het in oefening 2.

        **Wat ze echt veranderen.** De course trainde dezelfde logistische regressie op nog vier manieren, probeerde ook het model zoals het is met een lagere threshold, en telde op de 3.000 test-runs:

        | Getraind | Storingen gevonden (van 110) | Alarmen | Recall | Precision | AUC |
        |---|---|---|---|---|---|
        | zoals het is | 31 | 35 | 28,2% | 88,6% | 0,888 |
        | met gebalanceerde class weights | 89 | 630 | 80,9% | 14,1% | 0,899 |
        | na undersampling | 91 | 684 | 82,7% | 13,3% | 0,896 |
        | na oversampling | 89 | 629 | 80,9% | 14,1% | 0,900 |
        | na SMOTE, k = 5 | 89 | 622 | 80,9% | 14,3% | 0,901 |
        | zoals het is, threshold 0,031 | 87 | 630 | 79,1% | 13,8% | 0,888 |

        Alle vier de correcties laten het model veel vaker "storing" zeggen: het vindt 89 tot 91 van de 110 storingen, met 622 tot 684 alarmen. De AUC uit [Alle thresholds tegelijk](/course/open/ml-roc-auc?lang=nl), hoe goed het model storingen boven normale runs zet, beweegt nauwelijks: 0,888 zonder, 0,896 tot 0,901 met. En het model zoals het is doet bijna hetzelfde met een lagere threshold: bij 0,031 in plaats van 0,5 geeft het evenveel alarmen als met gebalanceerde class weights, 630, en vindt het 87 storingen. Hier verschuiven de weights en het resamplen vooral de threshold; ze leren het model weinig nieuws. Wat er overblijft om te kiezen, is hoe snel het "storing" mag zeggen: wat kost meer, een gemiste storing of een onnodige controle?

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **class weight** | de weight van elk voorbeeld van één klasse in de loss | `class_weight="balanced"` |
        | **undersampling** | de training set in balans brengen door willekeurig voorbeelden van de veelvoorkomende klasse weg te gooien | |
        | **oversampling** | de training set in balans brengen door willekeurig voorbeelden van de zeldzame klasse te herhalen | |
        | **SMOTE** | oversampling met nieuwe voorbeelden, elk tussen een voorbeeld van de zeldzame klasse en een van zijn naaste buren uit die klasse | `a + u * (b - a)` |
        | $w_c$ | de gebalanceerde weight van een voorbeeld van klasse $c$: $n / (2\,n_c)$ | |
        """),
        L.watch_out("Resample alleen het trainingsdeel, na de splitsing. De validation set en de test set moeten lijken op de runs die het model gaat tegenkomen, met ongeveer 3,4% storingen. Oversamplen vóór de splitsing zet kopieën van dezelfde storing aan beide kanten, en de test set beloont dan het onthouden ervan."),
        mo.md(r"""
        De slider zet de class weight van een storing; een normale run houdt weight 1. De grijze lijn toont het model zonder weights bij elke threshold, zoals in de precision-recall curve uit [Alle thresholds tegelijk](/course/open/ml-roc-auc?lang=nl): de recall naar rechts, de precision omhoog. De grote stip is het model dat met die weight getraind is, bij threshold 0,5, en de open stippen zijn de andere weights. Beweeg de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_weight_cell():
    def _frame(w):
        said = C.weight_said(w, L.lang)
        alt = ("De precision van het model zonder weights tegen zijn recall, bij elke threshold, dalend van linksboven "
               "naar rechtsonder; stippen voor de modellen die getraind zijn met een class weight van 1, 2, 5, 10, 30 "
               "en 100 liggen dicht bij die lijn. " + said)
        return (C.weight_picture(w, {"title": "Weight van een storing", "x": "recall (%)", "y": "precision (%)",
                                     "weight": "weight {w}", "alt": alt}), said)

    explore_weight = L.explore("failure-weight", "weight van een storing", _frame, start=1, values=C.WEIGHTS, role="hyper",
                               task="Ga van weight 1 naar 100. Wat doet de recall, wat de precision, en waar liggen de stippen?")
    explore_weight
    return


@app.cell(hide_code=True)
def text_after_weight():
    mo.md(r"""
    Van weight 1 tot 100 vindt het model 31, 43, 63, 72, 89 en 102 van de 110 storingen, met 35, 63, 142, 274, 636 en 1.282 alarmen. Elke stip ligt dicht bij de grijze lijn: een grotere weight schuift het model erlangs, zoals een lagere threshold doet. Weight 30 is ongeveer de gebalanceerde weight: 6.771 / 229 = 29,6.

    ### Recept

    Als één klasse zeldzaam is:

    1. Maak eerst de splitsing. De validation set en de test set houden het echte aandeel van de zeldzame klasse.
    2. Beoordeel op de precision en recall van de zeldzame klasse, of hun F1 score, nooit alleen op accuracy.
    3. Probeer eerst class weights: die veranderen niets aan de data. Of houd het model en verlaag zijn threshold.
    4. Kies de weight of de threshold op de validation set, naar wat een gemist zeldzaam voorbeeld kost tegenover een onnodig alarm.
    5. Resample, als je het doet, alleen het trainingsdeel: undersampling als er data genoeg is, oversampling of SMOTE als er weinig is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel leest de 10.000 runs, houdt de vijf getallen en het label `Machine failure`, 1 voor een storing, splitst de runs willekeurig in 3.000 om op te testen en 7.000 om op te trainen, en schaalt de getallen met het gemiddelde en de standaardafwijking van het trainingsdeel. `LogisticRegression` is de logistische regressie uit [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl), met `fit` en `predict` zoals in [Hetzelfde model met scikit-learn](/course/open/ml-sklearn?lang=nl). De cel traint het model twee keer, zonder class weights (`None`) en met `class_weight="balanced"`, en print voor beide hoeveel van de storingen in de test set het vindt, hoeveel alarmen het geeft, de precision en de recall. Voer de cel uit.
    """)
    return


@app.cell
def example_machines():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    _df = pd.read_csv("data/ai4i-2020.csv.gz")
    _numbers = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    _X = np.array(_df[_numbers])
    _y = np.array(_df["Machine failure"])
    print(len(_y), _y.sum())

    _order = np.random.default_rng(0).permutation(len(_y))
    _test, _train = _order[:3000], _order[3000:]           # 3.000 runs om op te testen, 7.000 om op te trainen
    _mean, _std = _X[_train].mean(axis=0), _X[_train].std(axis=0)
    _X = (_X - _mean) / _std                               # geschaald met de getallen van het trainingsdeel

    for _weights in [None, "balanced"]:
        _model = LogisticRegression(class_weight=_weights)
        _model.fit(_X[_train], _y[_train])
        _alarm = _model.predict(_X[_test]) == 1            # True waar het model zegt: storing
        _found = _y[_test][_alarm].sum()                   # de echte storingen tussen de alarmen
        print(_weights, _found, _alarm.sum(), round(_found / _alarm.sum(), 3), round(_found / _y[_test].sum(), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    10.000 runs, 339 storingen. Zonder weights geeft het model 35 alarmen, en 31 daarvan zijn storingen: precision 88,6%, maar het vindt er slechts 31 van de 110, recall 28,2%. Met gebalanceerde class weights geeft het 630 alarmen en vindt het er 89: recall 80,9%, precision 14,1%. Wat beter is, hangt af van wat een gemiste storing kost, en wat een onnodige controle.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_weights():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Gebalanceerde weights

        `balanced_weights(y)` krijgt de labels van het trainingsdeel, 1 voor een storing en 0 voor een normale run, en moet de weight van elk voorbeeld geven: $n / (2\,n_c)$, met $n_c$ het aantal storingen voor een storing en het aantal normale runs voor een normale run. `weights[y == 1] = 2.0` zou 2.0 zetten op elke plek waar `y` 1 is. Vervang de twee `...`. De check roept `balanced_weights(y_train)` aan met de 7.000 labels van het trainingsdeel, `_y[_train]` in het uitgewerkte voorbeeld, en probeert drie korte arrays.
        """),
        L.task("Laat `balanced_weights` elke storing en elke normale run zijn gebalanceerde weight geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_weights():
    box_weights = L.code_box("balanced-weights", lines=9)
    box_weights
    return (box_weights,)


@app.cell(hide_code=True)
def check_weights(box_weights):
    result_weights = L.check("balanced-weights", box_weights.value)
    result_weights
    return (result_weights,)


@app.cell(hide_code=True)
def hints_weights():
    L.hints(
        "balanced-weights",
        [
            "Beide klassen moeten in totaal even zwaar wegen, elk n / 2. Elk voorbeeld van een klasse krijgt zijn deel daarvan.",
            "Hoe meer voorbeelden een klasse heeft, hoe minder elk ervan weegt: deel door hoeveel het er zijn. `failures` van de n zijn storingen; hoeveel zijn er normale runs?",
            "`weights[y == 1] = n / (2 * ___)` en `weights[y == 0] = n / (2 * (___))`.",
        ],
        why="Elk van de 229 storingen weegt 15,28 en elk van de 6.771 normale runs 0,517: beide klassen wegen 3.500. Met deze weights getraind vindt logistische regressie 89 van de 110 storingen in de test set, met 630 alarmen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_new():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een nieuwe storing met SMOTE

        `new_failure(X_rare, i, j, u)` krijgt de geschaalde getallen van de storingen, een rij per storing, en moet één nieuwe storing maken: op de lijn van storing `i` naar zijn naaste andere storing nummer `j`, geteld vanaf 0 voor de naaste, op aandeel `u` van de weg. Vervang de twee `...`. De check roept `new_failure(failures, 0, 2, 0.4)` aan met de 229 storingen uit het trainingsdeel, en probeert vijf punten in een vlak, waarbij je kunt zien welk het dichtstbij ligt.
        """),
        L.task("Laat `new_failure` een nieuwe storing op de lijn zetten tussen storing `i` en een van zijn naaste andere storingen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_new():
    box_new = L.code_box("new-failure", lines=8)
    box_new
    return (box_new,)


@app.cell(hide_code=True)
def check_new(box_new):
    result_new = L.check("new-failure", box_new.value)
    result_new
    return (result_new,)


@app.cell(hide_code=True)
def hints_new():
    L.hints(
        "new-failure",
        [
            "Twee stappen: de plekken van de andere storingen van dichtbij naar ver, en dan het punt op aandeel `u` van de weg van `X_rare[i]` naar `b`.",
            "`np.argsort` geeft de plekken van dichtbij naar ver, zoals bij k-NN, maar de naaste van allemaal is storing `i` zelf, op afstand 0: laat die weg. Voor het punt begin je bij `X_rare[i]` en neem je een aandeel `u` van de stap naar `b`.",
            "`nearest = np.argsort(distances)[___:]` en `return X_rare[i] + u * (___)`.",
        ],
        why="De nieuwe storing ligt op de lijn van storing i naar een van zijn naaste andere storingen, dus hij blijft waar storingen zijn. Met i, j en u 6.542 keer willekeurig getrokken zijn dit de SMOTE-storingen uit de tabel: logistische regressie die ermee traint, vindt 89 van de 110 storingen in de test set, met 622 alarmen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_when():
    mo.md(r"""
    ### Oefening 3. Elke storing gevonden
    """)
    return


@app.cell(hide_code=True)
def choice_when_cell():
    choice_when = L.choice(
        "resample-when",
        question="Een collega oversamplet de storingen van alle 10.000 runs, splitst de runs daarna in een deel om op te trainen en een deel om op te testen, en traint k-NN met k = 5. Die vindt elke storing in de test set. Hoe komt dat?",
        options={
            "balanced": "k-NN werkt goed zodra de klassen in balans zijn",
            "right": "Kopieën van elke storing uit de test set zitten tussen de runs om op te trainen",
            "more": "De test set heeft nu veel meer storingen, dus die zijn makkelijker te vinden",
        },
        correct="right",
        why={
            "balanced": "Oversample je pas na de splitsing, dan vindt k-NN 83 van de 110 storingen in de test set, 75,5%: goed, niet perfect. De 100% komt door de volgorde: kopieën van dezelfde storing kwamen aan beide kanten terecht, en de naaste run om op te trainen van elke storing in de test set is zijn eigen kopie, op afstand 0.",
            "more": "Meer storingen maken de precision makkelijker, niet de recall: recall vraagt nog steeds hoeveel van de storingen gevonden zijn. Alle 2.913 storingen in die test set werden gevonden omdat elk ervan een kopie had tussen de runs om op te trainen, op afstand 0.",
            "right": "Elk van de 2.913 storingen in die test set had een kopie tussen de runs om op te trainen, en k-NN vond die op afstand 0. Oversample je na de splitsing, alleen in het trainingsdeel, dan vindt k-NN 83 van de 110 storingen in de test set, 75,5%.",
        },    )
    choice_when
    return (choice_when,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Precision en recall komen uit [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl), en [Precision en recall in één getal: de F1 score](/course/open/ml-f1-score?lang=nl) liet al zien hoe accuracy een model dat niets vindt bovenaan kan zetten. De threshold en de precision-recall curve komen uit [Alle thresholds tegelijk](/course/open/ml-roc-auc?lang=nl): class weights schuiven een model vooral langs die curve. SMOTE gebruikt de naaste buren uit [Vraag het de buren](/course/open/ml-knn?lang=nl). Oefening 3 is een eerste blik op een test set die te veel weet, wat een latere les helemaal behandelt. In de volgende les, *Instellingen zoeken: grid search en random search*, kies je de instellingen van een model op deze runs beter dan door ze één voor één te proberen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_when, result_new, result_weights):
    L.wrap_up(result_weights, result_new, choice_when)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
