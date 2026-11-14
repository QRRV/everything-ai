# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-automl. Route: course/route/lessen/klassiek-ml.yaml. Checks en plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-automl", "nl")
    C = L.checks   # de 27 kandidaten, hun scores op 778, 2.333 en 7.000 runs, één keer gemeten, en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Instellingen zoeken: grid search en random search](/course/open/ml-grid-random-search?lang=nl) en één over [Veel bomen stemmen: het random forest](/course/open/ml-random-forest?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_grid():
    quick_grid = L.choice(
        "quick-grid",
        question="Een grid search probeert 3 waarden van elk van 4 instellingen. Hoeveel pogingen zijn dat?",
        options={
            "sum": "12",
            "right": "81",
            "square": "16",
        },
        correct="right",
        why={
            "sum": "12 is 3 + 3 + 3 + 3. Een grid probeert elke combinatie: elke waarde van de eerste instelling met elke waarde van de tweede, enzovoort, dus de aantallen worden vermenigvuldigd. Zie *Het idee* in [Instellingen zoeken: grid search en random search](/course/open/ml-grid-random-search?lang=nl).",
            "square": "16 is 4 · 4. Een grid probeert elke combinatie van de 3 waarden van elke instelling: $3 \\cdot 3 \\cdot 3 \\cdot 3$. Zie *Het idee* in [Instellingen zoeken: grid search en random search](/course/open/ml-grid-random-search?lang=nl).",
            "right": "$3 \\cdot 3 \\cdot 3 \\cdot 3 = 81$, en elke poging is een hele cross-validation. Vandaag besteedt een zoektocht minder aan de pogingen die duidelijk slecht zijn.",
        },
    )
    quick_grid
    return (quick_grid,)


@app.cell(hide_code=True)
def quick_check_forest():
    quick_forest = L.choice(
        "quick-forest",
        question="`RandomForestClassifier(n_estimators=50, max_features=3)` op data met 5 features. Wat klopt?",
        options={
            "same": "Alle 50 bomen leren van dezelfde voorbeelden",
            "all": "Elke splitsing kijkt naar alle 5 features en neemt de beste",
            "right": "Elke boom leert van zijn eigen bootstrap-steekproef, en elke splitsing kijkt naar 3 willekeurig getrokken features",
        },
        correct="right",
        why={
            "same": "Elke boom leert van zijn eigen bootstrap-steekproef, n voorbeelden met teruglegging getrokken, dus de bomen verschillen. Zie *Het idee* in [Veel bomen stemmen: het random forest](/course/open/ml-random-forest?lang=nl).",
            "all": "Met `max_features=3` kijkt elke splitsing naar 3 willekeurig getrokken features, dus de bomen verschillen. Zie de nieuwe woorden van [Veel bomen stemmen: het random forest](/course/open/ml-random-forest?lang=nl).",
            "right": "Klopt. Vandaag zijn het aantal features per splitsing en de diepte van de bomen twee instellingen die een zoektocht kiest, naast k en p van k-NN.",
        },
    )
    quick_forest
    return (quick_forest,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In de vorige les zocht je de instellingen van één model, k-NN. Maar vaak weet je ook niet welk model het best past: k-NN, een random forest of iets anders. Laat een zoektocht ook het model kiezen, en het aantal kandidaten loopt snel op. 27 kandidaten een score geven met cross-validation op alle 7.000 runs om op te trainen kost 27 hele cross-validations, en de meeste kandidaten blijken na een korte blik duidelijk slechter dan de beste. Kan een zoektocht daar minder aan besteden?

    **Een voorbeeld uit het dagelijks leven.** Een bedrijf heeft 27 sollicitanten voor één baan. Een proefdag met elk van hen zou 27 dagen kosten. Dus belt het alle 27 een paar minuten, nodigt het de beste 9 uit voor een gesprek, en krijgen de beste 3 een proefdag. De lange proeven gaan naar de paar die de meeste kans maken op de baan. Het risico: iemand die op een proefdag zou schitteren maar aan de telefoon zenuwachtig is, krijgt er nooit een.
    """)
    return


@app.cell(hide_code=True)
def predict_budget_cell():
    guess_budget = L.guess(
        "budget-guess",
        question="27 kandidaten. Ronde 1 geeft elk 778 runs, ronde 2 geeft de beste 9 elk 2.333, en ronde 3 geeft de beste 3 alle 7.000. Alle 27 op alle 7.000 proberen deelt 27 · 7.000 = 189.000 runs uit. Hoeveel delen de drie rondes uit?",
        options={
            "half": "Ongeveer de helft: 95.000",
            "right": "Ongeveer een derde: 63.000",
            "tenth": "Ongeveer een tiende: 19.000",
        },
        reveal="**Ongeveer een derde.** Ronde 1 deelt 27 · 778 = 21.006 runs uit, ronde 2 9 · 2.333 = 20.997 en ronde 3 3 · 7.000 = 21.000: elke ronde kost ongeveer evenveel, samen 63.003.",
    )
    guess_budget
    return (guess_budget,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **AutoML.** Een model bouwen vraagt keuzes: hoe je de data voorbewerkt, welk model je neemt, en de hyperparameters ervan. **AutoML** laat een tool die keuzes maken: die zoekt zelf modellen en hun instellingen. Hier zijn de kandidaten 27 willekeurig getrokken, zoals bij een random search: een muntworp kiest tussen k-NN, met k en p zoals in [Instellingen zoeken: grid search en random search](/course/open/ml-grid-random-search?lang=nl), en een random forest van 50 bomen zoals in [Veel bomen stemmen: het random forest](/course/open/ml-random-forest?lang=nl), met een maximale diepte zoals in [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl) van 2 tot 30, getrokken op een logaritmische schaal, en 1 tot 5 features per splitsing. De trekking gaf 11 k-NN-modellen en 16 forests, waarvan twee dezelfde: een willekeurige trekking kan zichzelf herhalen.

        **Een budget.** In het dagelijks leven is een budget geld. Hier is het wat een zoektocht mag uitgeven: een aantal pogingen, een hoeveelheid tijd, of, zoals in deze les, uitgedeelde runs om op te trainen. Een run blijft één run van een machine, een rij van AI4I 2020: een kandidaat die een score krijgt op 778 runs kost 778; op alle 7.000 runs 7.000. Alle 27 een score geven op alle 7.000 kost $27 \cdot 7.000 = 189.000$.

        **Successive halving.** Kevin Jamieson en Ameet Talwalkar (2016) gebruikten het om instellingen te zoeken: geef elke kandidaat eerst een klein deel van het budget, houd het beste deel, geef die meer, en herhaal tot er één kandidaat over is. Hier houdt elke ronde het beste derde deel en geeft die 3 keer zoveel runs: 27 kandidaten op de eerste 778 runs om op te trainen, de beste 9 op 2.333, de beste 3 op alle 7.000. Elke score is de gemiddelde AUC van 5 folds, zoals in de vorige les. Elke ronde kost ongeveer 21.000 runs, drie rondes ongeveer 63.000: een derde van 189.000. De naam komt van de versie die de beste helft houdt.

        **Het risico.** Een paar runs geven een ruwe score. Onder de eerste 778 runs om op te trainen zitten maar 29 storingen, dus elke fold, ongeveer 155 runs, bevat er 3 tot 9, en de AUC van die fold rust op die paar. Een kandidaat die veel voorbeelden nodig heeft om te laten zien wat die kan, zoals een forest waarvan elke splitsing naar 4 of 5 features kijkt, kan er in ronde 1 slecht uitzien en afvallen voordat die meer krijgt.

        **De tools.** Optuna (Takuya Akiba en anderen, 2019) kiest elke nieuwe poging dicht bij eerdere goede pogingen en kan een poging vroeg afbreken als de eerste scores slecht zijn. auto-sklearn (Matthias Feurer en anderen, 2015) kiest elke nieuwe poging ook uit de scores tot nu toe, begint met instellingen die op vergelijkbare datasets werkten, en combineert zijn beste modellen tot een ensemble. AutoGluon (Nick Erickson en anderen, 2020) traint veel soorten modellen met goede standaardinstellingen en combineert ze. scikit-learn heeft successive halving zelf, nog als experimenteel gemarkeerd: `HalvingRandomSearchCV`.

        **Wat van jou blijft.** Elke tool heeft een trainingsdeel nodig en een test set die de tool nooit ziet, een metric die bij het probleem past, zoals de AUC voor zeldzame storingen, en een budget. Een tool kiest alleen goed met de maat die jij geeft.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **AutoML** | een tool het model, de instellingen ervan en de voorbewerking van de data laten kiezen, door te zoeken binnen een budget | |
        | **budget** | wat een zoektocht mag uitgeven: pogingen, tijd of uitgedeelde runs om op te trainen; geen geld, zoals in het dagelijks leven | |
        | **successive halving** | geef elke kandidaat eerst een klein budget, houd het beste deel, geef die meer, en herhaal tot er één over is | `HalvingRandomSearchCV` |
        | **kandidaat** | één model met zijn instellingen dat een zoektocht probeert | `("forest", 5, 3)` |
        """),
        L.watch_out("Een zoektocht met een kleiner budget is niet gratis: de eerste ronde moet groot genoeg zijn om goed van slecht te onderscheiden. Te weinig runs, en de scores zijn vooral toeval."),
        mo.md(r"""
        De slider gaat door de rondes. Links de AUC van elke kandidaat in de rondes tot nu toe, tegen het aantal runs om op te trainen dat elke kandidaat kreeg, op een logaritmische schaal: een dichte stip gaat door, een open stip valt af, en een grijze lijn volgt een kandidaat naar de volgende ronde. Bij ronde 3 zijn de kleine grijze stippen de scores van alle 27 op alle 7.000 runs, met de beste ervan omcirkeld. Rechts het budget tot nu toe, in duizenden runs, tegen alle 27 op alle 7.000. Beweeg de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_round_cell():
    def _frame(r):
        said = C.round_said(r, L.lang)
        alt = ("Links: de AUC van de kandidaten van successive halving, tegen de runs om op te trainen die elk kreeg, op een "
               "logaritmische schaal van 778, 2.333 en 7.000 runs. Rechts: twee balken, de runs die tot nu toe zijn "
               "uitgedeeld en de 189.000 van alle 27 kandidaten op alle 7.000 runs. " + said)
        labels = {"rounds": "AUC van elke kandidaat", "runs": "runs om op te trainen", "budget": "Budget",
                  "thousands": "runs (duizenden)", "halving": "halving", "full": "alle 27", "alt": alt}
        return (C.round_picture(r, labels, L.lang), said)

    explore_round = L.explore("round", "ronde", _frame, start=1, values=[1, 2, 3], role="",
                              task="Ga door de drie rondes. Welke kandidaten gaan door na ronde 1, en welke kandidaat zou winnen op alle 7.000 runs?")
    explore_round
    return


@app.cell(hide_code=True)
def text_after_round():
    mo.md(r"""
    Successive halving gaf 63.003 runs uit, een derde van 189.000. In tijd scheelde het minder: op de computer van de course duurde het ongeveer 14 seconden, tegen 27 voor alle 27 op alle 7.000, omdat een fit op een paar runs niet zoveel sneller is. De winnaar, een random forest met diepte 5 en 3 features per splitsing, heeft AUC 0,961 in cross-validation. Alle 27 op alle 7.000 runs vinden een dieper forest, met diepte 12 en 4 features per splitsing, en 0,969. Op 778 runs scoorde dat forest 0,898, plek 21 van 27, en successive halving liet het in ronde 1 vallen. Eén keer gemeten op de 3.000 test-runs zijn de twee winnaars ongeveer gelijk: 0,973 tegen 0,974.

    ### Recept

    Modellen en hun instellingen zoeken met successive halving:

    1. Zet een test set apart en kies een metric die bij het probleem past.
    2. Kies de kandidaten: modellen en een bereik voor elke instelling, willekeurig getrokken zoals bij een random search.
    3. Kies het budget, de factor en de rondes: met factor 3 en 3 rondes houdt ronde r (geteld vanaf 0) $27 / 3^r$ kandidaten en geeft elk $7.000 / 3^{2-r}$ runs.
    4. Geef elke kandidaat van een ronde een score met cross-validation op de runs van die ronde, houd het beste derde deel, en ga naar de volgende ronde.
    5. Controleer of de eerste ronde groot genoeg is: zijn de scores vooral toeval, begin dan met minder kandidaten en meer runs.
    6. Fit de winnaar op het hele trainingsdeel en meet die één keer op de test set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel leest en splitst de runs zoals in [Als één klasse zeldzaam is](/course/open/ml-imbalanced?lang=nl), en trekt de 27 kandidaten. `_model(c)` maakt het model van kandidaat `c`, en `_score(c, runs)` geeft de gemiddelde AUC van 5 folds op de eerste `runs` runs om op te trainen. Dan komen de drie rondes, met $7.000 / 3^{2-r}$ runs in ronde r, afgerond: 778, 2.333 en 7.000. In elke ronde: geef elke kandidaat een score, houd het beste derde deel met `np.argsort`, zoals in [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl), en print de runs, het aantal kandidaten en de beste van de ronde met de AUC ervan. Op de computer van de course duurt de cel ongeveer 15 seconden. Voer de cel uit.
    """)
    return


@app.cell
def example_halving():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    _df = pd.read_csv("data/ai4i-2020.csv.gz")
    _numbers = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    _X = np.array(_df[_numbers])
    _y = np.array(_df["Machine failure"])
    _order = np.random.default_rng(0).permutation(len(_y))
    _train = _order[3000:]                                 # de 7.000 runs om op te trainen; de 3.000 test-runs wachten

    def _model(c):
        if c[0] == "neighbors":
            return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=c[1], p=c[2]))
        return RandomForestClassifier(n_estimators=50, max_depth=c[1], max_features=c[2], random_state=0)

    def _score(c, runs):
        _part = _train[:runs]                              # de eerste `runs` runs om op te trainen
        return cross_val_score(_model(c), _X[_part], _y[_part], cv=KFold(5), scoring="roc_auc").mean()

    _rng = np.random.default_rng(0)
    _candidates = []
    for _ in range(27):
        if _rng.random() < 0.5:                            # een munt: k-NN of een random forest
            _candidates.append(("neighbors", int(10 ** _rng.uniform(0, np.log10(300))), round(_rng.uniform(1, 4), 2)))
        else:
            _candidates.append(("forest", int(10 ** _rng.uniform(np.log10(2), np.log10(30))), int(_rng.integers(1, 6))))

    _rounds = []
    for _r in range(3):
        _rounds.append(round(7000 / 3 ** (2 - _r)))        # 778, 2.333 en 7.000 runs
    for _runs in _rounds:
        _scores = []
        for _c in _candidates:
            _scores.append(_score(_c, _runs))
        _i = np.argmax(_scores)
        print(_runs, len(_candidates), _candidates[_i], round(_scores[_i], 3))
        _keep = round(len(_candidates) / 3)
        _kept = []
        for _j in np.argsort(_scores)[-_keep:]:             # de posities van het beste derde deel
            _kept.append(_candidates[_j])
        _candidates = _kept
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Ronde 1 geeft alle 27 een score op 778 runs; de beste is een forest met diepte 25 en 2 features per splitsing, met 0,965. Ronde 2 geeft de beste 9 een score op 2.333 runs; de beste is een forest met diepte 5 en 2 features, met 0,955. Ronde 3 geeft de beste 3 een score op alle 7.000 runs, en het forest met diepte 5 en 3 features wint, met 0,961. De beste van de ene ronde is niet altijd de beste van de volgende: met meer runs verandert de volgorde.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_plan():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Het plan van de rondes

        `plan(n, total, factor, rounds)` moet voor elke ronde geven hoeveel kandidaten er over zijn en hoeveel runs elk krijgt, als een lijst paren `(candidates, runs)`. Ronde `r` telt vanaf 0: die houdt `n` gedeeld door `factor`, één keer voor elke ronde ervoor, en de laatste ronde geeft elke kandidaat alle `total` runs, elke eerdere ronde `factor` keer minder dan de volgende. Vervang de twee `...`; rond allebei af op een geheel getal met `round`. De check roept `plan(27, 7000, 3, 3)` aan, dat `[(27, 778), (9, 2333), (3, 7000)]` moet geven, en nog drie plannen.
        """),
        L.task("Laat `plan` de kandidaten en de runs van elke ronde geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_plan():
    box_plan = L.code_box("plan", lines=7)
    box_plan
    return (box_plan,)


@app.cell(hide_code=True)
def check_plan(box_plan):
    result_plan = L.check("plan", box_plan.value)
    result_plan
    return (result_plan,)


@app.cell(hide_code=True)
def hints_plan():
    L.hints(
        "plan",
        [
            "Schrijf het plan van de les uit: hoe vaak is 27 in ronde 0, 1 en 2 door 3 gedeeld, en hoe vaak wordt 7.000 door 3 gedeeld?",
            "De kandidaten: `n` gedeeld door `factor`, één keer voor elke ronde vóór ronde `r`. Hoeveel is dat, en hoe schrijf je 'zo vaak door `factor`' met `**`? De runs: `total` gedeeld door `factor`, één keer voor elke ronde die na ronde `r` nog komt; in het plan van de les komen er na ronde 0 nog twee.",
            "`candidates = round(n / factor ** ___)` en `runs = round(total / factor ** (___ - 1 - ___))`.",
        ],
        why="Ronde r houdt $27 / 3^r$ kandidaten: 27, 9, 3. Elk krijgt $7.000 / 3^{2-r}$ runs: 778, 2.333, 7.000. Kandidaten keer runs is in elke ronde ongeveer 21.000, dus de rondes kosten evenveel, en drie ervan een derde van de volledige zoektocht.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_halving():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Successive halving

        `successive_halving(candidates, score, rounds)` krijgt een lijst kandidaten, een functie `score(c, runs)` die de score van kandidaat `c` op `runs` runs geeft, en de runs van elke ronde. Een functie is ook een waarde: `score` zonder haakjes is de functie zelf, dus die kun je aan de jouwe meegeven, en daarbinnen roept `score(c, runs)` die aan. In elke ronde moet die elke kandidaat een score geven op de runs van die ronde, de beste `keep` houden en daarmee verdergaan; aan het eind geeft die terug wat er over is. Vervang de twee `...`. De check roept de functie aan met de 27 kandidaten van de les, `score` uit de gemeten scores en `rounds = [778, 2333, 7000]`, en met nog drie lijsten, zoals 81 getallen in vier rondes.
        """),
        L.task("Laat `successive_halving` in elke ronde het beste derde deel houden."),
    ])
    return


@app.cell(hide_code=True)
def exercise_halving():
    box_halving = L.code_box("halving", lines=14)
    box_halving
    return (box_halving,)


@app.cell(hide_code=True)
def check_halving(box_halving):
    result_halving = L.check("halving", box_halving.value)
    result_halving
    return (result_halving,)


@app.cell(hide_code=True)
def hints_halving():
    L.hints(
        "halving",
        [
            "Twee dingen: de score van `c` in deze ronde, en welke posities bij de beste `keep` scores horen. De functie die een score geeft, is de `score` die je functie krijgt.",
            "Roep `score` aan met de kandidaat en de runs van deze ronde. `np.argsort(scores)` geeft de posities van de laagste score naar de hoogste; de beste `keep` staan aan het eind, en een slice met een negatief getal, zoals `scores[-2:]`, neemt het eind van een lijst.",
            "`scores.append(score(___, ___))` en `best = np.argsort(scores)[-___:]`.",
        ],
        why="`np.argsort(scores)[-keep:]` geeft de posities van de `keep` hoogste scores. Elke ronde houdt een derde, dus 27 kandidaten worden 9, dan 3, dan 1, en de latere rondes geven de paar die over zijn de meeste runs.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_small():
    mo.md(r"""
    ### Oefening 3. 243 kandidaten
    """)
    return


@app.cell(hide_code=True)
def choice_small_cell():
    choice_small = L.choice(
        "small-first",
        question="Je probeert 243 kandidaten met hetzelfde plan: houd elke ronde een derde en eindig met alle 7.000 runs. Dat kost 5 rondes, dus ronde 1 geeft elke kandidaat 86 runs, met 3 storingen ertussen. Wat gaat er mis?",
        options={
            "budget": "De zoektocht kost meer dan alle 243 op alle 7.000 runs proberen",
            "more": "Niets: meer kandidaten vinden altijd een betere winnaar",
            "right": "Ronde 1 meet vooral toeval, dus goede kandidaten kunnen meteen afvallen",
        },
        correct="right",
        why={
            "budget": "Elk van de 5 rondes deelt ongeveer 21.000 runs uit, samen ongeveer 105.000, tegen $243 \\cdot 7.000 = 1.701.000$. Het probleem zit in ronde 1: van de 5 folds bevatten er 3 helemaal geen storing, dus daarin valt geen AUC te meten, en de andere rusten op 1 of 2 storingen.",
            "more": "Meer kandidaten helpen alleen als ronde 1 goed van slecht kan onderscheiden. Met 3 storingen onder 86 runs bevatten 3 van de 5 folds helemaal geen storing: daarin valt geen AUC te meten, en de andere rusten op 1 of 2 storingen.",
            "right": "Klopt: van de 5 folds van ronde 1 bevatten er 3 helemaal geen storing, en de andere rusten op 1 of 2. Zelfs de 778 runs van deze les lieten het forest vallen dat op alle 7.000 het best is. Hyperband (Lisha Li en anderen, 2018) doet daarom successive halving meerdere keren, elke keer met minder kandidaten en meer runs om mee te beginnen.",
        },
    )
    choice_small
    return (choice_small,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De kandidaten zijn getrokken zoals bij de random search van [Instellingen zoeken: grid search en random search](/course/open/ml-grid-random-search?lang=nl), en elke score is een cross-validation zoals in [Betrouwbaarder kiezen: cross-validation](/course/open/ml-cross-validation?lang=nl), met de AUC uit [Alle thresholds tegelijk: de ROC curve en AUC](/course/open/ml-roc-auc?lang=nl) omdat storingen zeldzaam zijn, zoals in [Als één klasse zeldzaam is](/course/open/ml-imbalanced?lang=nl). De helft van de kandidaten zijn de random forests uit [Veel bomen stemmen: het random forest](/course/open/ml-random-forest?lang=nl), en hun maximale diepte is die uit [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl). Successive halving beslist met de maat die jij geeft, dus een eerlijke verdeling van de data en een passende metric blijven jouw werk, welke tool er ook zoekt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_small, result_halving, result_plan):
    L.wrap_up(result_plan, result_halving, choice_small)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
