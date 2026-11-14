# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-grid-random-search. Route: course/route/lessen/klassiek-ml.yaml. Checks en plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-grid-random-search", "nl")
    C = L.checks   # de pogingen van het grid en van de random search op de runs van de machine, één keer gemeten, en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Parameters en hyperparameters](/course/open/ml-hyperparameters?lang=nl) en één over [Elke stap in elke fold: pipelines](/course/open/ml-pipelines?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_hyper():
    quick_hyper = L.choice(
        "quick-hyper",
        question="k-NN met `n_neighbors=15`. Wie bepaalt die 15?",
        options={
            "fit": "`fit` leert het uit de voorbeelden om op te trainen",
            "right": "Jij, vóór de training: het is een hyperparameter",
            "data": "Het volgt uit het aantal voorbeelden om op te trainen",
        },
        correct="right",
        why={
            "fit": "`fit` van k-NN bewaart de voorbeelden om op te trainen en leert geen k: het gebruikt de k die jij gaf. Een getal dat je vóór de training kiest, is een hyperparameter. Zie *Het idee* in [Parameters en hyperparameters](/course/open/ml-hyperparameters?lang=nl).",
            "data": "Niets in de data legt k vast. Welke k goed is, hangt van de data af, maar jij kiest hem, vóór de training. Zie *Het idee* in [Parameters en hyperparameters](/course/open/ml-hyperparameters?lang=nl).",
            "right": "Jij kiest die, en de loss op de training set kan die niet voor je kiezen. Vandaag zoek je tegelijk naar een goede k en een tweede instelling.",
        },
    )
    quick_hyper
    return (quick_hyper,)


@app.cell(hide_code=True)
def quick_check_folds():
    quick_folds = L.choice(
        "quick-folds",
        question="`scores = cross_val_score(pipe, X, y, cv=KFold(5))`. Welk getal vergelijk je tussen twee instellingen?",
        options={
            "max": "`scores.max()`, de beste fold",
            "first": "`scores[0]`, de eerste fold",
            "right": "`scores.mean()`, het gemiddelde van de 5 folds",
        },
        correct="right",
        why={
            "max": "De beste fold is de gelukkigste van de vijf. Het gemiddelde gebruikt elk voorbeeld één keer voor de validatie. Zie het recept van [Elke stap in elke fold: pipelines](/course/open/ml-pipelines?lang=nl).",
            "first": "Eén fold is weer één validation set, met al zijn toeval. Het gemiddelde van de 5 is wat cross-validation je oplevert. Zie het recept van [Elke stap in elke fold: pipelines](/course/open/ml-pipelines?lang=nl).",
            "right": "Het gemiddelde van de 5 folds. Vandaag reken je het uit voor veel instellingen en houd je de beste.",
        },
    )
    quick_folds
    return (quick_folds,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De meeste modellen hebben meer dan één instelling die je zelf kiest. k-NN heeft zijn k en de manier waarop het een afstand meet; een random forest heeft het aantal bomen, hoe diep ze groeien en naar hoeveel features elke splitsing mag kijken. Eén instelling tegelijk proberen laat niet zien hoe ze samenwerken, en elke combinatie van veel waarden proberen duurt al snel te lang: 5 waarden van 4 instellingen zijn 625 combinaties, en elke combinatie is een hele cross-validation. Welke combinaties probeer je als je maar een beperkt aantal pogingen hebt?

    **Een voorbeeld uit het dagelijks leven.** Je test een nieuw taartrecept en kunt 9 taarten bakken. De temperatuur van de oven doet er veel toe, het rooster in de oven nauwelijks. Bak je 3 temperaturen op elk van 3 roosters, dan heb je maar 3 temperaturen geprobeerd. Bak je 9 taarten op 9 verschillende temperaturen, elk op een willekeurig rooster, dan heb je er 9 geprobeerd.
    """)
    return


@app.cell(hide_code=True)
def predict_values_cell():
    guess_values = L.guess(
        "values-guess",
        question="Een zoektocht probeert 3 waarden van k met elk van 3 waarden van p: 9 pogingen. Hoeveel verschillende waarden van k test die?",
        options={
            "nine": "9",
            "six": "6",
            "right": "3",
        },
        reveal="**3.** Elke waarde van k wordt 3 keer geprobeerd, één keer met elke p. Als p er nauwelijks toe doet, zoals bij de machines in deze les, vertellen 6 van de 9 pogingen je bijna niets nieuws over k. 9 willekeurig getrokken pogingen testen tot 9 waarden van k.",
    )
    guess_values
    return (guess_values,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Twee instellingen.** k-NN uit [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl) heeft hier twee instellingen: k, het aantal buren, en p, hoe het een afstand meet. Met p = 2 is het de afstand in een rechte lijn uit [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl): de wortel uit de som van de gekwadrateerde verschillen. Met p = 1 telt het de groottes van de verschillen op zonder ze te kwadrateren. Waarden tussen 1 en 2 liggen ertussenin, en boven 2 telt het grootste verschil steeds zwaarder. De **search space** is elke combinatie die je toestaat: hier k van 1 tot 300 en p van 1 tot 4. Elke poging is een cross-validation zoals in [Elke stap in elke fold: pipelines](/course/open/ml-pipelines?lang=nl), op de 7.000 runs om op te trainen uit [Als één klasse zeldzaam is](/course/open/ml-imbalanced?lang=nl), en de score is de gemiddelde AUC uit [Alle thresholds tegelijk](/course/open/ml-roc-auc?lang=nl) over de 5 folds, omdat storingen zeldzaam zijn.

        **Grid search.** Schrijf een paar waarden van elke instelling op en probeer elke combinatie: 3 waarden van k en 3 van p geven $3 \cdot 3 = 9$ pogingen. Op een vlak met k naar rechts en p omhoog vormen de pogingen een **grid** van rijen en kolommen.

        **Random search.** Trek elke poging willekeurig uit de search space, k en p elke keer opnieuw. Met 9 pogingen test je tot 9 waarden van k (k is een geheel getal, dus een waarde kan twee keer komen) en 9 van p.

        **Een logaritmische schaal voor k.** Van k = 1 naar 2 verandert het model veel, van 201 naar 202 nauwelijks. Trek k daarom gelijk verdeeld op een logaritmische schaal, zoals in [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl): trek de logaritme gelijk verdeeld tussen $\log_{10} 1 = 0$ en $\log_{10} 300 = 2{,}48$, en verhef 10 tot die macht. Dan krijgen 1 tot 10 en 10 tot 100 ongeveer evenveel pogingen; het grid verdeelt zijn waarden van k op dezelfde manier. `rng.uniform(low, high, n)` trekt n getallen gelijk verdeeld tussen `low` en `high`, zoals `rng.random` uit [Willekeurige getallen en seeds](/course/open/np-random?lang=nl) dat doet tussen 0 en 1. Je schrijft het trekken in oefening 2.

        **Eén instelling doet ertoe.** James Bergstra en Yoshua Bengio (2012) zagen dat meestal maar een paar van de instellingen er veel toe doen. Hier is dat k: de AUC stijgt van 0,716 bij k = 1 tot 0,952 rond k = 60, en daalt daarna langzaam. p doet er nauwelijks toe: bij elke k van de grids verschillen de waarden van p hooguit 0,01 in AUC, en k laat de AUC gaan van 0,716 tot 0,952. Een grid met 3 waarden per instelling test maar 3 waarden van k, hoeveel waarden van p het er ook bij neemt.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **search space** | elke combinatie van instellingen die een zoektocht mag proberen: een bereik of een lijst waarden voor elke instelling | |
        | **grid search** | elke combinatie proberen van een paar waarden die je voor elke instelling kiest | twee loops, de ene in de andere |
        | **random search** | combinaties proberen die willekeurig uit de search space getrokken zijn | `rng.uniform(low, high, n)` |
        | `p` van k-NN | hoe k-NN een afstand meet: 2 is de rechte lijn, 1 de som van de groottes van de verschillen, en boven 2 telt het grootste verschil steeds zwaarder | `KNeighborsClassifier(n_neighbors=k, p=p)` |
        | `scoring="roc_auc"` | laat `cross_val_score` de AUC van elke fold geven in plaats van het deel dat het goed heeft | `cross_val_score(pipe, X, y, cv=KFold(5), scoring="roc_auc")` |
        """),
        L.watch_out("Zoek alleen op het trainingsdeel. De test set meet de winnaar één keer, aan het eind; helpt die je kiezen, dan is zijn score geen eerlijke schatting meer voor nieuwe runs."),
        mo.md(r"""
        De slider zet het aantal pogingen, voor beide zoektochten hetzelfde. Links elke poging op het vlak van k en p: open stippen zijn het grid, dichte stippen de random search, en een ring markeert de beste poging van elk. Rechts de AUC van elke poging tegen zijn k; de grijze lijn is de AUC met p = 2 voor elke k. k staat op een logaritmische schaal. Beweeg de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_budget_cell():
    def _frame(n):
        said = C.budget_said(n, L.lang)
        alt = ("Links: de pogingen van een grid, in rijen en kolommen, en van een random search, ongelijk verspreid, op "
               "een vlak met k op een logaritmische schaal naar rechts en p omhoog. Rechts: de AUC van elke poging tegen "
               "zijn k, naast een lijn die stijgt van 0,72 bij k = 1 tot 0,95 rond k = 60 en daarna langzaam daalt. " + said)
        return (C.budget_picture(n, {"plane": "Pogingen", "score": "AUC van elke poging", "p": "p (macht)", "alt": alt}), said)

    explore_budget = L.explore("tries", "aantal pogingen", _frame, start=9, values=C.BUDGETS, role="hyper",
                               task="Ga van 4 naar 25 pogingen. Welke zoektocht vindt de hoogste AUC, en hoeveel waarden van k test elk?")
    explore_budget
    return


@app.cell(hide_code=True)
def text_after_budget():
    mo.md(r"""
    Met 4 pogingen wint het grid: zijn k = 72 ligt toevallig dicht bij de top van de grijze lijn. Vanaf 9 pogingen vindt de random search 0,952 bij k = 64, en komt het grid niet verder dan 0,947: zijn waarden van k liggen naast de top, zoals 116, 147 en 54. Meer pogingen in een grid helpen niet altijd: 16 pogingen vinden minder dan 9. De random search test 4, 9, 15 en 23 verschillende waarden van k. Eén trekking kan geluk hebben: de course deed 20 random searches van 9 pogingen, elk met een eigen seed, en hun beste AUC lag tussen 0,939 en 0,952; 12 van de 20 versloegen het grid van 9.

    De winst is hier klein. Eén keer gemeten op de 3.000 test-runs heeft de winnaar van de random search van 9 pogingen een AUC van 0,960, en die van het grid 0,962: een verschil van een paar duizendsten in cross-validation kan toeval zijn. Random search loont bij meer instellingen: een grid van 3 waarden van 4 instellingen kost 81 pogingen en test nog steeds maar 3 waarden van elk; 81 willekeurige pogingen trekken 81 waarden van elk.

    ### Recept

    Om de instellingen van een model te zoeken:

    1. Kies de instellingen en voor elk een bereik. Gebruik een logaritmische schaal voor een instelling die met factoren werkt, zoals k, een learning rate of de sterkte van regularisatie.
    2. Kies een budget: hoeveel pogingen je je kunt veroorloven, elk een hele cross-validation.
    3. Trek de pogingen willekeurig uit de bereiken. Een grid is prima voor één of twee instellingen met een paar waarden.
    4. Geef elke poging een score met het gemiddelde over dezelfde folds, met een metric die bij het probleem past.
    5. Ligt de beste poging aan de rand van een bereik, maak dat bereik dan groter en zoek opnieuw.
    6. Fit de winnaar op het hele trainingsdeel en meet die één keer op de test set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel leest de runs en splitst ze zoals in [Als één klasse zeldzaam is](/course/open/ml-imbalanced?lang=nl). `_score(k, p)` geeft de gemiddelde AUC van 5 folds voor k-NN met die instellingen, in een pipeline met schalen zoals in [Elke stap in elke fold: pipelines](/course/open/ml-pipelines?lang=nl). De cel maakt het grid van 9 pogingen met twee loops, trekt 9 pogingen willekeurig, geeft alle 18 een score en print de beste van elke zoektocht met zijn AUC. 18 cross-validations duren ongeveer tien seconden. Voer de cel uit.
    """)
    return


@app.cell
def example_search():
    import numpy as np
    import pandas as pd
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

    def _score(k, p):
        _pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, p=p))
        return cross_val_score(_pipe, _X[_train], _y[_train], cv=KFold(5), scoring="roc_auc").mean()

    _grid = []
    for _k in [3, 17, 116]:
        for _p in [1.5, 2.5, 3.5]:
            _grid.append((_k, _p))

    _rng = np.random.default_rng(0)
    _random = []
    for _ in range(9):
        _k = int(10 ** _rng.uniform(0, np.log10(300)))     # k gelijk verdeeld op een logaritmische schaal, van 1 tot 300
        _p = round(_rng.uniform(1, 4), 2)
        _random.append((_k, _p))

    for _name, _tries in [("grid", _grid), ("random", _random)]:
        _scores = []
        for _k, _p in _tries:
            _scores.append(_score(_k, _p))
        _i = np.argmax(_scores)                            # de plek van de beste poging
        print(_name, _tries[_i], round(_scores[_i], 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het grid vindt AUC 0,947 bij k = 116 en p = 1,5, de random search 0,952 bij k = 64 en p = 1,53. Het grid testte k = 3, 17 en 116; de random search 9 waarden van k, van 1 tot 137.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_grid():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Elke combinatie

        `grid(ks, ps)` krijgt een lijst waarden van k en een lijst waarden van p, en moet elke combinatie geven als een lijst paren `(k, p)`, zoals in het uitgewerkte voorbeeld. Vervang de `...`. De check roept `grid(ks, ps)` aan met `ks = [3, 17, 116]` en `ps = [1.5, 2.5, 3.5]`, en probeert nog drie paren lijsten; de volgorde van de paren maakt niet uit.
        """),
        L.task("Laat `grid` elke combinatie geven van een waarde van k en een waarde van p."),
    ])
    return


@app.cell(hide_code=True)
def exercise_grid():
    box_grid = L.code_box("grid", lines=7)
    box_grid
    return (box_grid,)


@app.cell(hide_code=True)
def check_grid(box_grid):
    result_grid = L.check("grid", box_grid.value)
    result_grid
    return (result_grid,)


@app.cell(hide_code=True)
def hints_grid():
    L.hints(
        "grid",
        [
            "Elke waarde van `ks` moet elke waarde van `ps` tegenkomen. Wat heb je nodig binnen een loop over `ks`?",
            "Een tweede loop, over `ps`, binnen de eerste. Voeg daarin het paar van de twee huidige waarden toe aan `tries`.",
            "`for k in ks:`, dan een stap verder ingesprongen `for p in ps:`, dan nog een stap verder `tries.append((___, ___))`.",
        ],
        why="Twee loops, de ene in de andere, maken elke combinatie: 3 waarden van k maal 3 van p zijn 9 pogingen. Elke waarde van k komt 3 keer terug; met 4 instellingen van elk 3 waarden zouden het 81 pogingen zijn.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_log():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Gelijk verdeeld op een logaritmische schaal

        `log_uniform(rng, low, high, n)` moet n getallen trekken tussen `low` en `high`, gelijk verdeeld op een logaritmische schaal: trek hun logaritmen gelijk verdeeld tussen `np.log10(low)` en `np.log10(high)` met `rng.uniform`, en maak er weer getallen van. Vervang de drie `...`. De check roept `log_uniform(np.random.default_rng(0), 1, 300, 9)` aan, en nog drie keer met andere bereiken, zoals van 0,0001 tot 0,1, een bereik voor een learning rate.
        """),
        L.task("Laat `log_uniform` n getallen gelijk verdeeld trekken op een logaritmische schaal."),
    ])
    return


@app.cell(hide_code=True)
def exercise_log():
    box_log = L.code_box("log-uniform", lines=6)
    box_log
    return (box_log,)


@app.cell(hide_code=True)
def check_log(box_log):
    result_log = L.check("log-uniform", box_log.value)
    result_log
    return (result_log,)


@app.cell(hide_code=True)
def hints_log():
    L.hints(
        "log-uniform",
        [
            "Twee stappen: trek de logaritmen gelijk verdeeld, en ga dan van elke logaritme terug naar zijn getal.",
            "De logaritmen liggen tussen de logaritme van `low` en die van `high`, met grondtal 10. De logaritme van een getal is de exponent van 10 die het getal geeft; terug verhef je 10 tot die macht.",
            "`logs = rng.uniform(np.log10(___), np.log10(___), n)` en `return 10 ** ___`.",
        ],
        why="De logaritmen van 1 en 300 zijn 0 en 2,48. Gelijk verdeeld daartussen getrokken krijgt elke factor 10 ongeveer evenveel trekkingen, en `10 ** logs` maakt er weer getallen van 1 tot 300 van. Met `low=0.0001` en `high=0.1` trekt het op dezelfde manier learning rates.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_few():
    mo.md(r"""
    ### Oefening 3. Vier instellingen, één die ertoe doet
    """)
    return


@app.cell(hide_code=True)
def choice_few_cell():
    choice_few = L.choice(
        "few-matter",
        question="Je kunt je 16 pogingen veroorloven om vier instellingen te zoeken. Je verwacht dat er maar één echt toe doet, maar niet welke. Wat doe je?",
        options={
            "grid": "Een grid met 2 waarden per instelling: 2 · 2 · 2 · 2 = 16 pogingen, elke combinatie",
            "right": "Een random search: elk van de 16 pogingen trekt nieuwe waarden van elke instelling",
            "one": "Eén instelling tegelijk zoeken, 4 pogingen elk, met de andere vast",
        },
        correct="right",
        why={
            "grid": "Dan krijgt elke instelling maar 2 waarden, ook die ene die ertoe doet: 16 pogingen om 2 waarden ervan te vergelijken. Een random search trekt bij elke poging een nieuwe waarde van elke instelling, dus ook die ene krijgt tot 16 verschillende waarden.",
            "one": "Dan krijgt elke instelling 4 waarden, maar elke zoektocht zet de andere drie vast, dus zie je niet hoe ze samenwerken, en die ene die ertoe doet krijgt maar 4 waarden.",
            "right": "Welke instelling er ook toe doet, die krijgt tot 16 verschillende waarden (een geheel getal zoals k kan twee keer komen). Dat is het punt van Bergstra en Bengio: als maar een paar instellingen ertoe doen, verspilt een random search minder pogingen dan een grid.",
        },
    )
    choice_few
    return (choice_few,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Elke poging is de cross-validation uit [Betrouwbaarder kiezen: cross-validation](/course/open/ml-cross-validation?lang=nl), in één regel met de pipeline uit [Elke stap in elke fold: pipelines](/course/open/ml-pipelines?lang=nl): het schalen wordt in elke fold opnieuw geleerd. k en p zijn hyperparameters zoals in [Parameters en hyperparameters](/course/open/ml-hyperparameters?lang=nl): de AUC uit [Alle thresholds tegelijk](/course/open/ml-roc-auc?lang=nl) op achtergehouden folds kiest ze, nooit de loss op de training set. Het trekken van k gebruikt de logaritmische schaal uit [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl). In de volgende les, *Laat de computer kiezen: AutoML*, besteedt een zoektocht zijn pogingen nog slimmer: die geeft veel kandidaten een beetje data en gaat verder met de beste.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_few, result_grid, result_log):
    L.wrap_up(result_grid, result_log, choice_few)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
