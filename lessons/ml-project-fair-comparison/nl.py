# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-project-fair-comparison. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-project-fair-comparison", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Elke stap in elke fold: pipelines](/course/open/ml-pipelines?lang=nl) en [Project: voorspel iets echts, van begin tot eind](/course/open/ml-project-tabular?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_refit():
    quick_refit = L.choice(
        "quick-refit",
        question="Wat doet `cross_val_score(pipe, X, y, cv=KFold(5))` in elke fold met de stappen van `pipe`?",
        options={
            "once": "Het fit ze één keer, op heel `X`, vóór de folds",
            "right": "Het fit een nieuwe kopie ervan op de trainingsvoorbeelden van die fold",
            "asis": "Het gebruikt ze zoals ze het laatst gefit zijn",
        },
        correct="right",
        why={
            "once": "Dan hadden de voorbeelden waarop elke fold gemeten wordt, geholpen om ze te fitten. Zie *Het idee* in [Elke stap in elke fold: pipelines](/course/open/ml-pipelines?lang=nl).",
            "asis": "Elke fold begint met een nieuwe kopie, dus wat een fold geleerd heeft, gaat niet mee naar de volgende. Zie *Het idee* in [Elke stap in elke fold: pipelines](/course/open/ml-pipelines?lang=nl).",
            "right": "Elke stap wordt in elke fold opnieuw geleerd, van de trainingsvoorbeelden van die fold, dus geen stap kan lekken.",
        },
    )
    quick_refit
    return (quick_refit,)


@app.cell(hide_code=True)
def quick_check_dummy():
    quick_dummy = L.choice(
        "quick-dummy",
        question="Wat voorspelt `DummyClassifier()` voor elk nieuw voorbeeld?",
        options={
            "random": "Een willekeurig getrokken klasse",
            "right": "De klasse die in de training set het vaakst voorkomt",
            "near": "De klasse van het dichtstbijzijnde trainingsvoorbeeld",
        },
        correct="right",
        why={
            "random": "Hij trekt niets: hij leert één klasse en zegt altijd die. Zie het recept in [Project: voorspel iets echts, van begin tot eind](/course/open/ml-project-tabular?lang=nl).",
            "near": "Dat is 1-nearest neighbors, een echt model. Zie het recept in [Project: voorspel iets echts, van begin tot eind](/course/open/ml-project-tabular?lang=nl).",
            "right": "Hij is de baseline: wat je haalt zonder naar de features te kijken. Een model moet hem verslaan om iets waard te zijn.",
        },
    )
    quick_dummy
    return (quick_dummy,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Deze module bracht drie modellen: [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl), [Veel bomen stemmen: het random forest](/course/open/ml-random-forest?lang=nl) en [Boom na boom op wat overblijft: gradient boosting](/course/open/ml-gradient-boosting?lang=nl). Daarvoor kwamen [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl) en [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl). Welke neem je voor een nieuwe vraag? Hier: is een pinguïn een mannetje of een vrouwtje, uit zijn vier metingen? Van buiten lijken ze op elkaar, en van 333 pinguïns is het geslacht bekend. Een vergelijking zegt alleen iets als elk model dezelfde kans krijgt: dezelfde pinguïns, dezelfde folds, de stappen die het nodig heeft, een baseline ernaast, en een test set die tot het allerlaatst niets beslist.

    **Een voorbeeld uit het dagelijks leven.** Een eerlijke wedstrijd heeft voor elke loper dezelfde baan, dezelfde afstand en dezelfde start. En wie één race met een honderdste van een seconde wint, is daarmee niet sneller gebleken: loop hem opnieuw en de volgorde kan veranderen.
    """)
    return


@app.cell(hide_code=True)
def predict_best():
    guess_best = L.guess(
        "best-model",
        question="Vijf modellen op dezelfde vijf folds van 266 pinguïns: logistische regressie, k-nearest neighbors, een boom met diepte 3, een random forest en gradient boosting. Welke haalt volgens jou de hoogste score in cross-validation voor het geslacht van een pinguïn?",
        options={
            "boosting": "Gradient boosting",
            "forest": "Het random forest",
            "right": "Een van de eenvoudigste: logistische regressie of k-nearest neighbors",
        },
        reveal="**k-nearest neighbors, 0,913**, dan logistische regressie 0,906, het forest 0,902, boosting 0,895 en de boom 0,864; altijd het geslacht zeggen dat het vaakst voorkomt, haalt 0,451. Maar met 53 pinguïns in een fold is één pinguïn 0,019, en de eerste vier liggen binnen 0,02 van elkaar: één pinguïn per fold.",
    )
    guess_best
    return (guess_best,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Er is vandaag geen nieuw idee: je meet een paar modellen zoals de vorige lessen er één maten, en je zorgt dat de vergelijking eerlijk is.

        1. **Dezelfde voorbeelden en dezelfde folds.** Eén `cv = KFold(5)` voor elk model, zodat elk model in elke fold op precies dezelfde pinguïns traint en gemeten wordt.
        2. **Elk model met de stappen die het nodig heeft**, in zijn eigen pipeline: logistische regressie en k-nearest neighbors standaardiseren eerst; bomen vragen naar één feature tegelijk, dus de schaal maakt voor hen niet uit.
        3. **Een baseline**: `DummyClassifier()`, die altijd het geslacht zegt dat het vaakst voorkomt. Elk model moet hem duidelijk verslaan.
        4. **Fold voor fold.** In elke fold zagen twee modellen dezelfde trainingspinguïns en worden ze op dezelfde andere pinguïns gemeten, dus het verschil per fold zegt meer dan twee gemiddelden die elk een beetje geluk of pech hadden.
        5. **Verschillen tegen de spreiding.** Met 53 pinguïns in een fold is één pinguïn 0,019. Een voorsprong die kleiner is dan hoeveel een score van fold tot fold beweegt, of tussen twee manieren om de folds te snijden, is geen voorsprong waar je op kunt bouwen.
        6. **De test set één keer.** Kies het model en zijn hyperparameters alleen met cross-validation, en besteed aan elk model evenveel zorg. Fit dan het model dat je koos op alle 266 pinguïns en meet het één keer op de 67 testpinguïns.
        """),
        L.watch_out("Hoe meer modellen en instellingen je probeert, hoe meer de beste score in cross-validation ook geluk is: hij is gekozen omdat hij de hoogste was. Daarom komt de test set als laatste, één keer, voor één model, en is zijn getal het getal dat je meldt."),
        mo.md(r"""
        Het plaatje toont de vijf modellen, elk op een rij. Links een open stip voor de accuracy op elk van de 5 folds en een dichte stip voor hun gemiddelde. Rechts het gemiddelde voor negen manieren om dezelfde 266 pinguïns in vijf folds te snijden: manier 1 is `KFold(5)`, op volgorde, manieren 2 tot en met 9 schudden eerst, met `KFold(5, shuffle=True, random_state=0)` tot en met `random_state=7`. De dichte stip is de manier van links. De baseline, rond 0,5, staat er niet in, zodat je de modellen uit elkaar kunt houden. Schuif met de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_cuts_cell():
    def _frame(cut):
        cut = int(cut)
        names = {"logistic": "logistische regressie", "knn": "k-NN", "tree": "boom", "forest": "forest", "boosting": "boosting"}
        listed = ", ".join(f"{names[n]} {L.number(C.mean(C.MEASURED[n][cut - 1]), 3)}" for n in C.MODELS)
        most = max(C.right(n, cut) for n in C.MODELS)
        tied = [names[n] for n in C.MODELS if C.right(n, cut) == most]
        best = tied[0] if len(tied) == 1 else ", ".join(tied[:-1]) + " en " + tied[-1]
        alt = (f"De eerste grafiek: voor elk van de vijf modellen een rij met zijn accuracy op de 5 folds van manier {cut}, tussen {L.number(min(min(C.MEASURED[n][cut - 1]) for n in C.MODELS))} en {L.number(max(max(C.MEASURED[n][cut - 1]) for n in C.MODELS))}, en hun gemiddelde. "
               + "De tweede grafiek: voor elk model het gemiddelde van alle negen manieren; k-NN ligt tussen 0,91 en 0,92, logistische regressie en het forest tussen 0,89 en 0,91, boosting tussen 0,88 en 0,90, de boom tussen 0,83 en 0,89. "
               + f"Manier {cut}: {listed}.")
        said = f"""Manier {cut}, de gemiddelde accuracy: {listed}. De meeste pinguïns goed: **{best}**, {most} van de 266."""
        return (C.picture(cut, {"title": f"Elke fold, manier {cut}", "score": "accuracy", "every": "Het gemiddelde, negen manieren",
                                "mean": "gemiddelde accuracy", "alt": alt, **names}), said)

    explore_cuts = L.explore("cuts", "manier van snijden", _frame, start=1, values=C.CUTS, role="",
                             task="Ga langs de negen manieren om de folds te snijden. Welk model staat steeds bovenaan, en met hoeveel?")
    explore_cuts
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    k-NN heeft in zeven van de negen manieren de meeste pinguïns goed, één tot vier van de 266 meer dan het volgende model; in manier 2 en 3 staat het gelijk met andere. Logistische regressie en het forest liggen tussen 0,89 en 0,91, boosting tussen 0,88 en 0,90, en de boom beweegt het meest, van 0,83 tot 0,89: op welke pinguïns hij traint, verandert zijn vragen. Vier pinguïns van 266 is 0,015, en het gemiddelde van k-NN zelf beweegt zoveel van de ene manier van snijden naar de andere. Dus k-NN ligt een beetje voor, en de beste vier liggen dicht bij elkaar.

    ### Recept

    Om modellen eerlijk te vergelijken:

    1. Houd de test set apart, en vergelijk op de rest.
    2. Maak één `cv = KFold(5)` voor elk model.
    3. Zet elk model in een pipeline met de stappen die het nodig heeft, en voeg `DummyClassifier()` toe als baseline.
    4. Meet elk model met `cross_val_score(model, X, y, cv=cv)`; vergelijk de gemiddelden, en vergelijk de folds een voor een.
    5. Kies hyperparameters op dezelfde manier, op dezelfde folds, en besteed aan elk model evenveel zorg.
    6. Kies één model. Fit het op alle voorbeelden waarop je vergeleek en meet het één keer op de test set; meld dat getal naast de baseline.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De 333 pinguïns met een bekend geslacht, geschud met seed 0: 266 om op te vergelijken, de andere 67 apart gehouden. Stap 1 tot en met 4 van het recept voor de zes modellen, in een dictionary van naam naar model: de boom met diepte 3, de andere met de standaardinstellingen van scikit-learn. `HistGradientBoostingClassifier` is `HistGradientBoostingRegressor` voor klassen: weer kleine bomen na elkaar, elk op wat de bomen ervoor nog fout doen. Voor elk model het gemiddelde over de folds, en met `.min()` en `.max()` de laagste en de hoogste fold; `round(x, 3)` rondt een getal af op 3 decimalen. `random_state=0` legt de toevalstrekkingen van de boom, het forest en boosting vast. Het commentaar met `#` zegt welke stap van het recept elk stuk is. Voer de cel uit: het forest en boosting doen er een paar seconden over.
    """)
    return


@app.cell
def example_compare():
    import numpy as np
    import pandas as pd
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["sex"].isna() == False]
    _order = np.random.default_rng(0).permutation(len(_df))
    # 1. 266 pinguïns om op te vergelijken; de andere 67 zijn de test set
    _compare = _df.iloc[_order[:266]]
    _features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    _X, _y = np.array(_compare[_features]), np.array(_compare["sex"])
    # 2. één manier om de folds te snijden, voor elk model
    _cv = KFold(5)
    # 3. elk model met de stappen die het nodig heeft, en de baseline
    _models = {
        "baseline": DummyClassifier(),
        "logistische regressie": make_pipeline(StandardScaler(), LogisticRegression()),
        "k-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "boom": DecisionTreeClassifier(max_depth=3, random_state=0),
        "forest": RandomForestClassifier(random_state=0),
        "boosting": HistGradientBoostingClassifier(random_state=0),
    }
    # 4. dezelfde folds voor allemaal: het gemiddelde, en de laagste en de hoogste fold
    for _name, _model in _models.items():
        _scores = cross_val_score(_model, _X, _y, cv=_cv)
        print(_name, round(_scores.mean(), 3), round(_scores.min(), 3), round(_scores.max(), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De baseline haalt 0,451, minder dan de helft. In elke fold zegt hij het geslacht dat het vaakst voorkomt onder de trainingspinguïns van die fold, en de twee geslachten komen bijna even vaak voor: fold 5 heeft 68% mannetjes, dus hebben de andere vier folds meer vrouwtjes, en zegt hij voor heel fold 5 vrouwtje. Elk model verslaat hem ruim. De beste vier liggen tussen 0,895 en 0,913, en elk van hun folds ligt ergens tussen 0,83 en 0,98. De boom, met zijn drie vragen, haalt 0,864, en zijn slechtste fold 0,774. Welke van de beste vier de beste is, zeggen deze getallen niet; de volgende oefening kijkt fold voor fold.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_folds():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Fold voor fold

        Het vak leest de 266 pinguïns in zoals in het uitgewerkte voorbeeld, in `X` en `y`, en maakt `cv = KFold(5)`. Vervang de twee `...`: de 5 scores op de folds van logistische regressie en van k-nearest neighbors met k = 5, elk in zijn pipeline met eerst `StandardScaler()`, op de folds `cv`. De laatste regel geeft het verschil per fold.
        """),
        L.task("Vergelijk logistische regressie en k-NN fold voor fold."),
    ])
    return


@app.cell(hide_code=True)
def exercise_folds():
    box_folds = L.code_box("fold-by-fold", lines=21)
    box_folds
    return (box_folds,)


@app.cell(hide_code=True)
def check_folds(box_folds):
    result_folds = L.check("fold-by-fold", box_folds.value)
    result_folds
    return (result_folds,)


@app.cell(hide_code=True)
def hints_folds():
    L.hints(
        "fold-by-fold",
        [
            "Twee regels zoals stap 4 van het uitgewerkte voorbeeld, één voor elk model, met `X`, `y` en `cv`.",
            "`cross_val_score` krijgt het model, hier een pipeline, dan `X`, `y` en de folds. De pipelines zijn die uit het uitgewerkte voorbeeld.",
            "`logistic = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression()), X, y, cv=___)`, en hetzelfde voor `KNeighborsClassifier(n_neighbors=5)`.",
        ],
        why="Met `cv=cv` worden beide modellen in elke fold op precies dezelfde pinguïns gemeten, dus `knn - logistic` zegt per fold hoeveel pinguïns k-NN er meer of minder goed heeft.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_choose():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Kies k, en test dan één keer

        Stap 5 en 6 van het recept voor k-nearest neighbors. Het vak heeft dezelfde `X`, `y` en `cv`, en de 67 testpinguïns in `X_test` en `y_test`. De loop probeert negen waarden van k en houdt in `best` die met de hoogste gemiddelde score in cross-validation. Vervang de drie `...`: die score voor deze `k`, de pipeline met `best`, en in `accuracy` het deel dat hij goed heeft op de testpinguïns. In een volledige vergelijking zou je de diepte van de boom en de instellingen van de andere modellen op dezelfde manier afstellen, op dezelfde folds; hier doe je het alleen voor k.
        """),
        L.task("Kies k met cross-validation, en meet de pipeline met die k één keer op de test set."),
    ])
    return


@app.cell(hide_code=True)
def exercise_choose():
    box_choose = L.code_box("choose-k", lines=28)
    box_choose
    return (box_choose,)


@app.cell(hide_code=True)
def check_choose(box_choose):
    result_choose = L.check("choose-k", box_choose.value)
    result_choose
    return (result_choose,)


@app.cell(hide_code=True)
def hints_choose():
    L.hints(
        "choose-k",
        [
            "De score is de regel uit oefening 1 voor k-NN, met `n_neighbors=k`, en zijn gemiddelde. De test is oefening 2 uit [Elke stap in elke fold: pipelines](/course/open/ml-pipelines?lang=nl).",
            "`cross_val_score` geeft 5 scores; `if score > top` heeft één getal nodig. Na de loop is `best` de k die je gebruikt; de testpinguïns komen pas in de laatste regel met `...`.",
            "`score = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)), X, y, cv=cv).___()`, `pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=___))` en `accuracy = np.mean(pipe.predict(X_test) == ___)`.",
        ],
        why="Elke k wordt beoordeeld op dezelfde folds van de 266 pinguïns, en de testpinguïns spelen bij de keuze geen rol. Alleen de pipeline met de gekozen k ziet ze, één keer, dus zijn accuracy daar is een eerlijke meting van een model dat zo gekozen is.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_test_pick():
    mo.md(r"""
    ### Oefening 3. Kiezen op de test set?
    """)
    return


@app.cell(hide_code=True)
def choice_test_pick_cell():
    choice_test_pick = L.choice(
        "test-pick",
        question="Een collega meet alle vijf modellen uit het uitgewerkte voorbeeld één keer op de 67 testpinguïns: k-NN met k = 5, de boom en boosting halen 0,866, logistische regressie 0,851 en het forest 0,821. De collega wil nu de boom nemen: die deelt de eerste plaats en is de eenvoudigste. Wat zeg je?",
        options={
            "agree": "Goed idee: de test set is de eerlijkste meting die er is",
            "right": "Nee: één pinguïn is hier 0,015, dus deze verschillen één tot drie pinguïns, en een test set waarmee je kiest, is geen eerlijke meting meer",
            "forest": "Neem er maar een, maar laat het forest vallen: dat doet het duidelijk het slechtst op nieuwe pinguïns",
        },
        correct="right",
        why={
            "agree": "Hij is eerlijk voor één model dat je vooraf koos. Kies je er de beste van vijf mee, dan is die score de beste van vijf pogingen, een beetje te goed, zoals de 0,932 uit oefening 2.",
            "forest": "0,821 tegen 0,866 is drie pinguïns van 67, en in cross-validation hoorde het forest bij de beste vier. Zou je het laten vallen om drie pinguïns?",
            "right": "Kies met cross-validation, op de 266 pinguïns, en gebruik de test set één keer, voor het ene model dat je koos. Zijn getal is dan een eerlijke meting; de vijf getallen hier laten vooral zien hoeveel 67 pinguïns kunnen schuiven.",
        },
    )
    choice_test_pick
    return (choice_test_pick,)


@app.cell(hide_code=True)
def text_report():
    mo.md(r"""
    ### Oefening 4. Het verslag
    """)
    return


@app.cell(hide_code=True)
def explain_report_cell():
    explain_report = L.explain(
        "report",
        question="Schrijf in drie of vier zinnen voor de pinguïnonderzoekers op welk model je zou gebruiken om het geslacht van een pinguïn te voorspellen, en waarom: met de baseline, de scores in cross-validation, hoe groot de verschillen zijn, en het getal op de test set.",
        model_answer="We vergeleken vijf modellen en een baseline op dezelfde vijf folds van 266 pinguïns. Altijd het geslacht zeggen dat het vaakst voorkomt, haalt 0,45; k-nearest neighbors op de gestandaardiseerde metingen haalt 0,91, logistische regressie en het random forest 0,90 tot 0,91, boosting 0,89 en een kleine boom 0,86. De beste vier verschillen ongeveer één pinguïn per fold, minder dan de folds van één model onderling verschillen, dus k-NN ligt maar een beetje voor. We kozen k = 9 met cross-validation, en op de 67 testpinguïns haalt hij 0,88: dat is het getal dat je voor nieuwe pinguïns mag verwachten.",
        points=[
            "de baseline naast de modellen: ongeveer 0,45 tegen 0,86 tot 0,91",
            "dat de beste vier dicht bij elkaar liggen: ongeveer één pinguïn per fold",
            "hoe het model en zijn k gekozen zijn: met cross-validation, zonder de test set",
            "het getal op de test set, één keer gemeten: 0,88",
        ],
    )
    explain_report
    return (explain_report,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Je hebt de modellen uit [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl), [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl), [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl), [Veel bomen stemmen: het random forest](/course/open/ml-random-forest?lang=nl) en [Boom na boom op wat overblijft: gradient boosting](/course/open/ml-gradient-boosting?lang=nl) naast elkaar gemeten, elk in zijn pipeline uit [Elke stap in elke fold: pipelines](/course/open/ml-pipelines?lang=nl), naast de baseline uit [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl). In de tak onderzoek vraagt *Eerlijke vergelijkingen* hetzelfde van een paper: kregen de nieuwe methode en de oude dezelfde data, dezelfde zorg en een baseline? In de volgende les, *Een neuron*, beginnen de neural networks, en ook die zullen deze modellen moeten verslaan.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_test_pick, explain_report, result_choose, result_folds):
    L.wrap_up(result_folds, result_choose, choice_test_pick, explain_report)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
