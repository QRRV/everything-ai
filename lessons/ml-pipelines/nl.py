# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-pipelines. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-pipelines", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Te mooi om waar te zijn: data leakage](/course/open/ml-data-leakage?lang=nl) en [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_inside():
    quick_inside = L.choice(
        "quick-inside",
        question="Uit welke voorbeelden hoort feature selection bij 5-fold cross-validation zijn kolommen te kiezen?",
        options={
            "all": "Uit alle voorbeelden, één keer, vóór de folds",
            "right": "In elke fold uit de voorbeelden waarop die fold traint",
            "val": "In elke fold uit de voorbeelden waarop die fold gemeten wordt",
        },
        correct="right",
        why={
            "all": "Dan hielpen de voorbeelden waarop elke fold gemeten wordt de kolommen kiezen: op pure ruis gaf dat 0,96. Zie *Het idee* in [Te mooi om waar te zijn: data leakage](/course/open/ml-data-leakage?lang=nl).",
            "val": "Op die voorbeelden meet de fold: ze mogen niets helpen kiezen. Zie het recept in [Te mooi om waar te zijn: data leakage](/course/open/ml-data-leakage?lang=nl).",
            "right": "Alles wat van data geleerd wordt, hoort bij de training, dus elke fold leert het opnieuw, van zijn eigen trainingsvoorbeelden.",
        },
    )
    quick_inside
    return (quick_inside,)


@app.cell(hide_code=True)
def quick_check_distance():
    quick_distance = L.choice(
        "quick-distance",
        question="De afstand tussen twee pinguïns, uit hun snavellengte en snavelhoogte in millimeters, hun flipperlengte in millimeters en hun lichaamsgewicht in grammen, zonder standaardiseren. Welke meting bepaalt die?",
        options={
            "all": "Alle vier evenveel",
            "right": "Het lichaamsgewicht",
            "bill": "De snavelhoogte",
        },
        correct="right",
        why={
            "all": "Elk verschil telt in zijn eigen eenheid, en grammen lopen in de duizenden. Zie *Het idee* in [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).",
            "bill": "De snavelhoogte verschilt een paar millimeter tussen pinguïns, het lichaamsgewicht honderden grammen. Zie *Het idee* in [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).",
            "right": "Het lichaamsgewicht verschilt honderden grammen, de andere metingen millimeters, dus de afstand is bijna alleen het lichaamsgewicht. Daarom standaardiseert k-nearest neighbors eerst.",
        },
    )
    quick_distance
    return (quick_distance,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl) standaardiseerde je de pinguïns met het gemiddelde en de standaardafwijking van de training set, en in [Te mooi om waar te zijn: data leakage](/course/open/ml-data-leakage?lang=nl) koos je de kolommen in elke fold opnieuw. Met de hand is elke stap die van data leert een paar regels in elke fold, en één vergeten regel lekt al. Neem de soort van een pinguïn uit zijn vier metingen, met de 5 dichtstbijzijnde buren, op 274 pinguïns: 5-fold cross-validation geeft 0,759 zonder standaardiseren, en 0,985 als het standaardiseren in elke fold geleerd wordt. Het zou mooi zijn als de stappen en het model één geheel waren, dat elke fold vanzelf opnieuw leert.

    **Een voorbeeld uit het dagelijks leven.** Voor elke vlucht loopt een piloot een checklist af: dezelfde stappen, in dezelfde volgorde, elke keer weer. Niemand vertrouwt daarvoor op het geheugen, hoe goed ook, want één vergeten stap is genoeg.
    """)
    return


@app.cell(hide_code=True)
def predict_scale_first():
    guess_scale_first = L.guess(
        "scale-first",
        question="Standaardiseer eerst alle 274 pinguïns, met hun eigen gemiddelde en standaardafwijking, en doe daarna de cross-validation met alleen het model. Standaardiseren binnen elke fold gaf 0,985. Wat denk je dat dit geeft?",
        options={
            "higher": "Veel hoger, zoals op de ruis uit de vorige les",
            "right": "Ongeveer hetzelfde",
            "lower": "Lager",
        },
        reveal="**Hetzelfde, 0,985.** Het gemiddelde en de standaardafwijking van alle 274 pinguïns verschillen nauwelijks van die van de 219 of 220 waarop een fold traint, dus dit lek is piepklein. 20 van de 1000 kolommen kiezen was een enorm lek. Een pipeline maakt ze allebei onmogelijk, zodat je nooit hoeft uit te zoeken welk soort je hebt.",
    )
    guess_scale_first
    return (guess_scale_first,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **pipeline** is een vaste keten van stappen met een model aan het eind, die je als één model fit en gebruikt. Elke stap vóór het model leert iets van de data en verandert daarmee de data: `StandardScaler()` leert het gemiddelde en de standaardafwijking van elke feature en standaardiseert daarmee, `SelectKBest(k=20)` leert welke 20 features hij houdt en houdt die. In scikit-learn heeft zo'n stap `fit`, die leert, en `transform`, die de data verandert met wat geleerd is.

        `pipe.fit(X, y)` gaat de stappen op volgorde langs: elke stap leert van de data die hij krijgt, verandert die data en geeft die door, en het model leert van wat er aan het eind aankomt. `pipe.predict(X_new)` gaat dezelfde stappen langs, maar verandert de data alleen, met wat van `X` geleerd is, en voorspelt dan. Dat is precies de regel uit de vorige les: alles wordt van de trainingsvoorbeelden geleerd, en nieuwe voorbeelden worden er alleen mee veranderd.

        Cross-validation heeft dan geen loop van jou meer nodig. `cross_val_score(pipe, X, y, cv=KFold(5))` maakt voor elke fold een nieuwe kopie van de hele pipeline, fit die op de trainingsvoorbeelden van die fold en meet hem op de andere, en geeft de 5 scores. Elke stap wordt in elke fold opnieuw geleerd, dus geen stap kan lekken, hoeveel stappen er ook zijn.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **pipeline** | een vaste keten van stappen die van data leren, met een model aan het eind, die je als één model fit en gebruikt | `make_pipeline(StandardScaler(), KNeighborsClassifier())` |
        | `StandardScaler()` | standaardiseren als stap: `fit` leert het gemiddelde en de standaardafwijking van elke feature, `transform` standaardiseert daarmee | |
        | `SelectKBest(k=20)` | feature selection als stap: houdt de 20 features waarvan de gemiddelden het meest verschillen tussen de labels, vergeleken met de spreiding binnen elk label | |
        | `KFold(5)` | 5 folds, op volgorde gesneden, zoals je met de hand deed; de voorbeelden zijn al geschud | |
        | `cross_val_score(pipe, X, y, cv=KFold(5))` | cross-validation in één regel: fit in elke fold een nieuwe kopie van `pipe` en geeft de score van elke fold | |
        """),
        L.watch_out("Een pipeline beschermt alleen de stappen die erin zitten. Wat je met `X` doet voordat je hem aan de pipeline geeft, zoals alles standaardiseren, gebeurt één keer, op alle voorbeelden, en zit in elke fold."),
        mo.md(r"""
        Het beeld laat k-nearest neighbors zien op de 274 trainingspinguïns met alle vier de metingen. Links de accuracy op elk van de 5 folds voor één k: volle stippen met `StandardScaler()` als eerste stap van de pipeline, open stippen zonder. Rechts het gemiddelde over de folds voor k van 1 tot 51, op dezelfde twee manieren. Verschuif de slider, of sleep over die grafiek.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_k_cell():
    def _frame(k):
        k = int(k)
        scaled, raw = C.fold_scores(k, True), C.fold_scores(k, False)
        alt = (f"De eerste grafiek: de accuracy op elk van de 5 folds met k = {k}: met standaardiseren tussen {L.number(min(scaled))} en {L.number(max(scaled))}, zonder tussen {L.number(min(raw))} en {L.number(max(raw))}. "
               + "De tweede grafiek: de gemiddelde accuracy voor k van 1 tot 51; met standaardiseren tussen 0,96 en 0,99, zonder tussen 0,74 en 0,78, voor elke k. "
               + f"Met k = {k}: {L.number(C.mean(scaled), 3)} en {L.number(C.mean(raw), 3)}.")
        said = f"""k = {k}: met standaardiseren in de pipeline **{L.number(C.mean(scaled), 3)}**, zonder **{L.number(C.mean(raw), 3)}**."""
        return (C.picture(k, {"title": f"Elke fold, k = {k}", "fold": "fold", "score": "accuracy",
                              "every": "Gemiddelde over de folds", "k": "k", "mean": "gemiddelde accuracy",
                              "scaled": "met StandardScaler", "raw": "zonder", "alt": alt}), said)

    explore_k = L.explore("k", "k", _frame, start=5, values=C.KS, role="hyper",
                          task="Verschuif k van 1 naar 51. Is er een k waarbij het model zonder standaardiseren het even goed doet, in welke fold dan ook?")
    explore_k
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Zonder standaardiseren blijft het gemiddelde tussen 0,74 en 0,78, wat k ook is, en in elke fold en voor elke k doet de pipeline met `StandardScaler()` het beter: de buren werden bijna alleen op lichaamsgewicht gekozen. Met standaardiseren ligt het gemiddelde tussen 0,96 en 0,99. k = 5 en k = 7 komen er het best uit, 0,985, en verschillen minder dan één pinguïn: de folds kunnen ze niet uit elkaar houden.

    ### Recept

    Zo meet je een model met stappen die van data leren:

    1. Zet de stappen op volgorde, het model als laatste: `pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))`.
    2. Meet hem met cross-validation, op dezelfde folds voor elk model dat je vergelijkt: `scores = cross_val_score(pipe, X, y, cv=KFold(5))`, en neem `scores.mean()`.
    3. Vergelijk pipelines, of waarden van een hyperparameter, op dat gemiddelde, zoals in [Betrouwbaarder kiezen: cross-validation](/course/open/ml-cross-validation?lang=nl).
    4. Fit de pipeline die je kiest op alle trainingsvoorbeelden, `pipe.fit(X, y)`, en meet hem één keer op de test set, met `pipe.predict(X_test)`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De 342 pinguïns met alle vier de metingen, geschud met seed 0: de eerste 274 om op te trainen en mee te kiezen, de andere 68 apart gehouden als test set. De vier metingen zijn de features, de soort is het label. Stap 1 tot en met 4 voor 5-nearest neighbors, eerst zonder standaardiseren, daarna met `StandardScaler()` als eerste stap; `np.round` laat de 5 scores van de folds zien met 3 decimalen, en `round(x, 3)` rondt één getal af op 3 decimalen. Aan het eind is `_pipe[0]` de eerste stap van de gefitte pipeline, en zijn `mean_` is het gemiddelde dat hij voor elke feature leerde. Het commentaar met `#` zegt welke stap van het recept elk stuk is. Voer de cel uit.
    """)
    return


@app.cell
def example_pipeline():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["body_mass_g"].isna() == False]
    _order = np.random.default_rng(0).permutation(len(_df))
    _train = _df.iloc[_order[:274]]
    _features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    _X, _y = np.array(_train[_features]), np.array(_train["species"])
    # 2. dezelfde 5 folds voor elk model
    _cv = KFold(5)
    # zonder standaardiseren: alleen het model
    _scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), _X, _y, cv=_cv)
    print(np.round(_scores, 3), round(_scores.mean(), 3))
    # 1. de stappen op volgorde, het model als laatste
    _pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    _scores = cross_val_score(_pipe, _X, _y, cv=_cv)
    print(np.round(_scores, 3), round(_scores.mean(), 3))
    # 4. gefit op alle 274 trainingspinguïns: wat de eerste stap leerde
    _pipe.fit(_X, _y)
    print(np.round(_pipe[0].mean_, 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Zonder standaardiseren scoren de folds tussen 0,727 en 0,8, gemiddeld 0,759; met `StandardScaler()` in de pipeline hebben drie folds elke pinguïn goed en is het gemiddelde 0,985. In elk van de 5 fits leerde de scaler het gemiddelde en de standaardafwijking van alleen de 219 of 220 trainingspinguïns van die fold. Gefit op alle 274 leerde hij een gemiddelde van 44 mm voor de snavellengte, 17,1 mm voor de snavelhoogte, 201,4 mm voor de vleugellengte en 4235,1 g voor het lichaamsgewicht, en daarmee standaardiseert hij elke nieuwe pinguïn, zoals in stap 4 van het recept uit [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def text_exercise_noise():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Feature selection als stap

        Het vak trekt de tabel met ruis uit [Te mooi om waar te zijn: data leakage](/course/open/ml-data-leakage?lang=nl) met seed 6: 50 mensen, 1000 kolommen willekeurige getallen en een willekeurig label. Vervang de twee `...`: een pipeline met `SelectKBest(k=20)` als stap en `KNeighborsClassifier(n_neighbors=5)` als model, en zijn scores met `cross_val_score`, op de folds `cv=KFold(5)`. De check kijkt naar het gemiddelde op de laatste regel, en probeert ook een andere tabel.
        """),
        L.task("Laat een pipeline de 20 kolommen binnen elke fold kiezen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_noise():
    box_noise = L.code_box("noise-pipeline", lines=15)
    box_noise
    return (box_noise,)


@app.cell(hide_code=True)
def check_noise(box_noise):
    result_noise = L.check("noise-pipeline", box_noise.value)
    result_noise
    return (result_noise,)


@app.cell(hide_code=True)
def hints_noise():
    L.hints(
        "noise-pipeline",
        [
            "Stap 1 en 2 uit het uitgewerkte voorbeeld, met `SelectKBest(k=20)` in plaats van `StandardScaler()`, en `X` en `y` in plaats van `_X` en `_y`.",
            "`make_pipeline` krijgt de stappen op volgorde, het model als laatste. `cross_val_score` krijgt de pipeline, heel `X`, heel `y` en de folds.",
            "`pipe = make_pipeline(SelectKBest(k=20), ___)` en `scores = cross_val_score(pipe, X, y, cv=___)`.",
        ],
        why="`SelectKBest(k=20)` is de eerste stap, dus in elke fold fit `cross_val_score` hem alleen op de 40 trainingsmensen van die fold, en krijgen de buren alleen die 20 kolommen. De kolommen hebben de 10 mensen waarop gemeten wordt nooit gezien.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_test():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Eén keer fitten, één keer meten

        Stap 4 van het recept. Het vak leest de pinguïns zoals in het uitgewerkte voorbeeld, met de 274 trainingspinguïns in `X_train` en `y_train` en de 68 testpinguïns in `X_test` en `y_test`. Vervang de drie `...`: de pipeline van `StandardScaler()` en `KNeighborsClassifier(n_neighbors=5)`, de fit op de trainingspinguïns, en in `accuracy` het deel van de testpinguïns dat hij in de goede soort zet.
        """),
        L.task("Fit de pipeline op de trainingspinguïns en meet hem één keer op de testpinguïns."),
    ])
    return


@app.cell(hide_code=True)
def exercise_test():
    box_test = L.code_box("test-once", lines=18)
    box_test
    return (box_test,)


@app.cell(hide_code=True)
def check_test(box_test):
    result_test = L.check("test-once", box_test.value)
    result_test
    return (result_test,)


@app.cell(hide_code=True)
def hints_test():
    L.hints(
        "test-once",
        [
            "Stap 1 en het eind van het uitgewerkte voorbeeld, met `X_train` en `y_train`; voorspel daarna voor de testpinguïns.",
            "`pipe.fit` krijgt de trainingspinguïns en hun soort. `pipe.predict(X_test)` geeft een soort voor elke testpinguïn: waarmee vergelijk je die, en hoe krijg je het deel dat hetzelfde is?",
            "`pipe.fit(X_train, ___)` en `accuracy = np.mean(pipe.predict(___) == y_test)`.",
        ],
        why="`pipe.fit` leert het gemiddelde, de standaardafwijking en de buren alleen van de trainingspinguïns. `pipe.predict(X_test)` standaardiseert de testpinguïns met wat hij leerde en voorspelt, dus de testpinguïns worden gemeten zoals nieuwe pinguïns gemeten zouden worden.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_scaled_first():
    mo.md(r"""
    ### Oefening 3. Gestandaardiseerd vóór de pipeline
    """)
    return


@app.cell(hide_code=True)
def choice_scaled_first_cell():
    choice_scaled_first = L.choice(
        "scaled-first",
        question="Een collega standaardiseert alle 274 trainingspinguïns met hun eigen gemiddelde en standaardafwijking, en geeft ze daarna aan `cross_val_score` met een pipeline van alleen `KNeighborsClassifier(n_neighbors=5)`. Is die score eerlijk?",
        options={
            "refit": "Ja: `cross_val_score` fit het model in elke fold opnieuw",
            "right": "Niet helemaal: de pinguïns waarop elke fold gemeten wordt, hielpen het gemiddelde en de standaardafwijking maken; `StandardScaler()` hoort in de pipeline",
            "labels": "Ja: standaardiseren kijkt niet naar de labels, dus er kan niets lekken",
        },
        correct="right",
        why={
            "refit": "Het model wordt opnieuw gefit, maar het standaardiseren niet: dat gebeurde één keer, vóór de folds. Van welke pinguïns kwam het gemiddelde?",
            "labels": "Zonder de labels is het lek klein: op de pinguïns is de score hetzelfde, 0,985. Maar de pinguïns waarop elke fold gemeten wordt, hielpen het gemiddelde en de standaardafwijking toch maken, en met andere stappen of minder voorbeelden kan dat uitmaken. Zie de gok in *Waarom*.",
            "right": "Hier maakt het niets uit, 0,985 allebei, maar dat kun je niet van tevoren weten. Met `StandardScaler()` als eerste stap leert elke fold zijn eigen gemiddelde en standaardafwijking van zijn eigen trainingspinguïns.",
        },
    )
    choice_scaled_first
    return (choice_scaled_first,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Een pipeline is de regel uit [Te mooi om waar te zijn: data leakage](/course/open/ml-data-leakage?lang=nl), ingebouwd in de code: elke stap leert alleen van het trainingsdeel. Zijn eerste stap is hier het standaardiseren uit [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl) en [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl), en `cross_val_score` doet de folds uit [Betrouwbaarder kiezen: cross-validation](/course/open/ml-cross-validation?lang=nl) in één regel, met de `fit` en `predict` uit [Hetzelfde model met scikit-learn](/course/open/ml-sklearn?lang=nl). In de volgende les, *Project: welk model is het best? Een eerlijke vergelijking*, zet je de modellen uit deze module naast elkaar, elk in zijn eigen pipeline, op dezelfde folds.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_scaled_first, result_noise, result_test):
    L.wrap_up(result_noise, result_test, choice_scaled_first)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
