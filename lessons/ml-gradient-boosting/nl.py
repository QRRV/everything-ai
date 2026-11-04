# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-gradient-boosting. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.
# termen:allow residual

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-gradient-boosting", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Hoe ver zit één voorspelling ernaast?](/course/open/ml-residuals?lang=nl) en [Veel bomen stemmen: het random forest](/course/open/ml-random-forest?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_residual():
    quick_residual = L.choice(
        "quick-residual",
        question="Een pinguïn weegt 4000 g, en een model voorspelt 4300 g. Wat is het residu?",
        options={
            "size": "300 g",
            "right": "−300 g",
            "pred": "4300 g",
        },
        correct="right",
        why={
            "size": "Het residu houdt zijn teken, dus het zegt ook naar welke kant de voorspelling ernaast zit. Zie *Het idee* in [Hoe ver zit één voorspelling ernaast?](/course/open/ml-residuals?lang=nl).",
            "pred": "Dat is de voorspelling zelf. Het residu zegt hoe ver die ernaast zit. Zie *Het idee* in [Hoe ver zit één voorspelling ernaast?](/course/open/ml-residuals?lang=nl).",
            "right": "Het label min de voorspelling: 4000 − 4300 = −300, negatief omdat de voorspelling te hoog is.",
        },
    )
    quick_residual
    return (quick_residual,)


@app.cell(hide_code=True)
def quick_check_forest():
    quick_forest = L.choice(
        "quick-forest",
        question="Waarvan leert elke boom van een random forest?",
        options={
            "fix": "Van wat de bomen voor hem fout hadden",
            "right": "Van zijn eigen bootstrap-steekproef, zonder naar de andere bomen te kijken",
            "val": "Van de validation set",
        },
        correct="right",
        why={
            "fix": "De bomen van een forest kijken niet naar elkaar: elke boom groeit op zijn eigen bootstrap-steekproef. Zie *Het idee* in [Veel bomen stemmen: het random forest](/course/open/ml-random-forest?lang=nl).",
            "val": "De validation set is om te kiezen, niet om van te leren. Zie het recept in [Veel bomen stemmen: het random forest](/course/open/ml-random-forest?lang=nl).",
            "right": "Elke boom leert alles op eigen houtje, en de stemming stemt de fouten weg die ze niet delen.",
        },
    )
    quick_forest
    return (quick_forest,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De bomen uit [Veel bomen stemmen: het random forest](/course/open/ml-random-forest?lang=nl) zijn diep, en elke boom leert de labels zelf, uit zijn eigen bootstrap sample; het forest bestrijdt hun variantie. Voorspel nu een getal met dezelfde twee metingen: het lichaamsgewicht van een pinguïn uit zijn snavellengte en snavelhoogte. Op de 68 pinguïns uit de validation set zit een forest van 100 van zulke bomen er 447 g naast (de RMSE uit [Hoe ver ernaast? MAE, RMSE en R²](/course/open/ml-regression-metrics?lang=nl)), een lijn 611 g, en altijd het gemiddelde zeggen 801 g. Een kleine boom, twee vragen diep, zit er 535 g naast: te eenvoudig, dus hij underfit. Wat als kleine bomen achter elkaar werken, en elke boom alleen leert wat de bomen voor hem nog fout hebben?

    **Een voorbeeld uit het dagelijks leven.** Een golfer slaat de bal zelden in één slag in de hole. De eerste slag brengt hem het grootste deel van de weg. Elke volgende slag mikt op wat er nog over is, vanaf waar de bal nu ligt, en vlak bij de hole worden de slagen klein en voorzichtig. Niemand slaat opnieuw de hele afstand.
    """)
    return


@app.cell(hide_code=True)
def predict_rounds():
    guess_rounds = L.guess(
        "forty-two-trees",
        question="42 kleine bomen achter elkaar: elke boom leert wat de bomen voor hem nog fout hebben, en telt mee voor een tiende van wat hij voorspelt. Hoe ver zitten ze er volgens jou naast?",
        options={
            "small": "Ongeveer 535 g, zoals één kleine boom",
            "right": "Minder dan de 447 g van het forest",
            "worse": "Meer dan 535 g: kleine bomen blijven te eenvoudig",
        },
        reveal="**415 g**, dichterbij dan het forest van 100 diepe bomen, met 447 g. Elke kleine boom op zich is te eenvoudig, maar samen, terwijl elke boom een deel herstelt van wat er nog over is, volgen ze de pinguïns veel beter.",
    )
    guess_rounds
    return (guess_rounds,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Boosting** bouwt een ensemble model voor model op: elk nieuw model leert van wat de modellen ervoor nog fout hebben. **Gradient boosting** doet dat met kleine bomen en residuen. Het begint met dezelfde voorspelling voor elk voorbeeld: het gemiddelde van de labels. Daarna rekent het ronde na ronde de residuen uit, het label min de voorspelling; laat het een kleine boom die leren; en telt het een klein deel van wat die boom voorspelt op bij de voorspelling. Dat deel is de **learning rate**, net als bij gradient descent: met 0,1 herstelt elke boom een tiende van wat hij vond. Het model is de start plus 0,1 keer wat al zijn bomen voorspellen.

        De bomen zijn **regressiebomen**: beslisbomen die een getal voorspellen. Ze stellen dezelfde vragen met een threshold, maar een blad voorspelt het gemiddelde van zijn voorbeelden uit de training set in plaats van de klasse die het vaakst voorkomt. Hier zijn die voorbeelden residuen, dus een blad zegt hoeveel er bij elke pinguïn die erin terechtkomt, bij moet.

        Waarom *gradient*? De afgeleide van het gekwadrateerde residu (y − ŷ)² naar de voorspelling ŷ is −2(y − ŷ): het residu keer −2. De residuen wijzen dus bergaf, en er een klein deel van optellen is een stap gradient descent waarin de voorspellingen zelf bewegen. De boom zet die stap ook voor nieuwe voorbeelden, waarvan je het label niet kent.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **boosting** | een ensemble model voor model opbouwen, waarbij elk model leert wat de modellen ervoor nog fout hebben | |
        | **gradient boosting** | boosting met kleine bomen die de residuen leren, en een klein deel van hun voorspelling erbij tellen | `GradientBoostingRegressor()` |
        | **regressieboom** | een beslisboom die een getal voorspelt: elk blad voorspelt het gemiddelde van zijn voorbeelden uit de training set | `DecisionTreeRegressor(max_depth=2)` |
        | `learning_rate=0.1` | het deel van wat elke boom voorspelt dat erbij komt | |
        | `n_estimators=42` | hoeveel bomen achter elkaar | |
        """),
        L.watch_out("Een forest en boosting combineren allebei bomen, maar andersom. Een forest laat diepe bomen naast elkaar groeien en middelt ze: dat verlaagt de variantie. Boosting laat kleine bomen na elkaar groeien en telt ze op: dat verlaagt de bias uit [Helpt meer data? Bias, variantie en learning curves](/course/open/ml-bias-variance?lang=nl), wat een model fout doet omdat het te eenvoudig is."),
        mo.md(r"""
        Het beeld laat boosting op de pinguïns zien. Links een stip voor elke pinguïn uit de validation set, naar rechts zijn echte lichaamsgewicht en omhoog het gewicht dat de eerste n bomen voorspellen: op de grijze diagonaal klopt een voorspelling precies, en het <span class="c-residual">streepje</span> ernaartoe is het residu. De kleine open grijze stippen zijn de trainingspinguïns. De andere grafiek heeft de RMSE op de training set (grijs) en op de validation set (<span class="c-loss">paars</span>) voor 1 tot 1000 bomen, en de 447 g van het forest als stippellijn. Verschuif de slider, of sleep over die grafiek.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_rounds_cell():
    def _frame(n):
        n = int(n)
        b = C.boosted()
        trees = f"{n} {'boom' if n == 1 else 'bomen'}"
        val, train = round(b["val"][n - 1]), round(b["train"][n - 1])
        alt = (f"De eerste grafiek: elke pinguïn uit de validation set als een stip, naar rechts zijn echte lichaamsgewicht van 2850 tot 6000 g en omhoog het gewicht dat {trees} voorspellen, met een streepje naar de diagonaal waar de voorspelling precies zou kloppen. "
               + ("De voorspellingen liggen allemaal dicht bij het gemiddelde, 4243 g, in een paar horizontale rijen." if n < 5 else "De stippen liggen langs de diagonaal." if n <= 100 else "De trainingspinguïns liggen op de diagonaal, de pinguïns uit de validation set er verder vanaf.")
               + f" De tweede grafiek: de RMSE voor 1 tot 1000 bomen; op de training set daalt hij helemaal, tot 158 g, op de validation set daalt hij tot 415 g bij 42 bomen en stijgt hij daarna tot 575 g. Met {trees}: {val} g op de validation set, {train} g op de training set.")
        said = f"""{trees}: <span class="c-loss ea-num">{val} g</span> ernaast op de pinguïns uit de validation set, en {train} g op de trainingspinguïns."""
        return (C.picture(n, {"title": f"Gradient boosting, {trees}", "x": "echt lichaamsgewicht (g)", "y": "voorspeld (g)",
                              "every": "RMSE", "trees": "bomen", "rmse": "RMSE (g)", "forest": "random forest", "alt": alt}), said)

    explore_rounds = L.explore("rounds", "bomen", _frame, start=1, values=C.SIZES, role="hyper",
                               task="Schuif het aantal bomen van 1 naar 1000. Waar is de RMSE op de validation set het kleinst, en wat doet de RMSE op de training set daarna?")
    explore_rounds
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Met 1 boom ligt elke voorspelling nog dicht bij het gemiddelde, in vier rijen: de boom heeft vier bladeren, en hij schoof de pinguïns maar een tiende van de weg op. Met 10 bomen draaien de stippen naar de diagonaal toe. Bij 42 bomen is de RMSE op de validation set het kleinst, 415 g, en tussen 30 en 60 bomen blijft hij onder 420 g. Daarna blijft de RMSE op de training set dalen, tot 158 g bij 1000 bomen, maar stijgt de RMSE op de validation set weer, tot 575 g. Zodra de bomen hebben geleerd wat zware pinguïns gemeen hebben, zitten er in de residuen alleen nog de eigenaardigheden van losse trainingspinguïns, en de volgende bomen leren die ook. Het aantal bomen is dus een hyperparameter, die je kiest op de validation set.

    ### Recept

    Zo maak je gradient boosting met T bomen en learning rate `rate`:

    1. Begin elke voorspelling bij het gemiddelde van de labels: `pred = np.zeros(len(y)) + np.mean(y)`.
    2. Reken de residuen uit: `residuals = y - pred`.
    3. Laat een kleine regressieboom ze leren: `DecisionTreeRegressor(max_depth=2).fit(X, residuals)`.
    4. Tel een deel op van wat hij voorspelt, en bewaar de boom: `pred = pred + rate * tree.predict(X)`.
    5. Doe stap 2 tot en met 4 T keer; kies T op de validation set.
    6. Om voor nieuwe voorbeelden te voorspellen: het gemiddelde plus `rate` keer wat elke boom voor ze voorspelt.

    scikit-learn doet het allemaal: `GradientBoostingRegressor(n_estimators=T, learning_rate=rate, max_depth=2)`, met `fit` en `predict`. Voor een grote tabel is `HistGradientBoostingRegressor()` veel sneller: die verdeelt elke feature eerst over hooguit 255 bakjes, een histogram, en probeert alleen de thresholds tussen de bakjes. Hij kan ook zelf met ontbrekende waarden omgaan.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De pinguïns uit het project, geschud met dezelfde seed: 206 om van te leren en 68 om mee te kiezen, met snavellengte en snavelhoogte als features en lichaamsgewicht als label. Stap 1 tot en met 6 voor 42 bomen, met de RMSE op de training set na de eerste drie; daarna de vragen van de eerste boom, die `export_text` uitschrijft zoals in [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl); daarna de RMSE op de validation set van de 42 bomen, van het model van scikit-learn met dezelfde instellingen, en van `HistGradientBoostingRegressor` met zijn eigen instellingen. `random_state=0` legt de volgorde vast waarin een boom de features probeert, zodat je dezelfde bomen krijgt. Het commentaar met `#` zegt welke stap van het recept elk stuk is. Voer de cel uit.
    """)
    return


@app.cell
def example_boosting():
    import math

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor, export_text

    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["body_mass_g"].isna() == False]
    _order = np.random.default_rng(0).permutation(len(_df))
    _train = _df.iloc[_order[:206]]
    _val = _df.iloc[_order[206:274]]
    _features = ["bill_length_mm", "bill_depth_mm"]
    _X_train, _y_train = np.array(_train[_features]), np.array(_train["body_mass_g"])
    _X_val, _y_val = np.array(_val[_features]), np.array(_val["body_mass_g"])
    # 1. dezelfde start voor elke pinguïn: het gemiddelde lichaamsgewicht van de training set
    _start = np.mean(_y_train)
    _pred = np.zeros(206) + _start
    _trees = []
    # 5. stap 2 tot en met 4, 42 keer
    for _i in range(42):
        # 2. de residuen: wat de bomen tot nu toe nog fout hebben
        _residuals = _y_train - _pred
        # 3. een kleine regressieboom leert ze
        _tree = DecisionTreeRegressor(max_depth=2, random_state=0).fit(_X_train, _residuals)
        # 4. een tiende van wat hij voorspelt komt erbij, en de boom wordt bewaard
        _pred = _pred + 0.1 * _tree.predict(_X_train)
        _trees.append(_tree)
        if _i < 3:
            print(round(math.sqrt(np.mean((_y_train - _pred) ** 2))))
    print(export_text(_trees[0], feature_names=_features))
    # 6. de pinguïns uit de validation set: de start plus een tiende van wat elke boom voorspelt
    _pred_val = np.zeros(68) + _start
    for _tree in _trees:
        _pred_val = _pred_val + 0.1 * _tree.predict(_X_val)
    print(round(math.sqrt(np.mean((_y_val - _pred_val) ** 2))))
    # scikit-learn: stap 1 tot en met 6 in één model
    _model = GradientBoostingRegressor(n_estimators=42, learning_rate=0.1, max_depth=2, random_state=0)
    _model.fit(_X_train, _y_train)
    print(round(math.sqrt(np.mean((_y_val - _model.predict(_X_val)) ** 2))))
    # de snelle voor grote tabellen, met zijn eigen instellingen
    _fast = HistGradientBoostingRegressor(random_state=0).fit(_X_train, _y_train)
    print(round(math.sqrt(np.mean((_y_val - _fast.predict(_X_val)) ** 2))))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De RMSE op de training set gaat van 821 g bij de start naar 770, 727 en 689 g: elke boom zet een tiende van een stap. De eerste boom vraagt eerst naar de snavelhoogte. De 72 trainingspinguïns met een snavel van hooguit 16,45 mm hoog en langer dan 38,2 mm, allemaal Gentoos, zijn gemiddeld 848 g zwaarder dan het gemiddelde, dus deze boom telt er een tiende van, 84,8 g, bij elk van hen op. Zijn bladeren zijn gemiddelden van residuen: `value: [848.48]` is hoeveel zwaarder, geen lichaamsgewicht. Na 42 bomen is de RMSE op de validation set 415 g, en het model van scikit-learn met dezelfde instellingen geeft dezelfde 415 g. `HistGradientBoostingRegressor` laat 100 bomen van elk hooguit 31 bladeren groeien, ook met learning rate 0,1, en komt op 398 g.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_round():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Eén ronde

        Stap 2 tot en met 4 van het recept, één keer. Het vak leest de trainingspinguïns zoals in het uitgewerkte voorbeeld, met `X` en `y`, en begint elke voorspelling in `pred` bij het gemiddelde. Vervang de drie `...`: de residuen, een regressieboom met diepte 2 die ze leert, en de nieuwe `pred`, met `rate` als learning rate. De check probeert ook een andere `rate`, en een andere start.
        """),
        L.task("Doe één ronde gradient boosting op `pred`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_round():
    box_round = L.code_box("one-round", lines=16)
    box_round
    return (box_round,)


@app.cell(hide_code=True)
def check_round(box_round):
    result_round = L.check("one-round", box_round.value)
    result_round
    return (result_round,)


@app.cell(hide_code=True)
def hints_round():
    L.hints(
        "one-round",
        [
            "Stap 2, 3 en 4 van het uitgewerkte voorbeeld, met `X`, `y` en `rate` in plaats van `_X_train`, `_y_train` en 0.1.",
            "Een residu is het label min de voorspelling. De boom leert `X` met de residuen als labels: welke twee dingen komen tussen de haakjes van `fit`? En `pred` krijgt er iets bij.",
            "`residuals = y - ___`, `tree = DecisionTreeRegressor(max_depth=2).fit(X, ___)` en `pred = pred + ___ * tree.predict(X)`.",
        ],
        why="`y - pred` is wat er voor elke pinguïn nog fout is. De boom leert het uit de snavelmetingen, zodat hij het voor elke pinguïn kan voorspellen, en `pred + rate * tree.predict(X)` schuift elke voorspelling een deel `rate` op in de richting die de boom zegt.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_predict():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Voorspellen

        Stap 6 van het recept in een functie. `predict(start, trees, rate, X)` krijgt de start, de lijst bomen, de learning rate en de features van nieuwe voorbeelden, en geeft voor elk voorbeeld de voorspelling terug: de start plus `rate` keer wat elke boom voorspelt. Vervang de twee `...`. De check roept `predict(start, trees, 0.1, X_new)` aan met de start en de eerste drie bomen uit het uitgewerkte voorbeeld, en vier pinguïns uit de validation set: dat moet `[4078.5, 4509.6, 4118.5, 4434.6]` geven. Hij probeert ook andere bomen en een andere `rate`.
        """),
        L.task("Laat `predict` de voorspelling van de start en alle bomen geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_predict():
    box_predict = L.code_box("predict", lines=7)
    box_predict
    return (box_predict,)


@app.cell(hide_code=True)
def check_predict(box_predict):
    result_predict = L.check("predict", box_predict.value)
    result_predict
    return (result_predict,)


@app.cell(hide_code=True)
def hints_predict():
    L.hints(
        "predict",
        [
            "Stap 6 van het uitgewerkte voorbeeld, met `start`, `trees`, `rate` en `X`.",
            "Waar begint elke voorspelling, nog voor er een boom is? En wat wordt `pred` in de loop: de voorspelling tot nu toe, plus wat?",
            "`pred = np.zeros(len(X)) + ___` en `pred = pred + ___ * tree.predict(X)`.",
        ],
        why="Elke voorspelling begint bij `start`, en elke boom telt er `rate` keer zijn voorspelling bij op. De bomen worden opgeteld, niet gemiddeld: elke boom leerde wat de bomen ervoor overlieten.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_too_many():
    mo.md(r"""
    ### Oefening 3. Waarom kunnen meer bomen het slechter maken?
    """)
    return


@app.cell(hide_code=True)
def choice_too_many_cell():
    choice_too_many = L.choice(
        "too-many",
        question="Bij het random forest maakten meer bomen de stemming stabieler: met 100 maakten alle tien forests dezelfde 3 fouten. Hier is de RMSE op de validation set het kleinst bij 42 bomen en stijgt hij tot 575 g bij 1000. Waarom kunnen meer bomen boosting slechter maken?",
        options={
            "small": "Omdat zijn bomen klein zijn",
            "right": "Elke nieuwe boom leert wat er van de residuen op de training set over is; als het patroon geleerd is, zijn dat de eigenaardigheden van losse trainingspinguïns, die nieuwe pinguïns niet hebben",
            "rate": "Omdat een learning rate van 0,1 te klein is",
        },
        correct="right",
        why={
            "small": "De kleine bomen zijn juist wat het laat werken: één boom is te eenvoudig, veel achter elkaar niet. Wat valt er voor boom 500 nog te leren? Zie het beeld.",
            "rate": "Een kleinere stap vraagt alleen meer bomen om er te komen, en een grotere minder. Wat blijft er daarna over voor de bomen? Zie de tekst na het beeld.",
            "right": "In een forest leert elke boom de labels zelf, uit zijn eigen bootstrap sample, en de stemming stemt hun eigenaardigheden weg. Bij boosting bouwt elke boom voort op alle bomen ervoor, dus tellen hun eigenaardigheden juist op. Daarom kies je het aantal bomen op de validation set.",
        },
    )
    choice_too_many
    return (choice_too_many,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Elke boom is de boom uit [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl), met het gemiddelde van zijn voorbeelden in een blad; hij leert de residuen uit [Hoe ver zit één voorspelling ernaast?](/course/open/ml-residuals?lang=nl), en de learning rate is hetzelfde idee als in [Te klein, te groot: de learning rate](/course/open/ml-learning-rate?lang=nl). Net als in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl) kies je het aantal bomen op de validation set; in *Early stopping* doe je hetzelfde voor een neural network, epoch na epoch. In de volgende les, *Te mooi om waar te zijn: data leakage*, zie je hoe een score op de validation set er veel beter uit kan zien dan het model is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_too_many, result_predict, result_round):
    L.wrap_up(result_round, result_predict, choice_too_many)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
