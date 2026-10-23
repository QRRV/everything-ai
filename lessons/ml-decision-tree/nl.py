# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-decision-tree. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-decision-tree", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Regels schrijven of leren van voorbeelden](/course/open/start-rules-vs-learning?lang=nl) en [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rule():
    quick_rule = L.choice(
        "quick-rule",
        question="De computer leerde de regel \"jas als het kouder is dan T °C\" van twaalf ochtenden. Hoe koos hij T?",
        options={
            "middle": "Hij nam het midden tussen de koudste en de warmste ochtend",
            "right": "Hij probeerde elke T en hield die met de minste fouten op de ochtenden over",
            "understand": "Hij beredeneerde uit wat kou is welke T past",
        },
        correct="right",
        why={
            "middle": "Dan zou hij helemaal niet naar de jassen kijken. Zie het recept in [Regels schrijven of leren van voorbeelden](/course/open/start-rules-vs-learning?lang=nl).",
            "understand": "De computer weet niets van kou; hij ziet alleen getallen en antwoorden. Zie het recept in [Regels schrijven of leren van voorbeelden](/course/open/start-rules-vs-learning?lang=nl).",
            "right": "Probeer elke regel van de soort, tel zijn fouten, houd de minste over. Deze les doet dat steeds opnieuw.",
        },
    )
    quick_rule
    return (quick_rule,)


@app.cell(hide_code=True)
def quick_check_overfit():
    quick_overfit = L.choice(
        "quick-overfit",
        question="Een model heeft elk voorbeeld uit de training set goed en veel voorbeelden uit de test set fout. Wat doet het?",
        options={
            "under": "Het underfit",
            "fine": "Het doet het goed",
            "right": "Het overfit",
        },
        correct="right",
        why={
            "under": "Dan zou het ook veel voorbeelden uit de training set fout hebben. Zie *Het idee* in [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl).",
            "fine": "Dan zou het het op nieuwe voorbeelden ongeveer even goed doen. Zie *Het idee* in [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl).",
            "right": "Perfect op de voorbeelden waarvan het leerde, slecht op nieuwe: het volgt de ruis.",
        },
    )
    quick_overfit
    return (quick_overfit,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Regels schrijven of leren van voorbeelden](/course/open/start-rules-vs-learning?lang=nl) leerde de computer één regel, jas als het kouder is dan T °C: één vraag, twee antwoorden. De pinguïns uit [Project: voorspel iets echts, van begin tot eind](/course/open/ml-project-tabular?lang=nl) zijn er in drie soorten, met elk vier metingen. Eén threshold kan de Gentoos apart zetten, maar laat de Adelies en de Chinstraps bij elkaar. Stel dus in elk deel een tweede vraag, en een derde waar dat nog niet genoeg is.

    **Een voorbeeld uit het dagelijks leven.** Met een vogelgids vind je de naam van een vogel met vragen. Is de snavel gekromd? Ja: is de vogel groter dan een kraai? Elk antwoord leidt naar de volgende vraag, tot er één soort over is. Biologen noemen zo'n lijst met vragen een determinatietabel.
    """)
    return


@app.cell(hide_code=True)
def predict_pure():
    guess_pure = L.guess(
        "pure-or-two",
        question="Twee bomen leren van de snavellengte en het lichaamsgewicht van de 206 trainingspinguïns. De een stelt hooguit twee vragen en heeft er 15 fout; de ander blijft vragen tot hij ze alle 206 goed heeft. Welke heeft volgens jou minder van de 68 pinguïns uit de validation set fout?",
        options={
            "pure": "Die elke trainingspinguïn goed heeft",
            "two": "Die met hooguit twee vragen",
            "same": "Ze hebben er evenveel fout",
        },
        reveal="**Die met hooguit twee vragen:** 7 van de 68 fout, tegen 9 voor de boom die elke trainingspinguïn goed heeft. Die boom stelt tot zeven vragen en eindigt in 23 bladeren, sommige voor één losse pinguïn: hij leerde ook hun eigenaardigheden.",
    )
    guess_pure
    return (guess_pure,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **beslisboom** voorspelt met de ene ja-nee-vraag na de andere, elk met een threshold, zoals een stroomschema. De eerste vraag is bijvoorbeeld "flipperlengte hooguit 207,5 mm?"; ja leidt de ene kant op en nee de andere, en aan elke kant kan een nieuwe vraag volgen. Elke vraag is een **splitsing**: één feature en één threshold die de voorbeelden in twee delen splitsen. Waar geen vraag meer volgt, geeft een **blad** de voorspelling: de klasse die de meeste van zijn voorbeelden uit de training set hebben.

        Een boom leert met de regel uit [Regels schrijven of leren van voorbeelden](/course/open/start-rules-vs-learning?lang=nl), steeds opnieuw, in elk deel: het recept hieronder geeft de stappen. Hij stopt waar een deel zuiver is, met maar één klasse, of waar hij zijn **maximale diepte** bereikt, een hyperparameter: het grootste aantal vragen dat je toestaat op een weg naar beneden. Zonder grens gaat de boom door tot elk deel zuiver is, en dan heeft hij elke trainingspinguïn goed.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **beslisboom** | een model dat voorspelt met de ene ja-nee-vraag na de andere, elk met een threshold, zoals een stroomschema, tot een blad het antwoord geeft | `DecisionTreeClassifier()` |
        | **splitsing** | één vraag van een boom: één feature en één threshold, die de voorbeelden in twee delen splitsen | `x <= t` |
        | **blad** | het eind van een weg naar beneden door de boom, waar geen vraag meer volgt; het zegt de klasse die het vaakst voorkomt onder zijn voorbeelden uit de training set | |
        | **diepte**, **maximale diepte** | het grootste aantal vragen op een weg van boven naar een blad; de maximale diepte is de grens die jij zet | `max_depth=2` |
        | **Gini impurity** | één getal voor hoe gemengd de klassen in een deel zijn: 0 als ze allemaal hetzelfde zijn | `criterion="gini"` |
        | `export_text(tree, feature_names=names)` | de vragen van een boom die geleerd heeft, als tekst: één regel per vraag, met de bladeren erin | |
        """),
        L.watch_out("Een splitsing is hier één vraag van de boom, niet de splitsing van de data in een training set, een validation set en een test set."),
        mo.md(r"""
        Het beeld laat een boom zien op twee van de metingen, de snavellengte en het lichaamsgewicht, van de 206 trainingspinguïns uit het project: een rondje is een Adelie, een driehoekje een Chinstrap en een vierkantje een Gentoo; kleur en vorm betekenen hier de soort. De <span class="c-pred">blauwe lijn</span> is de decision boundary van de boom: aan weerszijden ervan voorspelt de boom een andere soort. Hij bestaat alleen uit horizontale en verticale stukken, omdat elke vraag naar één meting kijkt. De andere grafiek telt de <span class="c-loss">fouten</span> voor elke maximale diepte: op de 206 trainingspinguïns, de doorgetrokken lijn, en op de 68 pinguïns uit de validation set, de gestippelde lijn. Verschuif de slider, of sleep over die grafiek.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_depth_cell():
    def _frame(depth):
        depth = int(depth)
        train, val = (m[C.DEPTHS.index(depth)] for m in C.curve())
        alt = (f"De eerste grafiek: de 206 trainingspinguïns naar snavellengte en lichaamsgewicht, een kleur en vorm per soort. Adelies hebben een korte snavel, Gentoos zijn zwaar, Chinstraps hebben een lange snavel en zijn licht. Met een maximale diepte van {depth} scheidt een blauwe lijn van horizontale en verticale stukken de gebieden waar de boom elke soort voorspelt"
               + ("; er is geen gebied voor de Chinstraps." if C.species_in_picture(depth) == 2 else
                  "; hoe groter de diepte, hoe kleiner de vakjes die hij rond losse pinguïns snijdt.")
               + f" De tweede grafiek: de fouten op de training set dalen van 45 bij diepte 1 naar 0 bij diepte 7; de fouten op de validation set dalen van 16 naar 7 bij diepte 2, 3 en 4, en zijn 9 bij diepte 7. Fout bij diepte {depth}: {train} van de 206 trainingspinguïns en {val} van de 68 pinguïns uit de validation set.")
        said = (f"""Diepte {depth}, {C.leaves(depth)} bladeren: <span class="c-loss ea-num">{train}</span> van de 206 trainingspinguïns fout, """
                f"""<span class="c-loss ea-num">{val}</span> van de 68 pinguïns uit de validation set.""")
        if C.species_in_picture(depth) == 2:
            said += " De boom zegt nooit Chinstrap."
        return (C.picture(depth, {"title": f"Beslisboom, diepte {depth}", "x": "snavellengte (mm)", "y": "lichaamsgewicht (g)",
                                  "every": "Fouten voor elke diepte", "depth": "maximale diepte", "mistakes": "fouten",
                                  "train": "training, van 206", "val": "validation, van 68", "alt": alt}), said)

    explore_depth = L.explore("depth", "maximale diepte", _frame, start=1, values=C.DEPTHS, role="hyper",
                              task="Schuif de diepte van 1 naar 7. Bij welke diepte zijn de minste pinguïns uit de validation set fout, en wat gebeurt er daarna met de delen?")
    explore_depth
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Met diepte 1 stelt de boom één vraag, snavellengte hooguit 43,15 mm, en zegt hij nooit Chinstrap. Met diepte 2 krijgt elk deel een eigen vraag, en zijn 7 pinguïns uit de validation set fout, evenveel als met diepte 3 en 4. Dieper snijdt de boom steeds kleinere vakjes rond losse pinguïns: bij diepte 7 heeft hij elke trainingspinguïn goed, maar 9 pinguïns uit de validation set fout. Net als bij de graad van een curve in [Overfitting en underfitting](/course/open/ml-overfitting?lang=nl) underfit de boom met te weinig vragen en overfit hij met te veel. Op de validation set kies je diepte 2, de eerste met de minste fouten.

    ### Recept

    Zo laat je een beslisboom groeien:

    1. Verdeel de voorbeelden voor elke feature en elke threshold in twee delen: `x <= t` en `x > t`. Elk deel zegt zijn meest voorkomende klasse; tel de fouten van de twee delen samen.
    2. Houd de splitsing met de minste fouten over.
    3. Doe stap 1 en 2 opnieuw in elk deel, tot een deel zuiver is of de boom zijn maximale diepte heeft bereikt. Elk blad zegt de klasse die het vaakst voorkomt onder zijn voorbeelden.
    4. Om te voorspellen volg je de vragen van boven naar beneden tot een blad.
    5. Kies de maximale diepte op de validation set.

    scikit-learn doet stap 1 tot en met 4: `DecisionTreeClassifier(max_depth=...)`, met `fit` en `predict`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De pinguïns uit het project, geschud met dezelfde seed: 206 om van te leren en 68 om mee te kiezen. Eerst stap 1 en 2 met de hand, voor de eerste vraag, op alle vier de metingen; `np.unique(x)[:-1]` geeft elke gemeten waarde één keer, van klein naar groot, behalve de grootste, die het tweede deel leeg zou laten. Daarna laat scikit-learn de boom groeien tot diepte 2. Het commentaar met `#` zegt welke stap van het recept elk stuk is. Voer de cel uit.
    """)
    return


@app.cell
def example_tree():
    import numpy as np
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier, export_text

    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["body_mass_g"].isna() == False]
    _order = np.random.default_rng(0).permutation(len(_df))
    _train = _df.iloc[_order[:206]]
    _val = _df.iloc[_order[206:274]]
    _features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    _y = np.array(_train["species"])
    # 1. één splitsing, snavellengte hooguit 45 mm: elk deel zegt zijn meest voorkomende soort
    _x = np.array(_train["bill_length_mm"])
    for _part in [_y[_x <= 45], _y[_x > 45]]:
        _names, _counts = np.unique(_part, return_counts=True)
        print(_names, _counts, len(_part) - max(_counts))
    # 1. en 2. elke feature en elke threshold: de splitsing met de minste fouten
    for _feature in _features:
        _x = np.array(_train[_feature])
        _best, _fewest = None, len(_y)
        for _t in np.unique(_x)[:-1]:
            _wrong = 0
            for _part in [_y[_x <= _t], _y[_x > _t]]:
                _names, _counts = np.unique(_part, return_counts=True)
                _wrong = _wrong + len(_part) - max(_counts)
            if _wrong < _fewest:
                _best, _fewest = _t, _wrong
        print(_feature, _best, _fewest)
    # 3. scikit-learn doet hetzelfde in elk deel, tot diepte 2
    _tree = DecisionTreeClassifier(max_depth=2, random_state=0)
    _tree.fit(_train[_features], _train["species"])
    print(export_text(_tree, feature_names=_features, show_weights=True))
    # 4. voorspel de pinguïns uit de validation set, en meet
    print(np.mean(_tree.predict(_val[_features]) == _val["species"]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Snavellengte hooguit 45 mm maakt 17 + 35 = 52 fouten: 4 Chinstraps en 13 Gentoos tussen de Adelies, 2 Adelies en 33 Chinstraps tussen de Gentoos. Flipperlengte hooguit 202 mm is de beste splitsing, met 40: die zet de Gentoos apart. Doe je dat opnieuw in elk deel, dan vraagt het recept links naar de snavellengte en rechts naar de snavelhoogte: samen 9 fouten.

    De boom van scikit-learn vraagt naar dezelfde drie metingen, met andere thresholds: hij legt elke threshold halverwege twee gemeten waarden, en in plaats van de minste fouten zoekt hij de splitsing die de Gini impurity van de delen het sterkst verlaagt: een fijnere score, die veel gelijke standen doorbreekt tussen splitsingen met evenveel fouten. Een flipperlengte van hooguit 202, 205, 206 of 207 mm maakt elk 40 fouten; de Gini impurity kiest 207, dus de threshold is 207,5 mm. De boom maakt dezelfde 9 fouten op de trainingspinguïns.

    Lees de boom van boven af: elke `|` verder naar rechts is één vraag dieper. Met `show_weights=True` telt `weights` in een blad zijn trainingspinguïns, Adelie, Chinstrap en Gentoo; `class` is wat het zegt. Hij heeft 3 van de 68 pinguïns uit de validation set fout: 0,96 goed. `random_state=0` beslist tussen vragen die even goed splitsen, dus de boom komt er elke keer hetzelfde uit.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_split():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De fouten van één splitsing

        Zet stap 1 van het recept in een functie. `split_mistakes(x, y, t)` krijgt één meting `x` van een aantal pinguïns en hun soorten `y`, twee even lange NumPy-arrays, en een threshold `t`, en geeft de fouten terug van de splitsing `x <= t` als elk deel zijn meest voorkomende soort zegt. Vervang de drie `...`, zoals in het uitgewerkte voorbeeld. De check roept `split_mistakes(x, y, 200)` aan met twaalf verzonnen pinguïns, `x` hun flipperlengtes: dat moet 4 geven. Hij probeert ook andere thresholds, altijd tussen de kleinste en de grootste `x`, en andere pinguïns.
        """),
        L.task("Laat `split_mistakes` teruggeven hoeveel pinguïns de splitsing bij `t` fout heeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_split():
    box_split = L.code_box("split-mistakes", lines=9)
    box_split
    return (box_split,)


@app.cell(hide_code=True)
def check_split(box_split):
    result_split = L.check("split-mistakes", box_split.value)
    result_split
    return (result_split,)


@app.cell(hide_code=True)
def hints_split():
    L.hints(
        "split-mistakes",
        [
            "Stap 1 van het recept, zoals in de eerste loop van het uitgewerkte voorbeeld, met `t` in plaats van 45.",
            "Welk masker houdt de soorten over van de pinguïns waarvan `x` hooguit `t` is? En hoeveel pinguïns van een deel horen niet bij zijn meest voorkomende soort?",
            "`for part in [y[x <= t], y[___]]:` en `wrong = wrong + len(part) - ___`.",
        ],
        why="Bij 200 mm heeft het eerste deel 4 Adelies en 2 Chinstraps, het tweede 4 Gentoos en 2 Chinstraps: `len(part) - max(counts)` is voor elk 2, samen 4.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_depth():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Kies de diepte

        Het vak leest de pinguïns en verdeelt ze zoals in het uitgewerkte voorbeeld, en maakt een boom voor elke maximale diepte van 1 tot 8. Vervang de eerste `...` door regels die de boom laten leren van `train`, op de vier metingen, en zet in `wrong` hoeveel pinguïns van `val` hij fout heeft. Zet daarna de diepte met de minste fouten in `best`. De laatste regel is `best`, zodat de check hem ziet. De check probeert ook andere dieptes.
        """),
        L.task("Zet de maximale diepte met de minste fouten op de validation set in `best`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_depth():
    box_depth = L.code_box("choose-depth", lines=19)
    box_depth
    return (box_depth,)


@app.cell(hide_code=True)
def check_depth(box_depth):
    result_depth = L.check("choose-depth", box_depth.value)
    result_depth
    return (result_depth,)


@app.cell(hide_code=True)
def hints_depth():
    L.hints(
        "choose-depth",
        [
            "Stap 3 en 4 van het uitgewerkte voorbeeld, met `depth` in plaats van 2; daarna stap 5 van het recept, met `np.argmin` zoals in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl).",
            "`tree.fit` krijgt de metingen en de soorten van `train`. Welke vergelijking van `tree.predict(val[features])` met `val[\"species\"]` is `True` bij een fout, en hoe tel je de `True`'s?",
            "`tree.fit(train[features], ___)`, `wrong = np.sum(tree.predict(val[features]) != ___)`, en `best = depths[___]`.",
        ],
        why="De fouten voor de acht dieptes zijn 13, 3, 3, 2, 4, 4, 4 en 4. De minste staan op positie 3, en `depths[3]` is 4. Vanaf diepte 6 is elk deel van de boom zuiver, dus groeit hij niet verder.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_read():
    mo.md(r"""
    ### Oefening 3. Lees de boom
    """)
    return


@app.cell(hide_code=True)
def choice_read_cell():
    choice_read = L.choice(
        "read-tree",
        question="De boom met diepte 2 uit het uitgewerkte voorbeeld krijgt een pinguïn met een flipperlengte van 215 mm en een snavelhoogte van 18,5 mm. Wat zegt hij?",
        options={
            "gentoo": "Gentoo: zijn flipper is langer dan 207,5 mm",
            "right": "Adelie, uit een blad met maar 4 trainingspinguïns",
            "chinstrap": "Chinstrap: zijn snavel is hoog",
        },
        correct="right",
        why={
            "gentoo": "Dat is alleen de eerste vraag. Een pinguïn die die kant op gaat, krijgt een tweede: welke? Kijk naar de regels één `|` verder naar rechts, onder `flipper_length_mm >  207.50`.",
            "chinstrap": "Volg de vragen van boven af: eerst de flipperlengte, dan aan die kant de snavelhoogte. In welk blad kom je uit, en wat is zijn `class`?",
            "right": "215 mm is meer dan 207,5 en 18,5 mm meer dan 17,65, dus eindigt hij in het laatste blad: weights 2, 2 en 0, twee Adelies en twee Chinstraps, en bij gelijke stand wint de eerste soort. Een blad van vier pinguïns zegt weinig: met andere trainingspinguïns kon het net zo goed Chinstrap zeggen.",
        },
    )
    choice_read
    return (choice_read,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De regel uit [Regels schrijven of leren van voorbeelden](/course/open/start-rules-vs-learning?lang=nl) is een boom met diepte 1; een boom vindt zo'n regel opnieuw in elk deel. Zijn maximale diepte is een hyperparameter die je op de validation set kiest, net als de graad in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl) en k in [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl); anders dan bij k-NN hoef je voor een boom niet te standaardiseren, omdat elke vraag naar één feature kijkt. scikit-learn laat hem groeien met dezelfde `fit` en `predict` als in [Project: voorspel iets echts, van begin tot eind](/course/open/ml-project-tabular?lang=nl). Een diepe boom volgt losse pinguïns, dus met andere trainingspinguïns kan hij er heel anders uitzien. In de volgende les, *Veel bomen stemmen: het random forest*, laat je veel van zulke bomen stemmen, en de stemming varieert veel minder dan één boom.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_read, result_depth, result_split):
    L.wrap_up(result_split, result_depth, choice_read)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
