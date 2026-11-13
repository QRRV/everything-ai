# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-nearest-centroid. Route: course/route/lessen/klassiek-ml.yaml. Checks en beelden: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-nearest-centroid", "nl")
    C = L.checks   # de pinguïns en k-NN uit ml-knn, de nearest centroid van de course, en de beelden


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl) en [Groepen zonder labels: k-means](/course/open/ml-kmeans?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_knn():
    quick_knn = L.choice(
        "quick-knn",
        question="k-NN heeft 206 trainingspinguïns, en je vraagt het naar één nieuwe pinguïn. Wat rekent het uit?",
        options={
            "right": "De afstand van de nieuwe pinguïn tot elk van de 206 trainingspinguïns",
            "means": "De afstand van de nieuwe pinguïn tot het gemiddelde van elke soort",
            "formula": "Niet veel: het heeft vooraf een formule geleerd uit de trainingspinguïns",
        },
        correct="right",
        why={
            "means": "Dat doet het model van vandaag. k-NN bewaart elke trainingspinguïn, en zoekt de k dichtstbijzijnde tussen ze allemaal. Zie *Het idee* in [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).",
            "formula": "k-NN leert geen parameters: het bewaart de trainingspinguïns zoals ze zijn, en doet zijn werk als je het iets vraagt. Zie *Het idee* in [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).",
            "right": "Alle 206 afstanden, dan de k dichtstbijzijnde, dan hun stemming: voor elke pinguïn waar je naar vraagt. Vandaag een model dat zijn werk doet voordat je iets vraagt.",
        },
    )
    quick_knn
    return (quick_knn,)


@app.cell(hide_code=True)
def quick_check_centroid():
    quick_centroid = L.choice(
        "quick-centroid",
        question="Wat is bij k-means de centroid van een cluster?",
        options={
            "middle": "Het voorbeeld dat het meest in het midden van het cluster ligt",
            "right": "Het gemiddelde van de voorbeelden in het cluster, één gemiddelde voor elke feature",
            "first": "Het voorbeeld waar het cluster mee begon",
        },
        correct="right",
        why={
            "middle": "Een centroid hoeft geen van de voorbeelden te zijn: het is hun gemiddelde, één getal voor elke feature, en meestal ligt er precies daar geen voorbeeld. Zie *Het idee* in [Groepen zonder labels: k-means](/course/open/ml-kmeans?lang=nl).",
            "first": "k-means begint bij een paar voorbeelden, maar elke keer dat de centroids verschuiven, gaat elke centroid naar het gemiddelde van zijn voorbeelden. Zie *Het idee* in [Groepen zonder labels: k-means](/course/open/ml-kmeans?lang=nl).",
            "right": "Het gemiddelde van zijn voorbeelden, feature voor feature. Vandaag krijgt elke soort zo'n gemiddelde, en omdat de soorten al bekend zijn, is één stap genoeg.",
        },
    )
    quick_centroid
    return (quick_centroid,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    k-NN bewaart elk voorbeeld uit de training set en doet zijn werk als je iets vraagt: voor elke nieuwe pinguïn 206 afstanden. Met een miljoen voorbeelden in de training set is dat een miljoen afstanden voor elke vraag, en alle miljoen voorbeelden moeten in het geheugen blijven. Veel modellen werken andersom: ze doen het werk één keer, voor er iets gevraagd wordt, en bewaren alleen wat ze geleerd hebben. De eenvoudigste daarvan is de nearest centroid: één gemiddelde voor elke klasse, vooraf uitgerekend. Hij antwoordt meteen en bewaart bijna niets. Wat hij daarvoor opgeeft, is waar deze les over gaat.

    **Een voorbeeld uit het dagelijks leven.** Om te zien of een munt in je hand een euro of een pond is, vergelijk je hem niet met elke munt die je ooit hebt vastgehouden. Je weet hoe een gewone euro en een gewoon pond eruitzien, en je kijkt op welke hij het meest lijkt.
    """)
    return


@app.cell(hide_code=True)
def predict_mistakes():
    guess_mistakes = L.guess(
        "mistakes",
        question="De 206 trainingspinguïns en de 68 pinguïns uit de validation set van de les over k-NN, naar snavellengte en lichaamsgewicht. k-NN met k = 15 had er 2 van de 68 fout. De nearest centroid geeft elke pinguïn de soort waarvan het gemiddelde het dichtstbij ligt. Hoeveel heeft hij er fout?",
        options={
            "fewer": "0 of 1: de gemiddelden van hele soorten zijn stabieler dan 15 buren",
            "right": "7",
            "many": "Ongeveer 30",
        },
        reveal="**7**, tegen 2 voor k-NN met k = 15, en 6 met k = 1. Vier daarvan zijn lichte Gentoos, aan de verkeerde kant van een rechte lijn. Met snavellengte en flipperlengte, die de soorten beter uit elkaar houden, heeft hij er 2 fout, net als k-NN op zijn best.",
    )
    guess_mistakes
    return (guess_mistakes,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Lui en ijverig.** k-NN doet niets als het zijn training set krijgt: het bewaart hem, en doet zijn werk als je iets vraagt, dus bij elke vraag groeit het werk mee met de training set. Dat heet **lazy learning**. **Eager learning** doet het werk voor er iets gevraagd wordt: het maakt van de training set een model, en daarna gaat voorspellen snel. De meeste modellen in deze course zijn eager: lineaire regressie bewaart een gewicht voor elke feature en een bias, en geen enkel voorbeeld uit de training set.

        **De nearest centroid.** De **nearest centroid classifier** is eager. Hij leert één ding: voor elke klasse het gemiddelde van haar voorbeelden uit de training set, haar centroid, zoals bij k-means. Hier zijn de klassen bekend, dus hoeft er niets herhaald te worden: één gemiddelde per klasse, en klaar. Om een nieuw voorbeeld te classificeren, rekent hij de afstand tot elke centroid uit en geeft hij de klasse van de dichtstbijzijnde terug. Voor de pinguïns bewaart hij 3 centroids van 2 getallen: 6 getallen, tegen de 412 getallen en 206 soorten die k-NN bewaart. En hij rekent per vraag 3 afstanden uit, in plaats van 206.

        **Rechte grenzen.** De punten die even ver van twee centroids liggen, liggen op een rechte lijn: de middelloodlijn, de lijn die loodrecht staat op de lijn tussen de twee centroids, door het punt halverwege. Tussen twee klassen is de grens van de nearest centroid dus altijd recht, en bij drie klassen komen drie rechte stukken samen in één punt. De grens van k-NN kan om de pinguïns heen buigen; deze niet. Hier kost dat iets: de grens tussen Chinstraps en Gentoos ligt bij ongeveer 4.400 g, en lichtere Gentoos vallen aan de verkeerde kant. Is elke klasse één ronde klont, dan heb je aan een rechte grens genoeg; ligt een klasse in een gebogen band, of in twee klonten, dan niet.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **lazy learning** | leren dat de voorbeelden uit de training set bewaart en pas werk doet als je iets vraagt, zoals k-NN | |
        | **eager learning** | leren dat van de voorbeelden uit de training set een model maakt voor er iets gevraagd wordt, zoals lineaire regressie | |
        | **nearest centroid classifier** | een classifier die het gemiddelde van elke klasse leert, en een nieuw voorbeeld de klasse van het dichtstbijzijnde gemiddelde geeft | `nc_predict(names, means, x)` |
        """),
        L.watch_out("Net als bij k-NN werkt de afstand alleen op gestandaardiseerde features: zonder standaardiseren beslist het lichaamsgewicht in grammen bijna alleen. Standaardiseer met het gemiddelde en de standaardafwijking van de training set, voordat je de gemiddelden van de klassen neemt."),
        mo.md(r"""
        Het beeld: de 206 trainingspinguïns naar snavellengte en lichaamsgewicht, een rondje voor een Adelie, een driehoekje voor een Chinstrap en een vierkantje voor een Gentoo; kleur en vorm betekenen hier de soort. Links de drie centroids, groot en omcirkeld, en de <span class="c-pred">grenzen</span> van de nearest centroid; rechts de grens van k-NN met k = 15, uit [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).
        """),
    ])
    return


@app.cell(hide_code=True)
def figure_compare():
    mo.Html(C.compare_picture({"nc": "Nearest centroid", "knn": "k-NN, k = 15", "x": "snavellengte (mm)", "y": "lichaamsgewicht (g)",
                               "alt": "Twee grafieken van de 206 trainingspinguïns naar snavellengte en lichaamsgewicht. Links: drie rechte blauwe lijnen komen samen bij ongeveer 44 mm en 4.330 g, en verdelen de grafiek in de gebieden van de drie centroids: Adelie bij 39,1 mm en 3.723 g, Chinstrap bij 48,6 mm en 3.686 g, Gentoo bij 47,6 mm en 5.096 g. Rechts: de grens van k-NN met k = 15 buigt tussen de pinguïns door."}))
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Een nieuwe pinguïn van jezelf: sleep de stip over het vlak, of verplaats hem met de pijltjestoetsen, snavellengte van links naar rechts en lichaamsgewicht van onder naar boven. De grafiek ernaast toont de pinguïn, de drie centroids, en een lijn naar elk; de lijn naar de dichtstbijzijnde is doorgetrokken.
    """)
    return


@app.cell(hide_code=True)
def plane_penguin_cell():
    _names = ["Adelie", "Chinstrap", "Gentoo"]

    def _frame(_b, _m):
        _d = C.distances(_b, _m)
        _nc, _knn = C.nc_says(_b, _m), C.knn_says(_b, _m)
        _same = f"k-NN met k = 15 rekent 206 afstanden uit, en is het eens: {_knn}." if _knn == _nc else f"k-NN met k = 15 rekent 206 afstanden uit, en zegt **{_knn}**."
        _said = (f"Een pinguïn van {_b} mm en {C.grouped(_m, L.lang)} g. Zijn afstand tot elke centroid: "
                 + ", ".join(f"{_n} {L.number(_v, 2)}" for _n, _v in zip(_names, _d))
                 + f". De nearest centroid zegt **{_nc}**. {_same}")
        _alt = (f"De drie centroids en hun rechte grenzen, en een nieuwe pinguïn bij {_b} mm en {_m} g met een lijn naar elke "
                f"centroid; de lijn naar de centroid van de {_nc} is doorgetrokken.")
        return C.plane_picture(_b, _m, {"title": "Jouw pinguïn en de centroids", "x": "snavellengte (mm)",
                                         "y": "lichaamsgewicht (g)", "alt": _alt}), _said

    _plane = L.plane("penguin", "snavellengte (mm)", "gewicht (g)", _frame, C.BILLS, C.MASSES, start=(40, 3600), role="",
                     task="Zoek een pinguïn waarover de nearest centroid en k-NN het oneens zijn. Waar op het vlak gebeurt dat?")
    _plane
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Ze zijn het alleen oneens vlak bij een grens, op dit vlak op drie plekken. Een pinguïn van 44 mm en 4.000 g ligt net iets dichter bij de centroid van de Chinstraps dan bij die van de Adelies, 0,95 tegen 0,98, terwijl de meeste van zijn 15 dichtstbijzijnde pinguïns Adelies zijn. En bij 36 mm en 5.600 g, en bij 38 mm en 5.200 g, zware pinguïns met een korte snavel, zegt de nearest centroid Gentoo en k-NN Adelie. Overal elders zeggen ze hetzelfde; alleen rekende k-NN er 206 afstanden voor uit, en de nearest centroid 3.

    ### Recept

    Zo classificeer je met de nearest centroid:

    1. Standaardiseer de features met het gemiddelde en de standaardafwijking van de training set, ook het nieuwe voorbeeld.
    2. Leer, één keer: voor elke klasse het gemiddelde van haar voorbeelden uit de training set, `X_train[y_train == name].mean(axis=0)`.
    3. Reken voor een nieuw voorbeeld de afstand tot elke centroid uit: `np.linalg.norm(means - x, axis=1)`.
    4. Geef de klasse van de dichtstbijzijnde terug: `names[np.argmin(dists)]`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De pinguïns uit de les over k-NN, op dezelfde manier gesplitst en gestandaardiseerd. De cel leert de drie centroids, vraagt naar de tweede pinguïn uit de validation set, die k = 1 in de les over k-NN fout had, telt de fouten op alle 68, en vergelijkt wat de twee modellen bewaren. Voer de cel uit.
    """)
    return


@app.cell
def example_centroid():
    import numpy as np
    import pandas as pd

    # 1. de pinguïns met alle metingen, geschud en gesplitst, gestandaardiseerd met de training set
    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["body_mass_g"].isna() == False]
    _X = np.array(_df[["bill_length_mm", "body_mass_g"]])
    _y = np.array(_df["species"])
    _order = np.random.default_rng(27).permutation(len(_df))
    _train, _val = _order[:206], _order[206:274]
    _mean, _std = _X[_train].mean(axis=0), _X[_train].std(axis=0)
    _X_train, _y_train = (_X[_train] - _mean) / _std, _y[_train]
    _X_val, _y_val = (_X[_val] - _mean) / _std, _y[_val]

    # 2. leren: het gemiddelde van elke soort
    _names = np.unique(_y_train)
    _means = []
    for _name in _names:
        _means.append(_X_train[_y_train == _name].mean(axis=0))
    _means = np.array(_means)
    print(_names)
    print(_means)

    # 3. en 4. de tweede pinguïn uit de validation set: zijn afstand tot elke centroid, en de dichtstbijzijnde
    _dists = np.linalg.norm(_means - _X_val[1], axis=1)
    print(_X[_val[1]], _y_val[1])
    print(_dists, _names[np.argmin(_dists)])

    # de fouten op alle 68 pinguïns uit de validation set, en wat de twee modellen bewaren
    _wrong = 0
    for _i in range(len(_X_val)):
        if _names[np.argmin(np.linalg.norm(_means - _X_val[_i], axis=1))] != _y_val[_i]:
            _wrong = _wrong + 1
    print(_wrong)
    print(_X_train.shape, _means.shape)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De drie centroids, gestandaardiseerd: de Adelies hebben een korte snavel en zijn licht, de Chinstraps hebben een lange snavel en zijn licht, en de Gentoos zijn zwaar. De tweede pinguïn uit de validation set, 43,5 mm en 3.400 g, is een Chinstrap, maar zijn afstand tot de centroid van de Adelies, 0,915, is net iets kleiner dan die tot de centroid van de Chinstraps, 1,017: de nearest centroid zegt Adelie. k-NN met k = 5 had hem goed, dankzij de Chinstraps onder zijn buren. Op alle 68 heeft de nearest centroid er 7 fout. Hij bewaart 3 × 2 = 6 getallen, waar k-NN er 206 × 2 = 412 bewaart en de soort van alle 206 pinguïns.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_centroids():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Leer de centroids

        `centroids(X_train, y_train)` moet de namen van de klassen en hun centroids teruggeven, één rij per klasse, zoals stap 2 van het recept. Vervang de `...` in de loop door één regel. De check roept `centroids(X_train, y_train)` aan met de tien verzonnen pinguïns uit de les over k-NN, al gestandaardiseerd, en probeert ook een set met drie metingen en een set met twee soorten.
        """),
        L.task("Zet het gemiddelde van de voorbeelden uit de training set van elke klasse, één gemiddelde voor elke feature, achteraan in `means`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_centroids():
    box_centroids = L.code_box("centroids", lines=9)
    box_centroids
    return (box_centroids,)


@app.cell(hide_code=True)
def check_centroids(box_centroids):
    result_centroids = L.check("centroids", box_centroids.value)
    result_centroids
    return (result_centroids,)


@app.cell(hide_code=True)
def hints_centroids():
    L.hints(
        "centroids",
        [
            "Stap 2 van het recept, zoals in het uitgewerkte voorbeeld. Welke rijen van `X_train` horen bij de klasse `name`?",
            "Vergelijk `y_train` met `name` voor een mask dat die rijen kiest. Neem dan het gemiddelde langs de rijen, één voor elke feature: welke `axis` is dat?",
            "`means.append(X_train[___].mean(axis=___))`",
        ],
        why="`X_train[y_train == name]` houdt de rijen van één klasse over; hun gemiddelde langs de rijen is haar centroid. Voor de vier verzonnen Adelies is dat -0,475 en -0,75.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_predict():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De dichtstbijzijnde centroid

        `nc_predict(names, means, x)` krijgt de namen en de centroids die `centroids` teruggeeft, en één nieuw voorbeeld `x`, en geeft de klasse van de dichtstbijzijnde centroid terug, zoals stap 3 en 4 van het recept. Vervang de twee `...`. De check roept `nc_predict(names, means, np.array([0.2, -0.9]))` aan met de centroids van de tien verzonnen pinguïns: dat moet `"Chinstrap"` geven. Hij probeert ook vijf andere punten.
        """),
        L.task("Laat `nc_predict` de klasse teruggeven waarvan de centroid het dichtst bij `x` ligt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_predict():
    box_predict = L.code_box("nc-predict", lines=6)
    box_predict
    return (box_predict,)


@app.cell(hide_code=True)
def check_predict(box_predict):
    result_predict = L.check("nc-predict", box_predict.value)
    result_predict
    return (result_predict,)


@app.cell(hide_code=True)
def hints_predict():
    L.hints(
        "nc-predict",
        [
            "Stap 3 en 4 van het recept: één afstand voor elke centroid, dan de klasse van de kleinste.",
            "`means - x` trekt `x` in één keer af van elke rij. `np.linalg.norm` met een `axis` geeft één lengte voor elke rij; welke functie geeft de positie van de kleinste? Zoek die positie op in `names`.",
            "`dists = np.linalg.norm(means - x, axis=___)` en `return names[np.___(dists)]`.",
        ],
        why="De afstanden van `np.array([0.2, -0.9])` tot de drie centroids zijn 0,691, 0,559 en 2,037; de kleinste staat op positie 1, en `names[1]` is Chinstrap.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_clumps():
    mo.md(r"""
    ### Oefening 3. Een klasse in twee klonten
    """)
    return


@app.cell(hide_code=True)
def choice_clumps_cell():
    choice_clumps = L.choice(
        "two-clumps",
        question="De voorbeelden van klasse A liggen in twee even grote klonten ver uit elkaar, één links en één rechts; die van klasse B liggen in één klont in het midden, ertussen. Wat doet de nearest centroid?",
        options={
            "two": "Hij geeft klasse A twee centroids, één voor elke klont, en heeft bijna alles goed",
            "one": "De centroid van A ligt in een van haar klonten, dus hij heeft alleen de andere klont fout",
            "right": "De centroid van A ligt tussen haar klonten, dicht bij B, dus hij heeft veel voorbeelden van beide klassen fout",
        },
        correct="right",
        why={
            "two": "Hij leert één gemiddelde voor elke klasse, hoe haar voorbeelden ook liggen. Zie *Het idee*.",
            "one": "Het gemiddelde van twee even grote klonten ligt halverwege, niet in een van de twee. Zie de instapcheck over de centroid.",
            "right": "Beide centroids liggen in het midden, dicht bij elkaar, dus de rechte grens ertussen snijdt door B: een hele klont van A en ongeveer de helft van B vallen aan de verkeerde kant. k-NN vraagt het de buren en heeft hier geen moeite mee: dat is wat het bewaren van elk voorbeeld uit de training set oplevert.",
        },
    )
    choice_clumps
    return (choice_clumps,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De centroid is die uit [Groepen zonder labels: k-means](/course/open/ml-kmeans?lang=nl), nu met de klassen bekend, dus is één stap genoeg. De afstand en het standaardiseren komen uit [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl), maar k-NN is lazy en bewaart elk voorbeeld, en de nearest centroid is eager en bewaart één gemiddelde per klasse. Daardoor is hij snel en klein, met rechte grenzen; k-NN betaalt met werk bij elke vraag voor grenzen die kunnen buigen. De decision boundary uit [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl) is ook recht, maar logistische regressie vindt haar lijn met gradient descent op de loss; de nearest centroid legt hem gewoon halverwege twee gemiddelden. In de volgende les, *De breedste straat: de maximum-margin classifier*, wordt een rechte grens op een andere manier gekozen: zo ver mogelijk van de dichtstbijzijnde voorbeelden van beide klassen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_clumps, result_centroids, result_predict):
    L.wrap_up(result_centroids, result_predict, choice_clumps)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
