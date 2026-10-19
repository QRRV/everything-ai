# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-project-tabular. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-project-tabular", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Drie korte vragen over [De workflow van machine learning](/course/open/ml-workflow?lang=nl), [Hetzelfde model met scikit-learn](/course/open/ml-sklearn?lang=nl) en [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_step():
    quick_step = L.choice(
        "quick-step",
        question="Een team besluit dat een model van minstens 95 van de 100 pinguïns de soort goed moet hebben. Welke stap van de workflow is dat?",
        options={
            "data": "3\\. Data",
            "right": "2\\. Metric",
            "model": "5\\. Model",
        },
        correct="right",
        why={
            "data": "De stap data gaat over waar de voorbeelden vandaan komen en hoe je ze splitst. Welke stap zegt hoe goed goed genoeg is? Zie *Het idee* in [De workflow van machine learning](/course/open/ml-workflow?lang=nl).",
            "model": "De stap model traint en kiest; het doel waarmee het beoordeeld wordt, is daarvoor afgesproken. Zie *Het idee* in [De workflow van machine learning](/course/open/ml-workflow?lang=nl).",
            "right": "Een metric, accuracy, en een doel ervoor, 0,95: afgesproken vóór elk trainen.",
        },
    )
    quick_step
    return (quick_step,)


@app.cell(hide_code=True)
def quick_check_fit():
    quick_fit = L.choice(
        "quick-fit",
        question="`model` is een model van scikit-learn, `X` bevat de features en `y` de labels. Met welke regels leert het, en voorspelt het daarna voor de rijen van `X_new`?",
        options={
            "swap": "`model.predict(X, y)`, dan `model.fit(X_new)`",
            "one": "`model.fit(X)`, dan `model.predict(X_new, y)`",
            "right": "`model.fit(X, y)`, dan `model.predict(X_new)`",
        },
        correct="right",
        why={
            "swap": "Welke van de twee leert, en welke vraagt alleen? Zie het recept in [Hetzelfde model met scikit-learn](/course/open/ml-sklearn?lang=nl).",
            "one": "Om te leren heeft een model de features en de labels nodig; om te voorspellen alleen de features. Zie het recept in [Hetzelfde model met scikit-learn](/course/open/ml-sklearn?lang=nl).",
            "right": "`fit` leert van de features en de labels; `predict` geeft één voorspelling per rij nieuwe features.",
        },
    )
    quick_fit
    return (quick_fit,)


@app.cell(hide_code=True)
def quick_check_scale():
    quick_scale = L.choice(
        "quick-scale",
        question="Een feature heeft in de training set een gemiddelde van 4200 en een standaardafwijking van 800. Wat is de gestandaardiseerde waarde van 5000?",
        options={
            "right": "1",
            "raw": "800",
            "ratio": "1,19",
        },
        correct="right",
        why={
            "raw": "Dat is 5000 min het gemiddelde. Waardoor deel je dat? Zie het recept in [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl).",
            "ratio": "Dat is 5000 gedeeld door het gemiddelde. Standaardiseren haalt eerst het gemiddelde eraf. Zie het recept in [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl).",
            "right": "$(5000 - 4200) / 800 = 1$: één standaardafwijking boven het gemiddelde.",
        },
    )
    quick_scale
    return (quick_scale,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je hebt de onderdelen een voor een geleerd: de workflow, een baseline, scikit-learn, standaardiseren, een validation set en een test set. Vandaag gebruik je ze samen op een echte tabel, van de vraag tot een getal dat je kunt melden, en schrijf je op wat je vond. Er is niets nieuws; nieuw is dat je alles in één keer doet, in de goede volgorde.

    **Een voorbeeld uit het dagelijks leven.** In rijlessen oefen je sturen, remmen en parkeren een voor een. Het examen is één rit door echt verkeer.
    """)
    return


@app.cell(hide_code=True)
def predict_right():
    guess_right = L.guess(
        "how-many",
        question="Logistische regressie krijgt vier metingen van elke pinguïn: de lengte en de hoogte van de snavel, de lengte van de vleugel en het lichaamsgewicht. Bij hoeveel van 68 nieuwe pinguïns kiest het volgens jou de goede soort?",
        options={
            "thirty": "Ongeveer 30: zoveel als altijd de meest voorkomende soort gokken",
            "fifty": "Ongeveer 50",
            "most": "Ongeveer 66",
            "all": "Alle 68",
        },
        reveal="**Ongeveer 66.** Op de validation set heeft het er 66 van de 68 goed, een accuracy van 0,97, terwijl altijd de meest voorkomende soort gokken, Adelie, er 30 haalt. De vier metingen zeggen veel over de soort. De twee die het mist, zijn allebei Chinstraps met een korte snavel, die op Adelies lijken.",
    )
    guess_right
    return (guess_right,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Er is vandaag geen nieuw idee: je doorloopt [De workflow van machine learning](/course/open/ml-workflow?lang=nl) op een echte tabel. De **Palmer-pinguïns** zijn 344 pinguïns van drie soorten, Adelie, Chinstrap en Gentoo, van drie eilanden bij Palmer Station op Antarctica. Kristen Gorman en het Palmer Station Long Term Ecological Research-programma maten ze van 2007 tot 2009; Allison Horst, Alison Hill en Kristen Gorman zetten ze in één tabel. De licentie is CC0: je mag er alles mee. Je zag de datasheet al in [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl). De course heeft de tabel als `data/penguins.csv`: één rij per pinguïn, met de soort, het eiland, vier metingen, het geslacht en het jaar.

        De zes stappen, voor deze tabel:

        1. **Vraag.** Welke soort is een pinguïn, op grond van zijn vier metingen? Een classificatie met drie klassen; het label is de kolom `species`. Hier staat het label in de tabel. Staat wat je wilt weten er niet in, dan heb je een vervanger nodig die je wel kunt meten, en dan leert het model die vervanger, niet wat je bedoelde.
        2. **Metric.** Accuracy: elke pinguïn telt even zwaar, en elke fout kost hetzelfde. Het doel: minstens 0,95.
        3. **Data.** Twee pinguïns hebben geen metingen: die gaan eruit. De andere 342 worden geschud met een seed en gesplitst: 206 om op te trainen, 68 om op te kiezen en 68 voor de test.
        4. **Baseline.** `DummyClassifier` uit scikit-learn, de tegenhanger voor classificatie van de `DummyRegressor` die je al kent: altijd de meest voorkomende soort van de training set.
        5. **Model.** `LogisticRegression` uit scikit-learn: dezelfde `fit` en `predict`, en bij drie klassen gebruikt het softmax. Standaard voegt het ook een kleine straf toe die de weights klein houdt. De features worden gestandaardiseerd met het gemiddelde en de standaardafwijking van de training set.
        6. **Nog een ronde.** Kijk naar de pinguïns die het fout doet, en naar elke soort apart. Pas als alles gekozen is: één keer op de test set.
        """),
        L.watch_out("Het gemiddelde en de standaardafwijking om te standaardiseren komen alleen uit de training set, ook voor de validation set en de test set. Die staan voor nieuwe pinguïns, die je nog niet hebt terwijl je traint."),
        mo.Html(C.picture({"title": "De 206 pinguïns van de training set", "x": "snavellengte (mm)", "y": "snavelhoogte (mm)",
                           "alt": "De 206 trainingspinguïns naar snavellengte en snavelhoogte, een kleur en vorm per soort. Adelies, rondjes, hebben korte, hoge snavels; Gentoos, vierkantjes, lange en lage; Chinstraps, driehoekjes, lange en hoge. Adelies en Chinstraps overlappen een beetje, bij een snavellengte van ongeveer 41 tot 46 mm."})),
        mo.md(r"""
        Kleur en vorm betekenen hier een soort. Twee metingen houden de drie soorten al bijna helemaal uit elkaar: de Gentoos hebben lange, lage snavels, de Adelies korte, hoge, en de Chinstraps lange, hoge. Waar de Adelies en Chinstraps overlappen, bij een snavellengte tussen ongeveer 41 en 46 mm, zijn fouten het waarschijnlijkst.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo doorloop je de workflow op een tabel met scikit-learn:

    1. Schrijf de vraag en het label op; spreek de metric en het doel af.
    2. Lees de tabel in, houd de rijen met elke feature, en splits hem met een seed: `df.iloc[order[:206]]` enzovoort.
    3. Meet een baseline op de validation set: `DummyClassifier().fit(X, y)`, dan `predict`.
    4. Standaardiseer met de training set: `(X - mean) / std`, met `mean` en `std` uit `train[features]`.
    5. Train `LogisticRegression()` op de training set, en meet de accuracy op de validation set.
    6. Kijk wat het fout doet, `val[pred != val["species"]]`, en maak nog een ronde. Meet aan het eind één keer op de test set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1 tot en met 5 met twee van de vier metingen: vleugellengte en lichaamsgewicht. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_two():
    import numpy as np
    import pandas as pd
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression

    # 1. welke soort is een pinguïn? Accuracy, met als doel minstens 0.95
    # 2. lees de tabel, houd de pinguïns met een lichaamsgewicht, en splits ze met een seed
    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["body_mass_g"].isna() == False]
    _order = np.random.default_rng(0).permutation(len(_df))
    _train = _df.iloc[_order[:206]]
    _val = _df.iloc[_order[206:274]]
    print(len(_df), len(_train), len(_val))
    _features = ["flipper_length_mm", "body_mass_g"]
    # 3. de baseline: altijd de meest voorkomende soort van de training set
    _baseline = DummyClassifier()
    _baseline.fit(_train[_features], _train["species"])
    print(np.mean(_baseline.predict(_val[_features]) == _val["species"]))
    # 4. standaardiseer met het gemiddelde en de standaardafwijking van de training set
    _mean = _train[_features].mean()
    _std = _train[_features].std()
    # 5. logistische regressie, gemeten op de validation set
    _model = LogisticRegression()
    _model.fit((_train[_features] - _mean) / _std, _train["species"])
    _pred = _model.predict((_val[_features] - _mean) / _std)
    print(np.mean(_pred == _val["species"]))
    return DummyClassifier, LogisticRegression, np, pd


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Er blijven 342 pinguïns over, 206 om op te trainen en 68 om op te kiezen. De baseline zegt altijd Adelie, de meest voorkomende soort van de training set, en heeft 0,44 van de validation set goed: 30 van de 68. Met vleugellengte en lichaamsgewicht haalt logistische regressie 0,84: ver boven de baseline, maar onder het doel van 0,95. Per soort verbergt die 0,84 meer: alle 25 Gentoos en 29 van de 30 Adelies zijn goed, tegen 3 van de 13 Chinstraps. Die twee metingen halen de Gentoos eruit, die groot zijn, maar niet de Adelies van de Chinstraps, die ongeveer even zwaar zijn. Dus: nog een ronde, met de snavel erbij.

    `_df.iloc[_order[:206]]` pakt de rijen op die 206 posities, zoals een array met posities deed in [Werkt het op nieuwe data? Training set en test set](/course/open/ml-train-test-split?lang=nl); `isna() == False` is `True` waar het lichaamsgewicht er wel is. De `.std()` van pandas deelt door één minder dan het aantal waarden, een van de andere tools uit [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl); bij 206 pinguïns maakt dat hier niets uit.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_model():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Alle vier de metingen

        Het vak leest en splitst de tabel zoals in het uitgewerkte voorbeeld, en zet alle vier de metingen in `features`. Vervang de `...` door stap 4 en 5: standaardiseer met het gemiddelde en de standaardafwijking van de training set, laat `LogisticRegression` leren van `train`, voorspel de soort van de pinguïns in `val` in `pred`, en zet de accuracy in `accuracy`. De laatste regel is `accuracy`, zodat de check het ziet. De check probeert ook andere features.
        """),
        L.task("Zet in `accuracy` de accuracy op de validation set van logistische regressie op alle vier de metingen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_model():
    box_model = L.code_box("model", lines=17)
    box_model
    return (box_model,)


@app.cell(hide_code=True)
def check_model(box_model):
    result_model = L.check("model", box_model.value)
    result_model
    return (result_model,)


@app.cell(hide_code=True)
def hints_model():
    L.hints(
        "model",
        [
            "Stap 4 en 5 van het uitgewerkte voorbeeld, met `features` in plaats van de twee metingen.",
            "`mean` en `std` komen alleen uit `train[features]`. Welke sets standaardiseer je ermee vóór `fit` en vóór `predict`?",
            "`mean = train[features].mean()`, `std = ___`, dan `model.fit((train[features] - mean) / std, ___)`, `pred = model.predict(___)` en `accuracy = np.mean(pred == ___)`.",
        ],
        why="Met alle vier de metingen heeft logistische regressie 66 van de 68 validatiepinguïns goed: een accuracy van ongeveer 0,97, boven het doel van 0,95. De snavelmetingen houden de Adelies en de Chinstraps uit elkaar.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mistakes():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Welke pinguïns doet het fout?

        Stap 6: kijk naar de fouten. Het vak traint het model van oefening 1, met zijn voorspellingen voor de validation set in `pred`. Vervang de `...` door één regel: zet in `wrong` de rijen van `val` waarvan de soort verschilt van de voorspelling. De laatste regel is `wrong`, zodat de check hem ziet. De check probeert ook andere features.
        """),
        L.task("Zet in `wrong` de pinguïns van de validation set waarvoor het model de verkeerde soort kiest."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mistakes():
    box_mistakes = L.code_box("mistakes", lines=18)
    box_mistakes
    return (box_mistakes,)


@app.cell(hide_code=True)
def check_mistakes(box_mistakes):
    result_mistakes = L.check("mistakes", box_mistakes.value)
    result_mistakes
    return (result_mistakes,)


@app.cell(hide_code=True)
def hints_mistakes():
    L.hints(
        "mistakes",
        [
            "Een vergelijking van `pred` met de soort geeft `True` of `False` per pinguïn. Hoe koos een masker rijen uit een tabel?",
            "Welke vergelijking is `True` precies waar het model het fout heeft?",
            "`wrong = val[pred ___ val[\"species\"]]`.",
        ],
        why="`pred != val[\"species\"]` is `True` voor de twee pinguïns die het model fout heeft, en `val[...]` houdt die rijen. Het zijn allebei Chinstraps van Dream met een snavel van 42,4 en 43,5 mm: kort voor een Chinstrap, lang voor een Adelie, precies waar de twee soorten elkaar in de figuur raken.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_proxy():
    mo.md(r"""
    ### Oefening 3. Een label dat niet in de tabel staat
    """)
    return


@app.cell(hide_code=True)
def choice_proxy_cell():
    choice_proxy = L.choice(
        "proxy",
        question="De onderzoekers willen ook weten welke pinguïns niet genoeg te eten krijgen. De tabel heeft daar geen kolom voor. Wat kunnen ze als label gebruiken?",
        options={
            "mass": "Het lichaamsgewicht zoals het is: de zwaarste pinguïns eten het best",
            "right": "Of een pinguïn licht is voor zijn soort en geslacht: een vervanger voor niet genoeg eten, die er ook naast kan zitten",
            "none": "Niets: zonder zo'n kolom is geen model mogelijk",
        },
        correct="right",
        why={
            "mass": "Gentoos zijn zwaarder dan Adelies, wat ze ook eten, en mannetjes zwaarder dan vrouwtjes. Wat leert het model dan: eten, of soort? Zie stap 1 in *Het idee*.",
            "none": "Een label dat je niet kunt meten, kun je vaak vervangen door een dat je wel kunt meten. Waar zit het addertje? Zie stap 1 in *Het idee*.",
            "right": "Een label gemaakt uit andere kolommen, vergeleken binnen soort en geslacht, is een vervanger voor wat je echt wilt weten. Een pinguïn kan ook om andere redenen licht zijn, dus schrijf op wat het label echt meet.",
        },
    )
    choice_proxy
    return (choice_proxy,)


@app.cell(hide_code=True)
def text_test():
    mo.md(r"""
    ### Eén keer op de test set

    Het model met vier metingen haalt het doel op de validation set, dus het kiezen is klaar. In de volgorde van het werk komt pas nu de test set erbij, één keer. Voer de cel uit: hij traint hetzelfde model en meet het, en de baseline, op de 68 pinguïns van de test set.
    """)
    return


@app.cell
def example_test(DummyClassifier, LogisticRegression, np, pd):
    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["body_mass_g"].isna() == False]
    _order = np.random.default_rng(0).permutation(len(_df))
    _train = _df.iloc[_order[:206]]
    _test = _df.iloc[_order[274:]]
    _features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    _mean = _train[_features].mean()
    _std = _train[_features].std()
    _model = LogisticRegression()
    _model.fit((_train[_features] - _mean) / _std, _train["species"])
    _pred = _model.predict((_test[_features] - _mean) / _std)
    _baseline = DummyClassifier()
    _baseline.fit(_train[_features], _train["species"])
    print(np.mean(_pred == _test["species"]), np.mean(_baseline.predict(_test[_features]) == _test["species"]))
    print(_test[_pred != _test["species"]][["species", "island", "bill_length_mm", "bill_depth_mm"]])
    return


@app.cell(hide_code=True)
def text_after_test():
    mo.md(r"""
    Op de test set heeft het model er 67 van de 68 goed, een accuracy van 0,99, tegen 0,43 voor de baseline. De enige die het mist, is weer een Chinstrap van Dream. Dat de test set iets hoger uitkomt dan de validation set, is hoogstwaarschijnlijk toeval: bij 68 pinguïns is één pinguïn meer of minder 0,015. Het getal dat je meldt, is 0,99 op de test set, naast de baseline.
    """)
    return


@app.cell(hide_code=True)
def explain_report_cell():
    explain_report = L.explain(
        "report",
        question="Schrijf in drie of vier zinnen voor de onderzoekers op wat je vond: de vraag, de metric en het doel, de baseline, het model en zijn getal op de test set, en wat er misgaat.",
        model_answer="We voorspelden de soort van een pinguïn uit vier metingen, met accuracy als metric en 0,95 als doel. Altijd de meest voorkomende soort gokken, Adelie, haalt 0,43 op de test set; logistische regressie op de gestandaardiseerde metingen haalt 0,99 (0,97 op de validation set). De fouten zijn Chinstraps met een korte snavel, die het voor Adelies houdt. Bij 68 testpinguïns is één pinguïn 0,015, dus deze getallen zijn grof.",
        points=[
            "de vraag en het label: de soort, uit vier metingen",
            "de metric en het doel: accuracy, minstens 0,95",
            "de baseline naast het model, allebei op de test set: ongeveer 0,43 tegen 0,99",
            "wat er misgaat: Chinstraps met korte snavels, aangezien voor Adelies",
        ],
    )
    explain_report
    return (explain_report,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [De workflow van machine learning](/course/open/ml-workflow?lang=nl) gaf de volgorde, [Hetzelfde model met scikit-learn](/course/open/ml-sklearn?lang=nl) de `fit` en `predict`, [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl) het standaardiseren, en [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl) de datasheet van deze pinguïns. Samen brachten ze een echte tabel van een vraag naar een gemeld getal. In de volgende les, *Wie staat er in de data, en wie ontbreekt?*, vraag je wat zo'n getal niet zegt: wie de data weglaat, en wie een model schade kan doen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_proxy, explain_report, result_mistakes, result_model):
    L.wrap_up(result_model, result_mistakes, choice_proxy, explain_report)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
