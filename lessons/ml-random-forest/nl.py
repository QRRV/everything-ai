# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-random-forest. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-random-forest", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl) en [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_pure():
    quick_pure = L.choice(
        "quick-pure",
        question="Een beslisboom zonder maximale diepte heeft alle 206 trainingspinguïns goed. Wat zegt dat over nieuwe pinguïns?",
        options={
            "all": "Die heeft hij ook allemaal goed",
            "right": "Niet veel: hij ging door tot elk deel zuiver was, en kan de eigenaardigheden van losse pinguïns geleerd hebben",
            "under": "Hij underfit",
        },
        correct="right",
        why={
            "all": "Op de pinguïns uit de validation set had die boom er 9 fout, meer dan een boom met diepte 2. Zie de slider in [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl).",
            "under": "Een boom die elk voorbeeld uit de training set goed heeft, is niet te eenvoudig. Zie de slider in [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl).",
            "right": "Dat hij al zijn eigen trainingspinguïns goed heeft, zegt niets over andere: dat zegt alleen de validation set.",
        },
    )
    quick_pure
    return (quick_pure,)


@app.cell(hide_code=True)
def quick_check_variance():
    quick_variance = L.choice(
        "quick-variance",
        question="Twee rijtjes, elke waarde even waarschijnlijk: 3, 3, 4, 3 en 1, 8, 2, 5. Welke heeft de grootste variantie?",
        options={
            "first": "3, 3, 4, 3",
            "right": "1, 8, 2, 5",
            "same": "Allebei even groot: ze hebben elk vier getallen",
        },
        correct="right",
        why={
            "first": "De variantie gaat over hoe ver de waarden van hun gemiddelde liggen, niet over hoe groot ze zijn. Zie *Het idee* in [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
            "same": "De variantie kijkt naar de gekwadrateerde afstand van elke waarde tot het gemiddelde. Zie *Het idee* in [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
            "right": "Zijn waarden liggen ver van hun gemiddelde, 4; die van het eerste liggen binnen 1 van het hunne.",
        },
    )
    quick_variance
    return (quick_variance,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl) zag je dat een boom die doorgroeit tot elk deel zuiver is, losse pinguïns volgt, dus andere trainingspinguïns geven een heel andere boom. Hier gebruiken de bomen snavellengte en snavelhoogte, waarop de soorten verder uit elkaar liggen dan op snavellengte en lichaamsgewicht. Trek willekeurig 206 van de 206 trainingspinguïns, sommige meer dan eens, laat een boom groeien tot elk deel zuiver is, en doe dat 100 keer: van de 68 pinguïns uit de validation set heeft de ene boom er 1 fout, een andere 8, en gemiddeld 5. Welke boom vertrouw je?

    **Een voorbeeld uit het dagelijks leven.** In 1906 bekeek Francis Galton 787 gokken naar het gewicht van een os op een veetentoonstelling. Losse gokken zaten er vaak ver naast, maar de middelste gok, de mediaan, was 1207 pond: 9 pond meer dan de echte 1198. Dat werkt omdat de gokkers zich op verschillende manieren vergisten. Hadden ze allemaal de luidste nagezegd, dan had de menigte niet beter gegokt dan één persoon.
    """)
    return


@app.cell(hide_code=True)
def predict_vote():
    guess_vote = L.guess(
        "hundred-vote",
        question="100 van zulke bomen stemmen: elke pinguïn uit de validation set krijgt de soort die de meeste van hen zeggen. Hoeveel heeft de stemming volgens jou fout?",
        options={
            "average": "Ongeveer 5, zoals een boom gemiddeld",
            "right": "3",
            "worst": "Meer dan 5: de slechte bomen verpesten het",
        },
        reveal="**3**, en niet door toeval: tien van zulke stemmingen van 100 bomen, elk met eigen willekeurige trekkingen, hebben er allemaal precies 3 fout. Een paar losse bomen deden het beter, met 1 of 2 fout, maar welke dat zijn, is toeval; de stemming geeft elke keer 3.",
    )
    guess_vote
    return (guess_vote,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **random forest** laat veel diepe bomen groeien en laat ze stemmen, met majority voting, zoals de buren bij k-NN. Elke boom leert van zijn eigen voorbeelden: een **bootstrap**-steekproef, n voorbeelden die willekeurig uit de n van de training set getrokken zijn, **met teruglegging**. Elk getrokken voorbeeld gaat terug, dus sommige komen twee keer of vaker voor en andere helemaal niet: gemiddeld bevat een bootstrap-steekproef ongeveer 63% van de voorbeelden. Bomen op bootstrap-steekproeven laten stemmen heet **bagging**. Een random forest voegt daar één ding aan toe: bij elke splitsing mag een boom maar naar een paar features kijken, willekeurig getrokken, zodat zelfs de eerste vragen van de bomen verschillen. De bomen van een forest vormen een **ensemble**: modellen waarvan je de voorspellingen combineert.

        Waarom varieert de stemming minder dan één boom? Verschillende bomen vergissen zich in verschillende pinguïns, en een fout die maar een paar bomen maken, wordt weggestemd. Alleen de pinguïns die de meeste bomen fout hebben, blijven fout, en welke dat zijn, verandert nauwelijks van forest tot forest; welke één boom fout heeft, is toeval.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **random forest** | veel diepe beslisbomen op bootstrap-steekproeven, met een paar willekeurige features per splitsing, die stemmen | `RandomForestClassifier()` |
        | **bootstrap**, bootstrap-steekproef | n voorbeelden met teruglegging uit je n trekken; wat je krijgt, is een bootstrap-steekproef | `rng.integers(0, n, size=n)` |
        | **met teruglegging** | elk getrokken voorbeeld gaat terug en kan opnieuw getrokken worden | |
        | **bagging** | modellen op bootstrap-steekproeven die stemmen of gemiddeld worden: bootstrap aggregating | |
        | **ensemble** | modellen waarvan je de voorspellingen combineert, door te stemmen of te middelen | |
        | `max_features=1` | naar hoeveel features, willekeurig getrokken, een boom bij elke splitsing mag kijken | |
        | `n_estimators=100` | hoeveel bomen een random forest van scikit-learn laat groeien | |
        """),
        L.watch_out("*Variantie* is hier de variantie uit [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl): hoe ver de fouten van een forest van hun gemiddelde liggen over veel willekeurige trekkingen."),
        mo.md(r"""
        Het beeld laat een forest zien op de 206 trainingspinguïns: een rondje is een Adelie, een driehoekje een Chinstrap en een vierkantje een Gentoo; kleur en vorm betekenen hier de soort. Elke boom kijkt per splitsing naar één willekeurige meting, en de <span class="c-pred">blauwe lijn</span> is de decision boundary van de stemming van de eerste n bomen. De andere grafiek heeft voor elk aantal bomen de <span class="c-loss">fouten</span> op de 68 pinguïns uit de validation set van tien van zulke forests, elk met eigen willekeurige trekkingen: een stip voor elk, en een lijn van de minste naar de meeste; de gevulde stippen zijn het forest links. Verschuif de slider, of sleep over die grafiek.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_trees_cell():
    def _frame(n):
        n = int(n)
        m = C.MEASURED[n]
        trees = f"{n} {'boom' if n == 1 else 'bomen'}"
        alt = (f"De eerste grafiek: de 206 trainingspinguïns naar snavellengte en snavelhoogte, een kleur en vorm per soort. Adelies hebben korte, hoge snavels, Gentoos lange, lage, Chinstraps lange, hoge. Een blauwe lijn scheidt de gebieden waar de stemming van {trees} elke soort voorspelt"
               + ("; met één boom heeft hij scherpe hoeken rond losse pinguïns." if n == 1 else "; met maar een paar bomen heeft hij nog scherpe hoeken." if n < 5 else "; met meer bomen wordt hij gladder.")
               + f" De tweede grafiek: voor 1 tot 100 bomen de fouten op de validation set van tien forests; met 1 boom lopen ze van 2 tot 8, met 100 bomen maken alle tien er 3. Met {trees}: "
               + (f"van {min(m)} tot {max(m)}." if min(m) < max(m) else f"alle tien {min(m)}."))
        said = (f"""{trees}: het forest links heeft <span class="c-loss ea-num">{C.mistakes(0, n)}</span> van de 68 pinguïns uit de validation set fout; """
                + (f"""de tien forests van <span class="c-loss ea-num">{min(m)}</span> tot <span class="c-loss ea-num">{max(m)}</span>""" if min(m) < max(m)
                   else f"""alle tien forests <span class="c-loss ea-num">{min(m)}</span>""")
                + f""", met een variantie van {L.number(C.variance(m)).rstrip("0").rstrip(".,")}.""")
        return (C.picture(n, {"title": f"Random forest, {trees}", "x": "snavellengte (mm)", "y": "snavelhoogte (mm)",
                              "every": "Tien forests", "trees": "bomen", "mistakes": "fouten (van 68)", "alt": alt}), said)

    explore_trees = L.explore("trees", "bomen", _frame, start=1, values=C.SIZES, role="hyper",
                              task="Schuif het aantal bomen van 1 naar 100. Hoe ver liggen de tien forests uit elkaar met 1 boom, en met 100?")
    explore_trees
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Met 1 boom maken de tien forests 2 tot 8 fouten, een variantie van 4,2, en de grens heeft scherpe hoeken rond losse pinguïns. Met 5 bomen maken ze er 2 tot 4, en met 100 maken alle tien er precies 3, een variantie van 0: dezelfde drie pinguïns, die de meeste losse bomen ook fout hebben. De grens wordt gladder, omdat de hoeken die maar een paar bomen hebben, worden weggestemd.

    ### Recept

    Zo bouw je een random forest van T bomen:

    1. Trek een bootstrap-steekproef: n posities van 0 tot en met n − 1, met teruglegging, `rng.integers(0, n, size=n)`.
    2. Laat een boom op die voorbeelden groeien tot elk deel zuiver is, met een paar willekeurige features per splitsing: `DecisionTreeClassifier(max_features=1)`.
    3. Doe stap 1 en 2 T keer, elke keer met nieuwe willekeurige trekkingen.
    4. Om te voorspellen laat je elke boom voorspellen, en geef je elk voorbeeld de klasse die de meeste bomen zeggen.

    scikit-learn doet het allemaal: `RandomForestClassifier(n_estimators=T, max_features=1)`, met `fit` en `predict`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De pinguïns uit het project, geschud met dezelfde seed: 206 om van te leren en 68 om mee te kiezen. Vijf bomen, stap 1 tot en met 4, en daarna het forest van 100 bomen van scikit-learn. `_votes[:, 54]` is kolom 54 van de tabel met stemmen: wat de vijf bomen zeggen over pinguïn 54 uit de validation set. Het commentaar met `#` zegt welke stap van het recept elk stuk is. Voer de cel uit.
    """)
    return


@app.cell
def example_forest():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["body_mass_g"].isna() == False]
    _order = np.random.default_rng(0).permutation(len(_df))
    _train = _df.iloc[_order[:206]]
    _val = _df.iloc[_order[206:274]]
    _features = ["bill_length_mm", "bill_depth_mm"]
    _X_train, _y_train = np.array(_train[_features]), np.array(_train["species"])
    _X_val, _y_val = np.array(_val[_features]), np.array(_val["species"])
    _rng = np.random.default_rng(0)
    _votes = []
    # 3. vijf bomen, elk met nieuwe willekeurige trekkingen
    for _i in range(5):
        # 1. een bootstrap-steekproef: 206 posities, getrokken met teruglegging
        _rows = _rng.integers(0, 206, size=206)
        # 2. een boom op die pinguïns, met één willekeurige meting per splitsing
        _tree = DecisionTreeClassifier(max_features=1, random_state=_i)
        _tree.fit(_X_train[_rows], _y_train[_rows])
        _votes.append(_tree.predict(_X_val))
        print(len(np.unique(_rows)), np.sum(_votes[_i] != _y_val))
    # 4. voor elke pinguïn uit de validation set de soort die de meeste van de vijf bomen zeggen
    _votes = np.array(_votes)
    _said = []
    for _j in range(len(_y_val)):
        _names, _counts = np.unique(_votes[:, _j], return_counts=True)
        _said.append(_names[np.argmax(_counts)])
    print(_votes[:, 54], _said[54], _y_val[54])
    print(np.sum(np.array(_said) != _y_val))
    # scikit-learn: stap 1 tot en met 4 voor 100 bomen
    _forest = RandomForestClassifier(n_estimators=100, max_features=1, random_state=0)
    _forest.fit(_X_train, _y_train)
    print(np.sum(_forest.predict(_X_val) != _y_val))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De vijf bootstrap-steekproeven bevatten 125 tot 134 verschillende pinguïns van de 206, en de vijf bomen hebben 3 tot 8 van de 68 pinguïns uit de validation set fout. Over pinguïn 54, een Chinstrap, zeggen twee bomen Gentoo en drie Chinstrap: de stemming klopt. Samen hebben de vijf bomen er 3 fout, zo weinig als de beste van hen. Het forest van 100 bomen van scikit-learn heeft er ook 3 fout. Het telt geen stemmen, maar middelt welk deel van elk blad elke soort is; met één soort in elk blad, zoals in een boom die doorgroeide tot elk deel zuiver is, is dat dezelfde stemming.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bootstrap():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een bootstrap-steekproef

        Stap 1 van het recept. `rng` is een random number generator met seed 1, en `n` is het aantal trainingspinguïns, 206. Vervang de `...`: zet in `rows` de posities van een bootstrap-steekproef, `n` hele getallen van 0 tot en met `n - 1`, getrokken met teruglegging. De check kijkt naar wat een bootstrap-steekproef is, niet naar de getallen zelf, en probeert ook een andere `n`.
        """),
        L.task("Zet in `rows` de posities van een bootstrap-steekproef van `n` pinguïns."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bootstrap():
    box_bootstrap = L.code_box("bootstrap", lines=6)
    box_bootstrap
    return (box_bootstrap,)


@app.cell(hide_code=True)
def check_bootstrap(box_bootstrap):
    result_bootstrap = L.check("bootstrap", box_bootstrap.value)
    result_bootstrap
    return (result_bootstrap,)


@app.cell(hide_code=True)
def hints_bootstrap():
    L.hints(
        "bootstrap",
        [
            "Stap 1 van het recept, zoals in het uitgewerkte voorbeeld, met `n` in plaats van 206.",
            "`rng.integers(low, high, size=...)` geeft hele getallen vanaf `low` tot `high`; `high` zelf doet niet mee. Welke `low` en `high` geven de posities van 0 tot en met `n - 1`, en hoeveel heb je er nodig?",
            "`rows = rng.integers(0, ___, size=___)`.",
        ],
        why="`rng.integers(0, n, size=n)` trekt `n` posities, elk van 0 tot en met `n - 1`, en elke trekking kan elke positie opnieuw geven: met teruglegging. `rng.permutation(n)` zou elke positie precies één keer geven: dezelfde pinguïns in een andere volgorde.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_vote():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De stemming

        Stap 4 van het recept in een functie. `vote(votes)` krijgt een tabel met soorten, één rij per boom en één kolom per pinguïn, en geeft voor elke pinguïn de soort terug die de meeste bomen zeggen, als een array. Vervang de twee `...`. De check roept `vote(votes)` aan met vijf bomen en vier verzonnen pinguïns: dat moet `["Adelie", "Gentoo", "Chinstrap", "Adelie"]` geven. Hij probeert ook andere tabellen.
        """),
        L.task("Laat `vote` voor elke pinguïn de soort teruggeven die de meeste bomen zeggen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_vote():
    box_vote = L.code_box("vote", lines=9)
    box_vote
    return (box_vote,)


@app.cell(hide_code=True)
def check_vote(box_vote):
    result_vote = L.check("vote", box_vote.value)
    result_vote
    return (result_vote,)


@app.cell(hide_code=True)
def hints_vote():
    L.hints(
        "vote",
        [
            "Stap 4 van het uitgewerkte voorbeeld, met `votes` en `j`.",
            "De stemmen voor één pinguïn zijn één kolom van de tabel: hoe neem je kolom `j`? En op welke positie in `counts` staan na `np.unique` de meeste stemmen?",
            "`np.unique(votes[:, ___], return_counts=True)` en `said.append(names[___])`.",
        ],
        why="Voor de eerste pinguïn zeggen de vijf bomen Adelie, Chinstrap, Adelie, Adelie en Gentoo: `names` is `['Adelie' 'Chinstrap' 'Gentoo']`, `counts` is `[3 1 1]`, `np.argmax(counts)` is 0, en `names[0]` is Adelie.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_features():
    mo.md(r"""
    ### Oefening 3. Waarom willekeurige features?
    """)
    return


@app.cell(hide_code=True)
def choice_features_cell():
    choice_features = L.choice(
        "random-features",
        question="100 bomen op bootstrap-steekproeven van alle vier de metingen: als elke splitsing naar alle vier mag kijken, stellen 77 bomen hun eerste vraag over de flipperlengte; met 2 willekeurige per splitsing doen 41 dat. Waarom wil een random forest dat?",
        options={
            "better": "Zodat elke boom het vaker goed heeft",
            "right": "Zodat de bomen meer verschillen: de stemming kan alleen fouten wegstemmen die de meeste bomen niet delen",
            "faster": "Zodat de bomen sneller leren",
        },
        correct="right",
        why={
            "better": "Een boom die niet naar de beste meting mag kijken, is op zichzelf iets slechter. Wat wint de stemming bij bomen die verschillen? Zie *Het idee*.",
            "faster": "Ze leren wel iets sneller, maar dat is niet de reden. Denk aan de gokkers van de os die allemaal de luidste nazeggen. Zie *Waarom*.",
            "right": "Bomen die allemaal dezelfde eerste vraag stellen, lijken op elkaar en hebben vaak dezelfde pinguïns fout, en een fout die de meeste bomen maken, blijft in de stemming staan. Willekeurige features laten de bomen verschillen, maar dat heeft een prijs: elke boom is op zichzelf iets slechter, dus `max_features` is een hyperparameter die je kiest.",
        },
    )
    choice_features
    return (choice_features,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Elke boom is de boom uit [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl); de stemming is de majority voting uit [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl), en de variantie uit [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl) zegt hoeveel een model met zijn willekeurige trekkingen verandert. De bootstrap komt op gevorderd niveau terug in *Hoe zeker is een getal? Betrouwbaarheidsintervallen en de bootstrap*. In de volgende les, *Een neuron*, zet je de eerste stap naar neural networks.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_features, result_bootstrap, result_vote):
    L.wrap_up(result_bootstrap, result_vote, choice_features)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
