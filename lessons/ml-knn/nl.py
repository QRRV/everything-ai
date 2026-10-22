# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-knn. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-knn", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl) en [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_distance():
    quick_distance = L.choice(
        "quick-distance",
        question="Wat is de afstand tussen de punten (1, 2) en (4, 6)?",
        options={
            "steps": "7",
            "right": "5",
            "square": "25",
        },
        correct="right",
        why={
            "steps": "Dat is 3 opzij plus 4 omhoog, alsof je langs de straten loopt. De afstand is de rechte lijn. Zie *Het idee* in [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl).",
            "square": "Dat is 9 + 16: het kwadraat van de afstand. Neem de wortel. Zie *Het idee* in [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl).",
            "right": "3 opzij en 4 omhoog: de wortel uit 9 + 16 is 5, de norm van het verschil.",
        },
    )
    quick_distance
    return (quick_distance,)


@app.cell(hide_code=True)
def quick_check_validation():
    quick_validation = L.choice(
        "quick-validation",
        question="Je past een curve van elke graad van 1 tot 9 aan. Op welke voorbeelden vergelijk je ze om de graad te kiezen?",
        options={
            "train": "De training set: daarvan leerden de curves",
            "right": "De validation set",
            "test": "De test set",
        },
        correct="right",
        why={
            "train": "Op de training set ziet de flexibelste curve er altijd het best uit. Zie *Het idee* in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl).",
            "test": "De test set meet het gekozen model één keer, aan het eind; een set die hielp kiezen, geeft een te optimistische score. Zie *Waarom* in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl).",
            "right": "Voorbeelden waarop de curves niet trainden, apart gehouden van de test set.",
        },
    )
    quick_validation
    return (quick_validation,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Het opzoekmodel uit [Werkt het op nieuwe data? Training set en test set](/course/open/ml-train-test-split?lang=nl) gaf een nieuw huis de prijs van het dichtstbijzijnde huis uit de training set. Met hetzelfde idee kun je ook classificeren: geef een nieuwe pinguïn de soort van de gemeten pinguïn die er het meest op lijkt. Maar één buur kan een uitschieter zijn. Vraag het aan een paar buren en laat ze stemmen, dan wordt één uitschieter weggestemd. Hoeveel buren je het vraagt, is een hyperparameter.

    **Een voorbeeld uit het dagelijks leven.** Wil je raden welke taal ze spreken in een dorp waar je nog nooit van gehoord hebt, dan kijk je naar de dorpen eromheen. Het ene dorp dat er het dichtst bij ligt, kan een uitzondering zijn; de vijf dichtstbijzijnde dorpen zijn een veiligere gok. Neem je alle dorpen van het land, dan weet je alleen welke taal er in het land het meest gesproken wordt.
    """)
    return


@app.cell(hide_code=True)
def predict_best_k():
    guess_best_k = L.guess(
        "best-k",
        question="k-NN vraagt het aan de k trainingspinguïns die het meest op een nieuwe lijken, en neemt de soort die de meesten van hen hebben. Bij welke k heeft k-NN volgens jou de minste van de 68 pinguïns uit de validation set fout?",
        options={
            "one": "k = 1: de ene pinguïn die er het meest op lijkt",
            "fifteen": "k = 15",
            "all": "k = 205: op één na alle 206 trainingspinguïns",
        },
        reveal="**k = 15**, met 2 van de 68 fout. Met k = 1 heeft k-NN er 6 fout: het volgt elke losse pinguïn, zelfs een die op een andere soort lijkt. Met k = 205 heeft het er 39 fout: bijna elke trainingspinguïn stemt mee, dus noemt het elke pinguïn een Adelie, de soort die het vaakst voorkomt in de training set.",
    )
    guess_best_k
    return (guess_best_k,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **k-nearest neighbors**, of **k-NN**, classificeert een nieuw voorbeeld in drie stappen: reken zijn afstand uit tot elk voorbeeld uit de training set, neem de k met de kleinste afstand, zijn **dichtstbijzijnde buren**, en laat ze stemmen. De klasse die de meesten hebben, wint: **majority voting**. k-NN leert geen parameters; het bewaart de voorbeelden uit de training set en doet zijn werk pas als het voorspelt.

        Met k = 1 is k-NN het opzoekmodel voor klassen: elk voorbeeld uit de training set is zijn eigen dichtstbijzijnde buur, dus heeft het ze allemaal goed, en de eigenaardigheden van losse pinguïns beslissen voor de pinguïns eromheen: het overfit. Een grotere k strijkt dat glad, tot het met een k zo groot als de training set altijd de klasse noemt die het vaakst voorkomt: het underfit. Dus kies je k op de validation set.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **k-nearest neighbors**, **k-NN** | kaa-en-en | een classifier die een nieuw voorbeeld de klasse geeft die de meesten van zijn k dichtstbijzijnde voorbeelden uit de training set hebben | `knn_predict(X_train, y_train, x, k)` |
        | **dichtstbijzijnde buren** | | de voorbeelden uit de training set met de kleinste afstand tot het nieuwe voorbeeld | `nearest` |
        | **majority voting** | | van een paar antwoorden het antwoord nemen dat het vaakst voorkomt | |
        | `np.linalg.norm(A - x, axis=1)` | | de afstand van `x` tot elke rij van `A`: één getal per rij | `dists` |
        | `np.argsort(a, kind="stable")` | arg sort | de posities van de elementen van `a` van het kleinste naar het grootste, gelijke in hun volgorde: bij 3, 1 en 2 de posities 1, 2 en 0 | `nearest` |
        | `np.unique(a, return_counts=True)` | | de verschillende waarden in `a`, gesorteerd, en hoe vaak elk voorkomt | `names, counts` |
        """),
        L.watch_out("De afstand telt de kwadraten van de verschillen in alle features op, dus een feature met grote getallen telt zwaarder. Het lichaamsgewicht is in grammen, duizenden; de snavellengte in millimeters, tientallen. Zonder standaardiseren is de afstand bijna alleen het lichaamsgewicht. Standaardiseer dus eerst, met het gemiddelde en de standaardafwijking van de training set, zoals in [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl)."),
        mo.md(r"""
        Het beeld laat de 206 trainingspinguïns van het uitgewerkte voorbeeld hieronder zien, naar snavellengte en lichaamsgewicht: een rondje is een Adelie, een driehoekje een Chinstrap en een vierkantje een Gentoo; kleur en vorm betekenen hier de soort. De <span class="c-pred">blauwe lijn</span> is de decision boundary van k-NN: aan weerszijden ervan voorspelt k-NN een andere soort. De andere grafiek telt voor elke k de <span class="c-loss">fouten</span> op de 68 pinguïns uit de validation set, met k op een logaritmische schaal. Verschuif de slider, of sleep over die grafiek.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_k_cell():
    def _frame(k):
        k = int(k)
        m = C.mistakes()[C.KS.index(k)]
        alt = (f"De eerste grafiek: de 206 trainingspinguïns naar snavellengte en lichaamsgewicht, een kleur en vorm per soort. Adelies hebben een korte snavel, Gentoos zijn zwaar, Chinstraps hebben een lange snavel en zijn licht. Met k = {k} "
               + ("is er geen blauwe lijn: k-NN noemt elk punt een Adelie." if C.species_in_picture(k) == 1 else
                  "scheidt een blauwe lijn de gebieden waar k-NN elke soort voorspelt; bij een kleine k buigt hij om losse pinguïns heen, bij een grotere is hij gladder.")
               + f" De tweede grafiek: de fouten op de validation set voor elke k, 6 bij k = 1, 2 bij k = 15 en 39 bij k = 205; bij k = {k} zijn het er {m}.")
        said = f"""Met k = {k}: <span class="c-loss ea-num">{m}</span> van de 68 pinguïns uit de validation set fout."""
        if C.species_in_picture(k) == 1:
            said += " Elke pinguïn heet een Adelie."
        return (C.picture(k, {"title": f"Decision boundary, k = {k}", "x": "snavellengte (mm)", "y": "lichaamsgewicht (g)",
                              "every": "Fouten voor elke k", "mistakes": "fouten (van 68)", "alt": alt}), said)

    explore_k = L.explore("k", "k", _frame, start=1, values=C.KS, role="hyper",
                          task="Schuif k van 1 naar 205. Waar is de grens het grilligst, en welke k maakt de minste fouten?")
    explore_k
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Bij k = 1 is de grens het grilligst: hij buigt om losse pinguïns heen die tussen een andere soort zitten, en k-NN heeft er 6 van de 68 fout. Hij wordt gladder naarmate k groeit, en k = 15 maakt de minste fouten: 2. Daarna verliezen de Chinstraps, de kleinste groep met 44 van de 206, steeds meer stemmen: bij k = 205 is de grens weg, is elke pinguïn een Adelie, en zijn er 39 fout.

    ### Recept

    Om een nieuw voorbeeld te classificeren met k-NN:

    1. Standaardiseer de features met het gemiddelde en de standaardafwijking van de training set, ook het nieuwe voorbeeld.
    2. Reken de afstand uit van het nieuwe voorbeeld tot elk voorbeeld uit de training set: `np.linalg.norm(X_train - x, axis=1)`.
    3. Neem de posities van de k dichtstbijzijnde: `np.argsort(dists, kind="stable")[:k]`.
    4. Tel hun stemmen, `np.unique(y_train[nearest], return_counts=True)`, en geef de klasse met de meeste terug.
    5. Om k te kiezen: doe dit voor elk voorbeeld uit de validation set met elke waarde van k, tel de fouten, en houd de k met de minste.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De pinguïns van het beeld: de 342 met alle metingen, geschud met een seed en gesplitst in 206 om op te trainen, 68 om mee te kiezen en 68 voor de test, die buiten zicht blijven. Daarna stap 1 tot en met 4 voor de tweede pinguïn uit de validation set, met k = 5. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_knn():
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
    print(_X[_val[1]], _y_val[1])
    # 2. de afstand van deze pinguïn uit de validation set tot alle 206 uit de training set
    _dists = np.linalg.norm(_X_train - _X_val[1], axis=1)
    # 3. de posities van de 5 dichtstbijzijnde, en hoe ver weg ze zijn
    _nearest = np.argsort(_dists, kind="stable")[:5]
    print(_dists[_nearest])
    # 4. hun soorten, de stemmen, en de soort met de meeste
    print(_y_train[_nearest])
    _names, _counts = np.unique(_y_train[_nearest], return_counts=True)
    print(_names, _counts)
    print(_names[np.argmax(_counts)])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De tweede pinguïn uit de validation set, 43,5 mm en 3400 g, is een Chinstrap. Zijn dichtstbijzijnde trainingspinguïn is een Adelie, dus k = 1 zou hem fout hebben; van zijn 5 dichtstbijzijnde zijn er 3 Chinstraps, en de stemming geeft de goede soort. `np.unique` sorteert de soorten en telt ze, 2 en 3; `np.argmax` geeft positie 1, en `_names[1]` is Chinstrap. Bij gelijke stemmen neemt hij de eerste: de soort die vooraan staat in het alfabet.

    De tweede en derde buur zijn precies hetzelfde gemeten, 42,5 mm en 3350 g, dus ze liggen precies even ver weg; `kind="stable"` houdt zulke pinguïns in hun volgorde in de training set, zodat k-NN op elke computer hetzelfde antwoord geeft.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_predict():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Je eigen k-NN

        Maak `knn_predict(X_train, y_train, x, k)` af. De functie krijgt de gestandaardiseerde voorbeelden uit de training set, één per rij, hun klassen, één nieuw voorbeeld `x` en `k`, en geeft de klasse terug die de meesten van de k dichtstbijzijnde hebben. Stap 2 van het recept, `dists`, staat er al: vervang de twee `...` door stap 3 en 4. De check roept `knn_predict(X_train, y_train, np.array([0.2, -0.9]), 3)` aan met tien verzonnen pinguïns, al gestandaardiseerd: dat moet `"Adelie"` geven. Hij probeert ook andere punten en andere waarden van k.
        """),
        L.task("Laat `knn_predict` de soort teruggeven die de meesten van de k dichtstbijzijnde trainingspinguïns hebben."),
    ])
    return


@app.cell(hide_code=True)
def exercise_predict():
    box_predict = L.code_box("knn-predict", lines=8)
    box_predict
    return (box_predict,)


@app.cell(hide_code=True)
def check_predict(box_predict):
    result_predict = L.check("knn-predict", box_predict.value)
    result_predict
    return (result_predict,)


@app.cell(hide_code=True)
def hints_predict():
    L.hints(
        "knn-predict",
        [
            "Stap 3 en 4 van het recept, zoals in het uitgewerkte voorbeeld, met `k` in plaats van 5.",
            "`np.argsort(dists, kind=\"stable\")` geeft elke positie, de dichtstbijzijnde eerst: welk deel ervan houd je? Na `np.unique` is `np.argmax(counts)` een positie: in welke van de twee arrays zoek je die op?",
            "`nearest = np.argsort(dists, kind=\"stable\")[___]` en `return names[___]`.",
        ],
        why="Met k = 3 zijn de drie dichtstbijzijnde van `np.array([0.2, -0.9])` een Chinstrap en twee Adelies: `names` is `['Adelie' 'Chinstrap']`, `counts` is `[2 1]`, `np.argmax(counts)` is 0, en `names[0]` is Adelie.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_choose():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Kies k

        Nu met de flipperlengte en het lichaamsgewicht in plaats van de snavellengte: `X_train` en `y_train` zijn de 206 trainingspinguïns, `X_val` en `y_val` de 68 uit de validation set, gesplitst en gestandaardiseerd zoals in het uitgewerkte voorbeeld. In dit vak staat `knn_predict` klaar: de versie van de course van je functie uit oefening 1. Tel in de loop 1 op bij `wrong` voor elke pinguïn uit de validation set die k-NN fout heeft; zet daarna de k met de minste fouten in `best`. Vervang de twee `...`. De laatste regel is `best`, zodat de check het ziet. De check probeert ook andere waarden van k.
        """),
        L.task("Zet de k met de minste fouten op de validation set in `best`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_choose():
    box_choose = L.code_box("choose-k", lines=10)
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
            "Stap 5 van het recept, met `np.argmin` zoals in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl).",
            "In de loop: is de voorspelling voor `X_val[i]` niet `y_val[i]`, tel dan 1 op bij `wrong`. Na de loop geeft `np.argmin(mistakes)` een positie: welke k staat op die positie in `ks`?",
            "`if knn_predict(X_train, y_train, X_val[i], k) != ___:`, daaronder `wrong = wrong + 1`, en `best = ks[___]`.",
        ],
        why="De fouten voor de tien waarden van k zijn 15, 16, 19, 17, 13, 13, 11, 13, 17 en 39. De minste staan op positie 6, en `ks[6]` is 41.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_train():
    mo.md(r"""
    ### Oefening 3. k = 1 op de training set
    """)
    return


@app.cell(hide_code=True)
def choice_train_cell():
    choice_train = L.choice(
        "k1-train",
        question="Met k = 1, snavellengte en lichaamsgewicht heeft k-NN 6 van de 68 pinguïns uit de validation set fout. Hoeveel van zijn eigen 206 trainingspinguïns heeft het fout?",
        options={
            "six": "Ongeveer evenveel als op de validation set: zo'n 18 van de 206",
            "right": "Geen enkele",
            "more": "Meer dan met k = 15",
        },
        correct="right",
        why={
            "six": "Dat geldt voor pinguïns die het nooit zag. Welke pinguïn ligt het dichtst bij een trainingspinguïn zelf? Zie *Het idee*.",
            "more": "Welke pinguïn ligt het dichtst bij een trainingspinguïn zelf, en hoe ver weg ligt die? Zie *Het idee*.",
            "right": "Elke trainingspinguïn is zijn eigen dichtstbijzijnde buur, op afstand 0, dus k = 1 geeft zijn eigen soort terug. Met k = 15 heeft het er 17 fout. Net als bij het opzoekmodel zegt een perfecte score op de training set niets over nieuwe pinguïns.",
        },
    )
    choice_train
    return (choice_train,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het opzoekmodel uit [Werkt het op nieuwe data? Training set en test set](/course/open/ml-train-test-split?lang=nl) was k-NN met k = 1 voor een getal, de prijs, in plaats van een klasse; nu stemmen meer buren, en k is een hyperparameter die je kiest op de validation set, zoals de graad in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl). De afstand is die uit [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl), en hij werkt alleen goed op gestandaardiseerde features. In de volgende les, *Betrouwbaarder kiezen: cross-validation*, zie je dat 68 pinguïns in de validation set weinig zijn om mee te kiezen, en hoe elke pinguïn een keer aan de beurt kan komen in de validation set.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_predict, result_choose, choice_train):
    L.wrap_up(result_predict, result_choose, choice_train)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
