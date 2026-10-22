# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-cross-validation. Route: course/route/lessen/stam-1-basis.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-cross-validation", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_vote():
    quick_vote = L.choice(
        "quick-vote",
        question="Met k = 5 zijn de dichtstbijzijnde trainingspinguïns van een nieuwe pinguïn een Adelie, een Chinstrap, een Chinstrap, een Adelie en een Chinstrap, van dichtbij naar ver weg. Wat voorspelt k-NN?",
        options={
            "nearest": "Adelie: de dichtstbijzijnde is een Adelie",
            "right": "Chinstrap",
            "none": "Niets: de stemmen staan gelijk",
        },
        correct="right",
        why={
            "nearest": "Dat zou k = 1 zeggen. Met k = 5 stemmen ze alle vijf. Zie *Het idee* in [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).",
            "none": "Tel de stemmen van alle vijf. Zie het uitgewerkte voorbeeld in [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).",
            "right": "Drie Chinstraps tegen twee Adelies: majority voting geeft Chinstrap. Dat was de tweede pinguïn uit de validation set van het uitgewerkte voorbeeld.",
        },
    )
    quick_vote
    return (quick_vote,)


@app.cell(hide_code=True)
def quick_check_choose():
    quick_choose = L.choice(
        "quick-choose",
        question="Hoe koos je k?",
        options={
            "train": "De k met de minste fouten op de pinguïns uit de training set",
            "right": "De k met de minste fouten op de pinguïns uit de validation set",
            "test": "De k met de minste fouten op de pinguïns uit de test set",
        },
        correct="right",
        why={
            "train": "Op de pinguïns uit de training set maakt k = 1 helemaal geen fouten, en dat zegt niets over nieuwe pinguïns. Zie oefening 3 in [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).",
            "test": "De test set blijft erbuiten tot alles gekozen is. Zie stap 5 van het recept in [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl).",
            "right": "k-NN bewaart die pinguïns niet, dus ze laten zien welke k werkt op nieuwe pinguïns.",
        },
    )
    quick_choose
    return (quick_choose,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In de vorige les won k = 15 op de validation set met 2 fouten, tegen 4 voor k = 9. Maar dat zijn 68 pinguïns: een paar andere pinguïns in de validation set, en een andere k wint. Je kunt niet zomaar een grotere validation set nemen, want elke pinguïn die je erheen verplaatst, is er een minder om op te trainen. Wat wel kan: elke pinguïn een keer aan de beurt laten komen. Verdeel ze in delen, en laat elk deel één keer de validation set zijn terwijl de andere trainen. Dan tel je de fouten over alle pinguïns.

    **Een voorbeeld uit het dagelijks leven.** Je kiest de beste van twee penaltynemers niet na vijf schoten elk: één gelukstreffer kan de doorslag geven. Laat ze er vijftig nemen, en tel ze allemaal.
    """)
    return


@app.cell(hide_code=True)
def predict_how_many():
    guess_how_many = L.guess(
        "how-many",
        question="Schud de 274 pinguïns buiten de test set acht keer, en kies k elke keer op één validation set van 68, met de andere 206 om op te trainen. Hoeveel verschillende waarden van k denk je dat de acht schudbeurten kiezen?",
        options={
            "one": "1: altijd dezelfde k",
            "few": "2 of 3",
            "many": "4 of meer",
        },
        reveal="**5 verschillende waarden**: 15, 25, 1, 25, 3, 9, 1 en 1. Met 68 pinguïns beslist één fout meer of minder, en welke pinguïns in de validation set komen, is toeval. Cross-validation, hieronder, kiest k = 9 in vijf van de acht schudbeurten, en 5 of 15 in de andere.",
    )
    guess_how_many
    return (guess_how_many,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Cross-validation** verdeelt de voorbeelden waarmee je mag kiezen, hier de 274 pinguïns buiten de test set, in delen die bijna even groot zijn, de **folds**: vaak 5, zoals hier, of 10. Elke fold is één keer de validation set: train op de andere folds, en tel de fouten in deze fold. Na 5 beurten is elke pinguïn precies één keer in de validation set geweest, dus tel je de fouten van de 5 folds op: een telling over alle 274 pinguïns in plaats van 68. Doe dat voor elke waarde van de hyperparameter, en houd de waarde met de minste. Train daarna nog één keer, op alle 274 met die waarde, en meet één keer op de test set.

        Het kost meer: 5 folds zijn 5 keer trainen voor elke waarde. Voor k-NN is trainen alleen de pinguïns standaardiseren en bewaren, dus dat valt hier mee.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **cross-validation** | | schatten hoe goed een model het doet op nieuwe voorbeelden, door elke fold één keer de validation set te laten zijn en de fouten of de loss over alle folds op te tellen; hier om een hyperparameter te kiezen | |
        | **fold** | | een van de delen, bijna even groot, waarin je de voorbeelden verdeelt voor cross-validation | `_val` |
        | `np.concatenate([a, b])` | concatenate | de arrays `a` en `b` achter elkaar, als één array: `[1, 2]` en `[3]` geven `[1, 2, 3]` | `_train` |
        """),
        L.watch_out("Dit heet vaak k-fold cross-validation, met k het aantal folds: een andere k dan die van k-nearest neighbors. Deze les zegt 5 folds, en houdt k voor de buren."),
        mo.md(r"""
        Acht schudbeurten van de 274 pinguïns; schudbeurt 1 is de volgorde van de vorige les. Bij elke laat de ene grafiek voor elke k zien welk deel van de 68 pinguïns van één validation set k-NN <span class="c-loss">fout</span> heeft; de andere welk deel van alle 274 over de 5 folds van cross-validation. k staat op een logaritmische schaal. De dunne lijnen zijn de andere zeven schudbeurten, en de open stip de k die elke manier kiest. Verschuif de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_shuffle_cell():
    def _frame(s):
        s = int(s)
        one, cv = C.chosen(C.one_split(s)), C.chosen(C.cross_validation(s))
        lo1, hi1 = C.spread("one")
        lo2, hi2 = C.spread("cv")
        alt = (f"Twee grafieken van het deel van de pinguïns dat k-NN fout heeft voor elke k van 1 tot 121, elk met acht lijnen, waarvan één dik voor schudbeurt {s}. "
               f"Bij één validation set van 68 liggen de lijnen ver uit elkaar, tussen {L.number(lo1)} en {L.number(hi1)} voor k van 1 tot 41, en schudbeurt {s} kiest k = {one}. "
               f"Bij cross-validation over alle 274 liggen de lijnen dicht bij elkaar, tussen {L.number(lo2)} en {L.number(hi2)} voor k van 1 tot 41, en schudbeurt {s} kiest k = {cv}. "
               "Bijna elke lijn stijgt sterk van k = 71 naar 121.")
        return (C.picture(s, {"one": "Eén validation set: 68 pinguïns", "cv": "Cross-validation: alle 274", "share": "deel fout", "alt": alt}),
                f"""Schudbeurt {s}. Eén validation set kiest <span class="c-hyper">k = {one}</span>; cross-validation kiest <span class="c-hyper">k = {cv}</span>.""")

    explore_shuffle = L.explore("shuffle", "schudbeurt", _frame, start=1, low=1, high=8, step=1, role="",
                                task="Ga de acht schudbeurten langs. Welke k kiest één validation set elke keer, en welke kiest cross-validation?")
    explore_shuffle
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    Eén validation set kiest in acht schudbeurten vijf verschillende waarden van k: 15, 25, 1, 25, 3, 9, 1 en 1. Zijn lijnen springen op en neer, want één pinguïn is 0,015 van 68: één pinguïn met geluk of pech verschuift ze al. De lijnen van cross-validation liggen bijna op elkaar. Het kiest k = 9 in vijf schudbeurten, en 5 of 15 in de andere; twee keer staat k = 5 gelijk met k = 9, en `np.argmin` neemt de eerste. En in elke schudbeurt telt het meer fouten voor k = 1 dan voor k = 9, terwijl één validation set drie keer k = 1 koos, twee keer bij een gelijke stand met vijf andere waarden van k.

    ### Recept

    Om een hyperparameter te kiezen met cross-validation over 5 folds:

    1. Houd de test set apart, en schud de andere voorbeelden.
    2. Snijd de geschudde posities in 5 folds: fold `f` is `order[55 * f:55 * (f + 1)]`, en de training set is de rest, `np.concatenate([order[:55 * f], order[55 * (f + 1):]])`.
    3. Voor elke waarde van de hyperparameter en elke fold: train op de andere folds, en tel de fouten in deze fold.
    4. Tel de fouten van de 5 folds op, en houd de waarde met de minste.
    5. Train op allemaal met die waarde, en meet één keer op de test set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De pinguïns van de vorige les, met snavellengte en lichaamsgewicht: de 68 van de test set apart, en de andere 274 in de volgorde van schudbeurt 1. `_mistakes` is k-NN uit de vorige les: het standaardiseert met de pinguïns waarop het traint, laat de k dichtstbijzijnde stemmen, en telt wat het fout heeft. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_cross_validation():
    import numpy as np
    import pandas as pd

    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["body_mass_g"].isna() == False]
    _X = np.array(_df[["bill_length_mm", "body_mass_g"]])
    _y = np.array(_df["species"])
    _order = np.random.default_rng(27).permutation(len(_df))
    # 1. de test set apart; de andere 274 zijn al geschud
    _rest, _test = _order[:274], _order[274:]


    def _mistakes(train, val, k):
        mean, std = _X[train].mean(axis=0), _X[train].std(axis=0)
        X_train, X_val = (_X[train] - mean) / std, (_X[val] - mean) / std
        wrong = 0
        for i in range(len(val)):
            nearest = np.argsort(np.linalg.norm(X_train - X_val[i], axis=1), kind="stable")[:k]
            names, counts = np.unique(_y[train][nearest], return_counts=True)
            if names[np.argmax(counts)] != _y[val][i]:
                wrong = wrong + 1
        return wrong


    _ks = [1, 3, 5, 9, 15, 25, 41, 71, 121]
    _totals = []
    for _k in _ks:
        _per_fold = []
        _total = 0
        for _f in range(5):
            # 2. fold _f valideert, de andere vier folds trainen
            _val = _rest[55 * _f:55 * (_f + 1)]
            _train = np.concatenate([_rest[:55 * _f], _rest[55 * (_f + 1):]])
            # 3. de fouten op deze fold
            _wrong = _mistakes(_train, _val, _k)
            _per_fold.append(_wrong)
            _total = _total + _wrong
        # 4. opgeteld over de 5 folds: elke pinguïn één keer geteld
        print(_k, _per_fold, _total)
        _totals.append(_total)
    _best = _ks[np.argmin(_totals)]
    print(_best)
    # 5. alle 274 om te stemmen, en één keer op de test set
    print(_mistakes(_rest, _test, _best))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Elke regel geeft k, de fouten op elk van de 5 folds, en hun som. Bij k = 9 tellen de folds 4, 5, 2, 5 en 3 fouten: op zo'n 55 pinguïns elk, alles van 2 tot 5, weer het toeval van één validation set. Opgeteld telt cross-validation 19 fouten van 274 voor k = 9, de minste, en 20 voor k = 15: bijna even goed. In de vorige les had één validation set 2 fouten voor k = 15 en 4 voor k = 9; dat verschil kwam vooral door welke 68 pinguïns er toevallig in zaten. Met k = 9 en alle 274 pinguïns om te stemmen heeft k-NN 2 van de 68 pinguïns uit de test set fout: dat is het getal dat je rapporteert. Het is een kleiner deel dan 19 van 274: met 68 pinguïns in de test set speelt ook hier toeval mee. Eerlijk betekent dat de test set niet meekoos, niet dat zijn getal precies is.

    `np.concatenate` zet het deel vóór fold `_f` en het deel erna achter elkaar: bij de eerste fold is het deel ervoor leeg, bij de laatste het deel erna. De laatste fold heeft 54 pinguïns, omdat 274 geen veelvoud van 5 is; `_rest[220:275]` stopt gewoon aan het eind.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_why():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Waarom betrouwbaarder?
    """)
    return


@app.cell(hide_code=True)
def choice_why_cell():
    choice_why = L.choice(
        "why-reliable",
        question="Waarom kiest cross-validation k betrouwbaarder dan één validation set van 68 pinguïns?",
        options={
            "train": "In elke fold bewaart k-NN meer pinguïns: 219 of 220 in plaats van 206",
            "right": "Het telt de fouten op alle 274 pinguïns, dus een paar pinguïns met geluk of pech wegen minder zwaar",
            "test": "Het gebruikt ook de test set",
        },
        correct="right",
        why={
            "train": "Dat klopt, maar een paar pinguïns meer om te stemmen maakt weinig uit. Wat het meest uitmaakt, is op hoeveel pinguïns de fouten geteld worden. Zie *Het idee*.",
            "test": "De test set blijft apart, in elke fold. Zie stap 1 van het recept.",
            "right": "Elke pinguïn komt één keer aan de beurt in de validation set, dus de telling gaat over 274 pinguïns in plaats van 68, en het toeval middelt beter uit.",
        },
    )
    choice_why
    return (choice_why,)


@app.cell(hide_code=True)
def text_exercise_folds():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Folds voor een curve

        Cross-validation werkt voor elke hyperparameter, ook voor de graad van een curve. `x` en `y` zijn de 18 temperaturen van oefening 2 in [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl), de 12 om op te trainen en de 6 om te valideren samen, en `order` schudt hun posities. Het vak gebruikt 3 folds van 6. Vervang de twee `...`: zet de posities van fold `f` in `val`, en de posities van de andere twee folds in `train`. Het vak leert dan van `train` een curve van elke graad, berekent de som van zijn gekwadrateerde residuen op `val`, en deelt door 18: de MSE over alle 18 temperaturen, elk één keer gemeten. De laatste regel is `cv_mses`, zodat de check het ziet. De check probeert ook andere graden.
        """),
        L.task("Zet de posities van fold `f` in `val`, en die van de andere folds in `train`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_folds():
    box_folds = L.code_box("folds", lines=15)
    box_folds
    return (box_folds,)


@app.cell(hide_code=True)
def check_folds(box_folds):
    result_folds = L.check("folds", box_folds.value)
    result_folds
    return (result_folds,)


@app.cell(hide_code=True)
def hints_folds():
    L.hints(
        "folds",
        [
            "Stap 2 van het recept, met folds van 6 in plaats van 55.",
            "Fold `f` begint op positie `6 * f` van `order` en stopt vóór `6 * (f + 1)`. `train` is wat vóór dat stuk komt, gevolgd door wat erna komt.",
            "`val = order[6 * f:___]` en `train = np.concatenate([order[:6 * f], order[___:]])`.",
        ],
        why="Met 3 folds van 6 zijn de MSE's van cross-validation voor graad 1 tot 6 ongeveer 7,26, 1,00, 1,28, 1,65, 1,25 en 2,35: graad 2 heeft de laagste, zoals de validation set in die les koos.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_after():
    mo.md(r"""
    ### Oefening 3. En dan?
    """)
    return


@app.cell(hide_code=True)
def choice_after_cell():
    choice_after = L.choice(
        "after-cv",
        question="Cross-validation koos k = 9, met 19 fouten van 274. Wat doe je nu?",
        options={
            "fold": "De k-NN houden van de fold die de minste fouten maakte",
            "right": "k-NN met k = 9 alle 274 pinguïns laten gebruiken, en het één keer meten op de test set",
            "report": "19 van 274 rapporteren als hoe k-NN het op nieuwe pinguïns doet",
        },
        correct="right",
        why={
            "fold": "De k-NN van elke fold bewaarde maar vier vijfde van de pinguïns. Nu de waarde gekozen is, gebruik je ze allemaal. Zie stap 5 van het recept.",
            "report": "Die 19 hielpen k kiezen, dus het getal is een beetje te optimistisch, net als bij één validation set. Zie stap 5 van het recept.",
            "right": "De test set hielp niet mee kiezen, dus zijn telling is een eerlijke schatting voor nieuwe pinguïns: hier 2 van 68.",
        },
    )
    choice_after
    return (choice_after,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Hyperparameters kiezen: de validation set](/course/open/ml-validation-set?lang=nl) koos één validation set de graad; cross-validation laat elk voorbeeld een keer aan de beurt komen, en koos ook graad 2. In [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl) won k = 15 op 68 pinguïns; over alle 274 zijn k = 9 tot 41 ongeveer even goed, en is k = 1 in elke schudbeurt slechter. In de volgende les, *Regularisatie en weight decay*, houd je een flexibel model op een andere manier in toom: door grote weights iets te laten kosten. Hoeveel ze kosten, is weer een hyperparameter, die je op dezelfde manier kiest.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_why, result_folds, choice_after):
    L.wrap_up(choice_why, result_folds, choice_after)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
