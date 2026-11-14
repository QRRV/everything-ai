# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson data-centric-ai. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("data-centric-ai", "nl")
    C = L.checks   # de 274 pinguïns met 41 labels die de course verkeerd maakte, hun out-of-fold kansen, de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Betrouwbaarder kiezen: cross-validation](/course/open/ml-cross-validation?lang=nl) en één over [Kansen scoren met de log: log loss](/course/open/ml-log-loss?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_folds():
    quick_folds = L.choice(
        "quick-folds",
        question="Cross-validation verdeelt de 274 pinguïns in 5 folds. Hoe vaak zit elke pinguïn in de validation set?",
        options={
            "never": "Nooit: alle 274 zitten in de training set",
            "right": "Eén keer: elke fold is één keer aan de beurt",
            "five": "5 keer, één keer voor elke fold",
        },
        correct="right",
        why={
            "never": "Elke fold is één keer de validation set, dus elke pinguïn zit er één keer in. Zie *Het idee* in [Betrouwbaarder kiezen: cross-validation](/course/open/ml-cross-validation?lang=nl).",
            "five": "In de andere vier beurten zit de pinguïn in de training set. Zie *Het idee* in [Betrouwbaarder kiezen: cross-validation](/course/open/ml-cross-validation?lang=nl).",
            "right": "Goed: elke pinguïn wordt één keer gevalideerd, door een model dat op de andere vier folds trainde. Dat gebruik je vandaag.",
        },
    )
    quick_folds
    return (quick_folds,)


@app.cell(hide_code=True)
def quick_check_surprise():
    quick_surprise = L.choice(
        "quick-surprise",
        question="Een model gaf wat er echt gebeurde een kans van 0,1: de verrassing is $-\\log(0{,}1) \\approx 2{,}3$. Wat is de verrassing van een uitkomst waaraan het 0,9 gaf?",
        options={
            "same": "Ook ongeveer 2,3",
            "nine": "Ongeveer 21, negen keer zoveel",
            "right": "Ongeveer 0,1: een waarschijnlijke uitkomst verrast weinig",
        },
        correct="right",
        why={
            "same": "De verrassing hangt af van de kans: hoe kleiner die is, hoe groter de verrassing. Zie *Het idee* in [Kansen scoren met de log: log loss](/course/open/ml-log-loss?lang=nl).",
            "nine": "Een hogere kans geeft een kleinere verrassing, geen grotere. Zie *Het idee* in [Kansen scoren met de log: log loss](/course/open/ml-log-loss?lang=nl).",
            "right": "Goed: $-\\log(0{,}9) \\approx 0{,}11$. Vandaag is een lage kans op een label een grote verrassing, en een reden om nog eens te kijken.",
        },
    )
    quick_surprise
    return (quick_surprise,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model leert van zijn labels, en labels bevatten fouten: verkeerd getypt, verkeerd beoordeeld, verkeerd overgenomen. In tien bekende test sets schatten Northcutt en collega's (2021) dat gemiddeld minstens 3,3% van de labels verkeerd is, en minstens 6% in de validation set van ImageNet. Je kunt niet elk label met de hand controleren. Welke controleer je eerst?

    **Een voorbeeld uit het dagelijks leven.** Een boekhouder controleert niet alle 300 bonnetjes opnieuw. Die kijkt eerst naar de bonnetjes die opvallen: een koffie van 450 euro is eerder een tikfout dan een koffie van 4,50.
    """)
    return


@app.cell(hide_code=True)
def predict_found_cell():
    guess_found = L.guess(
        "found-guess",
        question="De course maakte 41 van de 274 labels van de pinguïns verkeerd. Als je 30 willekeurige labels controleert, vind je er ongeveer 4,5. Hoeveel vind je als je de 30 labels controleert die k-NN, getraind zonder die pinguïns, het minst waarschijnlijk vindt?",
        options={
            "same": "Ongeveer 5: k-NN kan niet weten welke labels verkeerd zijn",
            "some": "Ongeveer 12",
            "right": "Ongeveer 25",
        },
        reveal="**Ongeveer 25.** De 30 verdachtste labels bevatten hier 25 van de 41 verkeerde; met andere willekeurig verkeerd gemaakte labels 22 tot 27. Een verkeerd label past niet bij de pinguïns eromheen, en dat merkt k-NN.",
    )
    guess_found
    return (guess_found,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Verkeerde labels.** Verkeerde labels in een dataset heten **label noise**. Een model dat erop traint, leert een deel van de fouten, en een test set met zulke labels meet tegen de fouten.

        Hier maakt de course zelf 41 van de 274 labels van de pinguïns buiten de test set verkeerd, 15%, willekeurig: elk wordt een van de twee andere soorten. Zo weet de course welke labels verkeerd zijn, en kan ze tellen hoeveel je er vindt. Bij echte data weet niemand dat; controleren is de enige manier om erachter te komen. De 68 pinguïns in de test set houden hun goede labels: je hebt een eerlijk getal nodig.

        **Het model als speurder.** Een verkeerd label past niet bij de pinguïns eromheen: een pinguïn met het label Gentoo tussen Adelies. k-NN met k = 9, de k die cross-validation koos, zegt hoe sterk: de kans die het aan een soort geeft, is het deel van de 9 dichtstbijzijnde pinguïns met die soort. Kijk naar de kans op het label dat elke pinguïn heeft. Een lage kans is een grote verrassing, $-\log(p)$: dat label is verdacht. Controleer de verdachte labels eerst, van de laagste kans naar de hoogste.

        **Getraind zonder de pinguïn zelf.** De k-NN mag de pinguïn zelf niet bewaard hebben. Anders is de pinguïn een van zijn eigen buren, op afstand 0, en stemt hij op zijn eigen label, goed of fout. Gebruik daarom de folds van cross-validation: fold 1 krijgt zijn kansen van een k-NN die alleen fold 2 tot en met 5 bewaarde, fold 2 van fold 1, 3, 4 en 5, enzovoort. Elke pinguïn krijgt een kans van een model dat hem nooit zag: een **out-of-fold-voorspelling**.

        **Verbeter de data.** Controleer de verdachte labels met de hand, in de veldnotities, op een foto of met een expert, en verbeter de labels die verkeerd zijn. Train daarna opnieuw en meet op de test set. Het model hetzelfde houden en de data verbeteren heet **data-centric AI**; het model veranderen, bijvoorbeeld een andere k kiezen, is de modelgerichte aanpak.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **label noise** | verkeerde labels in een dataset: verkeerd getypt, beoordeeld of overgenomen | |
        | **out-of-fold-voorspelling** | een voorspelling voor een voorbeeld door een model dat op de andere folds trainde, zonder dat voorbeeld | `_p` |
        | **data-centric AI** | een systeem beter maken door de data te verbeteren, zoals labels verbeteren, terwijl het model hetzelfde blijft | |
        | `np.mean(a == b)` | het deel van de plaatsen waar `a` en `b` hetzelfde zijn: `a == b` geeft `True` en `False`, en het gemiddelde telt `True` als 1 en `False` als 0 | `np.mean(_given[_train][_nearest] == _given[_i])` |
        | `np.sum(a != b)` | het aantal plaatsen waar `a` en `b` verschillen | |
        """),
        L.watch_out("Een lage kans zegt: kijk hier eens naar. Ze zegt niet dat het label verkeerd is: sommige pinguïns lijken niet op hun eigen soort. Verander alleen labels die je controleerde."),
        mo.md(r"""
        Met de slider kies je hoeveel labels je controleert, de verdachtste eerst. Links hoeveel van de 41 verkeerde labels je vindt, tegenover willekeurig controleren, de gestreepte lijn. Rechts bij hoeveel van de 68 pinguïns in de test set k-NN fout zit nadat je hebt verbeterd wat je vond: met k = 1, die naar één buur luistert, en met k = 9. Schuif de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_checked_cell():
    def _frame(c):
        said = C.said(c, L.lang)
        alt = ("Links: hoeveel van de 41 verkeerde labels je vindt tegenover hoeveel labels je controleert, de verdachtste "
               "eerst en willekeurig. Rechts: de fouten van k-NN met k = 1 en k = 9 op de 68 pinguïns in de test set nadat "
               "je hebt verbeterd wat je vond. " + said)
        return (C.picture(c, C.words(L.lang) | {"alt": alt}, L.lang), said)

    explore_checked = L.explore("checked", "labels gecontroleerd", _frame, start=30, values=C.CHECKS, role="hyper",
                                task="Schuif de slider van 0 naar 100. Waar vind je met controleren nog maar weinig verkeerde labels erbij, en wat doet verbeteren voor k = 1 en voor k = 9?")
    explore_checked
    return


@app.cell(hide_code=True)
def text_after_checked():
    mo.md(r"""
    De eerste 30 controles vinden 25 verkeerde labels, de volgende 20 er nog 13; na 50 controles zijn er nog maar 3 over: één komt bij controle 58 boven, en de andere twee lijken voor k-NN goed: 7 en 8 van hun 9 buren hebben hun label. k = 1 heeft het meeste last van verkeerde labels: 14 fouten op de test set met die labels, 6 met alle labels goed, en 7 na het verbeteren van de 25 uit de eerste 30 controles. k = 9 stemt de meeste verkeerde labels zelf al weg: 3 fouten, en 2 na het verbeteren.

    En de modelgerichte aanpak? Opnieuw k kiezen met cross-validation, op de labels met de fouten, kiest k = 15, en die zit ook bij 2 pinguïns in de test set fout: een grotere stemming verbergt willekeurige fouten ook. Maar de labels blijven verkeerd. Het volgende model dat je erop traint, heeft er weer last van, en cross-validation vergelijkt zelf ook met die labels: een goede voorspelling voor een pinguïn met een verkeerd label telt als fout. Verbeterde labels blijven verbeterd: elk model dat je er later op traint, en elke vergelijking die cross-validation ermee maakt, begint bij de goede labels.

    ### Recept

    Zo vind je de verkeerde labels in je data:

    1. Verdeel de voorbeelden in 5 folds, zoals bij cross-validation.
    2. Train voor elke fold op de andere vier, en laat het model elk voorbeeld van deze fold een kans geven op zijn eigen label. Bij k-NN: het deel van zijn 9 dichtstbijzijnde buren met dat label.
    3. Sorteer van de laagste kans naar de hoogste.
    4. Controleer de eerste labels met de hand, en verbeter de labels die verkeerd zijn.
    5. Train opnieuw op de verbeterde data, en meet op de test set, die nergens aan meedeed.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel maakt dezelfde 41 labels verkeerd als de slider, geeft elk van de 274 pinguïns de kans op zijn label out-of-fold, met k = 9, en print de 8 verdachtste: het nummer van de pinguïn onder de 274, zijn label, de kans, en de soort die hij echt is. Daarna telt ze de verkeerde labels onder de 30 verdachtste. Voer de cel uit.
    """)
    return


@app.cell
def example_suspects():
    import numpy as np
    import pandas as pd

    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["body_mass_g"].isna() == False]
    _X = np.array(_df[["bill_length_mm", "body_mass_g"]])
    _y = np.array(_df["species"])
    _order = np.random.default_rng(27).permutation(len(_df))
    _rest = _order[:274]

    # de course maakt 41 van de 274 labels verkeerd: elk wordt een van de twee andere soorten
    _rng = np.random.default_rng(15)
    _given = np.array(_y)                                  # een nieuwe array met dezelfde labels
    _wrong = _rng.permutation(_rest)[:41]
    for _i in _wrong:
        _others = []
        for _s in ["Adelie", "Chinstrap", "Gentoo"]:
            if _s != _y[_i]:
                _others.append(_s)
        _given[_i] = _others[_rng.integers(0, 2)]

    _p = []
    for _f in range(5):
        # 1. fold _f krijgt zijn kansen van de andere vier folds
        _val = _rest[55 * _f:55 * (_f + 1)]
        _train = np.concatenate([_rest[:55 * _f], _rest[55 * (_f + 1):]])
        _mean, _std = _X[_train].mean(axis=0), _X[_train].std(axis=0)
        _X_train = (_X[_train] - _mean) / _std
        for _i in _val:
            # 2. de 9 dichtstbijzijnde pinguïns in de andere folds
            _dists = np.linalg.norm(_X_train - (_X[_i] - _mean) / _std, axis=1)
            _nearest = np.argsort(_dists, kind="stable")[:9]
            # 3. de kans op zijn eigen label: het deel van de 9 met dat label
            _p.append(np.mean(_given[_train][_nearest] == _given[_i]))
    # 4. de verdachtste eerst
    _suspects = np.argsort(_p, kind="stable")[:30]
    for _j in _suspects[:8]:
        print(_j, _given[_rest[_j]], round(_p[_j], 2), _y[_rest[_j]])
    print(np.sum(_given[_rest[_suspects]] != _y[_rest[_suspects]]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De 8 verdachtste hebben allemaal kans 0: geen van hun 9 buren heeft hun label. De eerste zes zijn verkeerd. Pinguïn 3, een Adelie van 40,6 mm en 3550 g, heeft het label Gentoo, en Gentoos wegen rond de 5000 g. Pinguïns 107 en 109 zijn wel goed: Adelies met een snavel van ongeveer 46 mm, bijna zo lang als die van een Chinstrap. Het model vindt ze verdacht; je kijkt, en laat ze zoals ze zijn. Van de 30 verdachtste labels zijn er 25 verkeerd.

    `_given[_train][_nearest]` neemt eerst de labels van de trainingsfolds, en daarna die van de 9 dichtstbijzijnde daarvan: `_nearest` bevat posities in `_train`, niet in alle pinguïns.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_p_given():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De kans op het eigen label van een pinguïn

        `p_given(X_train, y_train, x, label, k)` krijgt gestandaardiseerde pinguïns uit de training set en hun labels, één pinguïn `x` met zijn label `label`, en `k`. Ze moet de kans geven die k-NN aan dat label geeft: het deel van de `k` dichtstbijzijnde pinguïns uit de training set met als label `label`. Vervang de twee `...`. De check roept `p_given(X_train, y_train, np.array([0.2, -0.9]), "Chinstrap", 3)` aan met de tien verzonnen pinguïns uit [Vraag het de buren: k-nearest neighbors](/course/open/ml-knn?lang=nl), wat een derde moet geven, en nog zeven andere.
        """),
        L.task("Laat `p_given` het deel van de `k` dichtstbijzijnde met het eigen label van de pinguïn geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_p_given():
    box_p_given = L.code_box("p-given", lines=8)
    box_p_given
    return (box_p_given,)


@app.cell(hide_code=True)
def check_p_given(box_p_given):
    result_p_given = L.check("p-given", box_p_given.value)
    result_p_given
    return (result_p_given,)


@app.cell(hide_code=True)
def hints_p_given():
    L.hints(
        "p-given",
        [
            "Twee stappen: welke pinguïns uit de training set zijn de `k` dichtstbijzijnde, en welk deel daarvan heeft `label`?",
            "Net als bij k-NN zet `np.argsort(dists, kind=\"stable\")` de dichtstbijzijnde vooraan. Een array vergelijken met `==` geeft `True` en `False`: wat maakt `np.mean` daarvan?",
            "`nearest = np.argsort(dists, kind=\"stable\")[___]` en `return np.mean(y_train[___] == label)`.",
        ],
        why="Bij het voorbeeld zijn de 3 dichtstbijzijnde een Chinstrap en twee Adelies: `y_train[nearest] == \"Chinstrap\"` geeft `True`, `False` en `False`, en het gemiddelde daarvan is een derde.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_suspects():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De verdachtste eerst

        `suspects(p, n)` krijgt de kansen `p` van de labels, en moet de posities van de `n` verdachtste geven, de laagste kans eerst. Vervang de twee `...`. De check roept `suspects(np.array([0.89, 0.11, 1.0, 0.0, 0.56, 0.22]), 3)` aan, wat `[3, 1, 5]` moet geven, en nog vijf andere.
        """),
        L.task("Laat `suspects` de posities van de `n` laagste kansen geven, de laagste eerst."),
    ])
    return


@app.cell(hide_code=True)
def exercise_suspects():
    box_suspects = L.code_box("suspects", lines=6)
    box_suspects
    return (box_suspects,)


@app.cell(hide_code=True)
def check_suspects(box_suspects):
    result_suspects = L.check("suspects", box_suspects.value)
    result_suspects
    return (result_suspects,)


@app.cell(hide_code=True)
def hints_suspects():
    L.hints(
        "suspects",
        [
            "Welke functie geeft posities in de volgorde van hun waarden, van klein naar groot?",
            "`np.argsort` geeft de posities van de laagste waarde naar de hoogste, gelijke in hun volgorde met `kind=\"stable\"`. Je hebt alleen het begin ervan nodig.",
            "`order = np.argsort(p, kind=\"stable\")` en `return order[___]`.",
        ],
        why="Bij het voorbeeld geeft `np.argsort` 3, 1, 5, 4, 0 en 2: eerst kans 0, dan 0,11 en 0,22. De eerste 3 zijn de verdachte labels.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_self():
    mo.md(r"""
    ### Oefening 3. Waarom zonder de pinguïn zelf
    """)
    return


@app.cell(hide_code=True)
def choice_self_cell():
    choice_self = L.choice(
        "self",
        question="Je geeft elke pinguïn de kans op zijn label met k = 1, van een k-NN die alle 274 pinguïns bewaarde, ook de pinguïn zelf. Wat krijg je?",
        options={
            "found": "0 voor elk verkeerd label en 1 voor elk goed label: zo vind je ze allemaal",
            "right": "1 voor elke pinguïn: zijn dichtstbijzijnde buur is hijzelf",
            "same": "Hetzelfde als out-of-fold: één pinguïn op 274 maakt weinig uit",
        },
        correct="right",
        why={
            "found": "Daarvoor zou een model de goede soort moeten kennen. Een k-NN die de pinguïn zelf bewaarde, vindt eerst de pinguïn zelf, op afstand 0, met het label dat hij heeft, fout of niet.",
            "same": "Met k = 1 beslist de dichtstbijzijnde buur alleen, en dat is de pinguïn zelf, op afstand 0. Out-of-fold is dat een andere pinguïn.",
            "right": "Goed: elke pinguïn is het met zichzelf eens, dus elke kans is 1 en zegt niets. De volgorde is dan gewoon die van de tabel: de eerste 30 controleren vindt 6 van de 41 verkeerde labels, ongeveer wat willekeurig controleren vindt (gemiddeld 4,5). Out-of-fold vindt k = 1 er 14 en k = 9 er 25.",
        },
    )
    choice_self
    return (choice_self,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Verkeerde labels glippen er zelfs met de goede richtlijnen uit [Goed labelen: richtlijnen en overeenstemming](/course/open/data-annotation?lang=nl) doorheen. De folds van [Betrouwbaarder kiezen: cross-validation](/course/open/ml-cross-validation?lang=nl) geven elke pinguïn een model dat hem nooit zag, en een lage kans is een grote verrassing, zoals bij [Kansen scoren met de log: log loss](/course/open/ml-log-loss?lang=nl). De data verbeteren is een manier om een model beter te maken, naast het model en zijn hyperparameters kiezen. De volgende les, *Data die je zelf maakt*, maakt nieuwe voorbeelden als er te weinig zijn.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_self, result_p_given, result_suspects):
    L.wrap_up(result_p_given, result_suspects, choice_self)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
