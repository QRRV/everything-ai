# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-max-margin. Route: course/route/lessen/klassiek-ml.yaml. Checks en beelden: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-max-margin", "nl")
    C = L.checks   # de Chinstraps en Gentoos uit ml-knn, hun straat voor elke C, margin en slacks, en de beelden


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl) en [Regularisatie en weight decay](/course/open/ml-regularization?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_boundary():
    quick_boundary = L.choice(
        "quick-boundary",
        question="Een model met twee features voorspelt klasse 1 waar $w_1 x_1 + w_2 x_2 + b$ boven 0 ligt, en klasse 0 waar het eronder ligt. Wat vormen de punten waar het precies 0 is?",
        options={
            "right": "Een rechte lijn: de decision boundary",
            "mistakes": "De voorbeelden uit de training set die het model fout heeft",
            "curve": "Een kromme die om de voorbeelden heen buigt",
        },
        correct="right",
        why={
            "mistakes": "Waar $w_1 x_1 + w_2 x_2 + b = 0$ is, hangt alleen af van $w$ en $b$, niet van de voorbeelden: het is een lijn, met aan zijn twee kanten een andere voorspelde klasse. Zie *Het idee* in [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl).",
            "curve": "In $w_1 x_1 + w_2 x_2 + b = 0$ staan geen kwadraten of producten van features: met twee features is het een rechte lijn. Zie *Het idee* in [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl).",
            "right": "Een rechte lijn, met klasse 1 aan de ene kant en klasse 0 aan de andere. Vandaag kies je welke rechte lijn: die met de meeste ruimte aan beide kanten.",
        },
    )
    quick_boundary
    return (quick_boundary,)


@app.cell(hide_code=True)
def quick_check_lambda():
    quick_lambda = L.choice(
        "quick-lambda",
        question="L2-regularisatie telt $\\lambda$ keer de som van de gekwadrateerde weights op bij de loss. Wat doet een grotere $\\lambda$?",
        options={
            "train": "Hij past het model beter aan de training set aan",
            "right": "Hij houdt de weights kleiner",
            "rate": "Hij maakt de stappen van gradient descent groter",
        },
        correct="right",
        why={
            "train": "De straf maakt de fit op de training set alleen maar slechter: het is de prijs van grote weights. Zie *Het idee* in [Regularisatie en weight decay](/course/open/ml-regularization?lang=nl).",
            "rate": "$\\lambda$ is niet de learning rate: hij bepaalt hoe zwaar grote weights meetellen in de loss. Zie *Het idee* in [Regularisatie en weight decay](/course/open/ml-regularization?lang=nl).",
            "right": "Grote weights kosten meer, dus blijven ze kleiner. Vandaag komt dezelfde som van gekwadrateerde weights terug: hoe kleiner die is, hoe breder de straat.",
        },
    )
    quick_lambda
    return (quick_lambda,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Op snavellengte en lichaamsgewicht kan een rechte lijn de 44 Chinstraps en de 66 Gentoos uit de training set scheiden zonder één fout. Dan zijn er talloze van zulke lijnen: draai er een een beetje, of schuif hem een beetje op, en hij scheidt ze nog steeds. Logistische regressie stopt bij een ervan, waar gradient descent ook eindigt. Welke neem je voor pinguïns die je nog niet gezien hebt? Die met de meeste ruimte aan beide kanten, zodat een nieuwe pinguïn die iets voorbij de andere van zijn soort ligt, toch aan de goede kant terechtkomt. En als een paar pinguïns die ruimte afknijpen, kan het de moeite waard zijn om ze toe te laten.

    **Een voorbeeld uit het dagelijks leven.** Op een smalle weg rijd je in het midden, zo ver mogelijk van beide randen, zodat je na een kleine uitwijking nog steeds op de weg zit.
    """)
    return


@app.cell(hide_code=True)
def predict_support():
    guess_support = L.guess(
        "support",
        question="Neem van alle rechte lijnen tussen de 44 Chinstraps en de 66 Gentoos uit de training set die met de breedste straat eromheen, zonder pinguïn op de straat. Hoeveel van de 110 pinguïns bepalen waar die straat ligt?",
        options={
            "all": "Alle 110: elke pinguïn trekt aan de lijn",
            "half": "Ongeveer de helft: die dicht bij de andere soort",
            "right": "3",
        },
        reveal="**3**: een Chinstrap van 46 mm en 4.150 g, en twee Gentoos, van 45,5 mm en 4.200 g en van 49,6 mm en 4.750 g. Ze liggen op de randen van de straat. Laat een van de andere 107 weg, en de straat blijft precies waar hij is.",
    )
    guess_support
    return (guess_support,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **De marge.** Een rechte lijn $w \cdot x + b = 0$ verdeelt het vlak, zoals de decision boundary van logistische regressie. Geef elke Chinstrap $y = -1$ en elke Gentoo $y = +1$. Dan is $y\,(w \cdot x + b)$ positief voor een voorbeeld aan zijn eigen kant van de lijn, en negatief aan de verkeerde kant; gedeeld door de lengte van $w$, $\lVert w \rVert$, is het hoe ver het voorbeeld van de lijn ligt. De **marge** van een lijn is de kleinste van die afstanden over de training set: de ruimte die het dichtstbijzijnde voorbeeld heeft. Ligt er een voorbeeld aan de verkeerde kant, dan is de marge negatief.

        **De breedste straat.** De **maximum-margin classifier** neemt van alle rechte lijnen die met de grootste marge. Denk aan een straat tussen de twee klassen, zo breed als kan zonder voorbeeld erop: de lijn is het midden ervan. De voorbeelden op de randen zijn de **support vectors**. Alleen zij bepalen waar de straat ligt: laat een ander voorbeeld weg, en de straat blijft hetzelfde.

        **Geschaald op 1.** Vermenigvuldig $w$ en $b$ met hetzelfde positieve getal en de lijn blijft waar hij is. Schaal ze dus tot $y\,(w \cdot x + b) = 1$ voor de support vectors. De randen van de straat zijn dan $w \cdot x + b = -1$ en $+1$, de marge is $1 / \lVert w \rVert$, en de straat is $2 / \lVert w \rVert$ breed. De breedste straat is de kleinste $\lVert w \rVert$, met $y\,(w \cdot x + b) \geq 1$ voor elk voorbeeld: de kleinste $\tfrac{1}{2} \lVert w \rVert^2$, de L2-straf van regularisatie.

        **De soft margin.** Hier knijpen drie pinguïns de straat af tot een breedte van 0,11, in gestandaardiseerde eenheden, en komt één Gentoo uit de validation set aan de kant van de Chinstraps terecht. Een **soft margin** laat voorbeelden op de straat toe, of zelfs over de lijn, tegen een prijs. De **slack variable** $\xi_i$, kortweg de slack, van voorbeeld $i$ zegt hoe ver het van zijn rand van de straat af blijft, geteld in marges, zodat 1 helemaal tot de lijn is: $\xi_i = 1 - y_i\,(w \cdot x_i + b)$, of 0 als dat onder 0 ligt. Hij is 0 buiten de straat aan de eigen kant, tussen 0 en 1 op de straat, en boven 1 aan de verkeerde kant van de lijn. De soft margin neemt de $w$ en $b$ met de kleinste
        """),
        L.formula(r"\tfrac{1}{2} \lVert [[param:w]] \rVert^2 + [[hyper:C]] \sum_{i} [[loss:\xi_i]]"),
        mo.md(r"""
        $C$ is een hyperparameter: de prijs van één eenheid slack. Een grote $C$ maakt elke slack duur, dus is de straat smal, met weinig voorbeelden erop; met een $C$ die groot genoeg is, is het de straat zonder voorbeeld erop. Een kleine $C$ maakt ruimte goedkoop: een brede straat met veel voorbeelden erop. Deel alles door $C$, en het is de som van de slacks plus een L2-straf op $w$ met $\lambda = 1 / (2C)$: een kleine $C$ is sterke regularisatie.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **marge** | | de kleinste afstand van de voorbeelden uit de training set tot de lijn, elk aan zijn eigen kant; negatief als er een aan de verkeerde kant ligt | `margin(w, b, X, y)` |
        | **support vector** | | een voorbeeld uit de training set op een rand van de straat of erop: alleen deze bepalen waar de straat ligt | |
        | **maximum-margin classifier** | | de rechte lijn met de grootste marge: het midden van de breedste straat | |
        | **soft margin** | | een straat die voorbeelden toelaat, erop of over de lijn, tegen een prijs per eenheid slack | |
        | **slack variable**, $\xi_i$ | ksi i | hoe ver voorbeeld $i$ van zijn rand van de straat af blijft, geteld in marges: 0 buiten de straat aan de eigen kant, 1 op de lijn | `slacks(w, b, X, y)` |
        | $C$ | C | de prijs van één eenheid slack: een hyperparameter | `C` |
        | `m[m > 2] = 2` | | toewijzen via een mask: elk getal van `m` waar het mask True is, wordt 2, de andere blijven | `m[m > 2] = 2` |
        """),
        L.watch_out("$C$ werkt andersom dan $\\lambda$: een grotere $C$ betekent een smallere straat en minder regularisatie. En net als bij k-NN is de marge een afstand: standaardiseer de features eerst, anders beslist het lichaamsgewicht in grammen bijna alleen."),
        mo.md(r"""
        Het beeld: de Chinstraps (driehoekjes) en de Gentoos (vierkantjes) naar snavellengte en lichaamsgewicht, de <span class="c-pred">lijn</span> van de breedste straat zonder pinguïn erop, en gestippeld de randen van de straat. Links de pinguïns uit de training set, met een ring om de drie support vectors; rechts die uit de validation set, met een <span class="c-loss">paarse ring</span> om die aan de verkeerde kant.
        """),
    ])
    return


@app.cell(hide_code=True)
def figure_street():
    mo.Html(C.street_picture(C.HARD, {"train": "Training set", "val": "Validation set", "x": "snavellengte (mm)", "y": "lichaamsgewicht (g)",
                                       "alt": "Twee grafieken van de Chinstraps en Gentoos naar snavellengte en lichaamsgewicht, met een rechte blauwe lijn die van linksonder naar rechtsboven loopt tussen de lichte Chinstraps en de zware Gentoos, en vlak ernaast twee gestippelde lijnen: een smalle straat. Links de training set: drie pinguïns op de randen van de straat zijn omcirkeld, een Chinstrap van 46 mm en 4.150 g en Gentoos van 45,5 mm en 4.200 g en van 49,6 mm en 4.750 g. Rechts de validation set: één Gentoo, van 48,7 mm en 4.450 g, is omcirkeld aan de kant van de Chinstraps."}))
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Verschuif de slider: hij zet $C$, van 0,01 tot 1.000, elke stap ongeveer drie keer de vorige, en de laatste tien keer. Het beeld toont de straat van de soft margin bij die $C$, zoals de course hem vindt met een solver die voor deze loss gemaakt is: links de pinguïns uit de training set, met een ring om elke pinguïn op de straat of op een rand; rechts die uit de validation set, met een paarse ring om elke pinguïn aan de verkeerde kant.
    """)
    return


@app.cell(hide_code=True)
def explore_c_cell():
    def _frame(_c):
        _f = C.facts(_c)
        _shown = C.c_text(_c, L.lang)
        _alt = (f"De straat van de soft margin met C = {_shown}, {L.number(_f['width'], 2)} breed in gestandaardiseerde eenheden. Links de training set: "
                f"{_f['edge']} pinguïns omcirkeld op de straat of de randen, waarvan {_f['wrong']} aan de verkeerde kant van de lijn. Rechts de validation set: "
                f"{_f['val_wrong']} van de {_f['n_val']} pinguïns aan de verkeerde kant.")
        _said = (f"Met $C$ = {_shown}: een straat van <span class=\"ea-num\">{L.number(_f['width'], 2)}</span> breed, in gestandaardiseerde eenheden. "
                 f"{_f['edge']} pinguïns uit de training set liggen erop of op de randen, waarvan {_f['wrong']} aan de verkeerde kant van de lijn. "
                 f"Op de validation set: <span class=\"c-loss ea-num\">{_f['val_wrong']}</span> van de {_f['n_val']} fout.")
        return C.street_picture(_c, {"train": "Training set", "val": "Validation set", "x": "snavellengte (mm)",
                                      "y": "lichaamsgewicht (g)", "alt": _alt}), _said

    explore_c = L.explore("c", "C", _frame, start=1, values=C.CS, role="hyper",
                          task="Zoek de waarden van $C$ zonder fout op de validation set. Hoe breed is de straat daar, tegenover $C$ = 1.000?")
    explore_c
    return (explore_c,)


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Met $C$ = 0,1, 0,3 en 3 is geen enkele pinguïn uit de validation set fout. Daarvan geeft $C$ = 0,1 de breedste straat: 1,34 breed, twaalf keer de 0,11 van $C$ = 1.000. Er liggen 38 pinguïns uit de training set op of op een rand, maar geen aan de verkeerde kant van de lijn. Met $C$ = 1.000 is het de straat zonder pinguïn erop, afgeknepen door drie pinguïns, en komt de Gentoo van 48,7 mm en 4.450 g uit de validation set aan de kant van de Chinstraps terecht. Een heel kleine $C$, 0,01, maakt de straat zo breed dat er 88 pinguïns uit de training set op liggen, 3 over de lijn, en dat 2 pinguïns uit de validation set fout zijn. Met maar 39 pinguïns in de validation set zegt één fout meer of minder weinig; maar de bredere straat kostte er geen enkele.

    ### Recept

    Zo vind je een lijn met ruimte aan beide kanten:

    1. Standaardiseer de features, en geef de ene klasse $y = -1$ en de andere $y = +1$.
    2. Voor een lijn $w$, $b$: hoe ver elk voorbeeld aan zijn eigen kant ligt, is `y * (X @ w + b)`; de marge is de kleinste daarvan, gedeeld door `np.linalg.norm(w)`.
    3. De slacks: `1 - y * (X @ w + b)`, met elke waarde onder 0 op 0 gezet.
    4. De soft margin neemt de $w$ en $b$ met de kleinste $\tfrac{1}{2} \lVert w \rVert^2 + C \cdot$ (de som van de slacks). Probeer een paar waarden van $C$, en kies op de validation set.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De pinguïns uit de les over k-NN, op dezelfde manier gesplitst en gestandaardiseerd. De cel houdt de Chinstraps en de Gentoos over, neemt de $w$ en $b$ van de breedste straat zonder pinguïn erop, zoals de course ze vond, en kijkt hoe ver elke pinguïn aan zijn eigen kant ligt. `(... == "Gentoo") * 2 - 1` geeft $+1$ voor een Gentoo en $-1$ voor een Chinstrap: True telt als 1 en False als 0. De opmerkingen met `#` zeggen welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_street():
    import numpy as np
    import pandas as pd

    # de pinguïns uit de les over k-NN, op dezelfde manier gesplitst en gestandaardiseerd
    _df = pd.read_csv("data/penguins.csv")
    _df = _df[_df["body_mass_g"].isna() == False]
    _X = np.array(_df[["bill_length_mm", "body_mass_g"]])
    _y = np.array(_df["species"])
    _order = np.random.default_rng(27).permutation(len(_df))
    _train, _val = _order[:206], _order[206:274]
    _mean, _std = _X[_train].mean(axis=0), _X[_train].std(axis=0)
    _X_train, _y_train = (_X[_train] - _mean) / _std, _y[_train]
    _X_val, _y_val = (_X[_val] - _mean) / _std, _y[_val]

    # 1. alleen de Chinstraps en de Gentoos: y = -1 voor een Chinstrap, +1 voor een Gentoo
    _keep, _keep_val = _y_train != "Adelie", _y_val != "Adelie"
    _X_cg, _X_cg_val = _X_train[_keep], _X_val[_keep_val]
    _t = (_y_train[_keep] == "Gentoo") * 2 - 1
    _t_val = (_y_val[_keep_val] == "Gentoo") * 2 - 1
    print(len(_t), len(_t_val))

    # 2. de breedste straat zonder pinguïn erop: hoe ver elke pinguïn aan zijn eigen kant ligt
    _w, _b = np.array([-12.307, 13.335]), 3.984
    _sides = _t * (_X_cg @ _w + _b)
    print(np.min(_sides), np.sum(_sides < 1.01))
    print(_X[_train][_keep][_sides < 1.01])
    print(1 / np.linalg.norm(_w), 2 / np.linalg.norm(_w))

    # dezelfde straat op de validation set: wie ligt aan de verkeerde kant?
    _sides_val = _t_val * (_X_cg_val @ _w + _b)
    print(np.sum(_sides_val < 0), _X[_val][_keep_val][_sides_val < 0])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    110 pinguïns in de training set en 39 in de validation set. De kleinste waarde van `_sides` is 0,9998: 1, zoals de schaling belooft, op de afronding van $w$ na; en precies 3 pinguïns zitten onder 1,01: de support vectors, een Gentoo van 49,6 mm en 4.750 g, een Chinstrap van 46 mm en 4.150 g en een Gentoo van 45,5 mm en 4.200 g. De marge is $1 / \lVert w \rVert$ = 0,055, de straat 0,110 breed. Op de validation set ligt één pinguïn aan de verkeerde kant: de Gentoo van 48,7 mm en 4.450 g, lichter dan de lijn bij zijn snavel verwacht.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_margin():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De marge van een lijn

        `margin(w, b, X, y)` moet de marge teruggeven van de lijn $w \cdot x + b = 0$ op de voorbeelden `X` met labels `y`, $-1$ of $+1$, zoals stap 2 van het recept. Vervang de twee `...`. De check roept `margin(np.array([1.0, 1.0]), 0.0, X, y)` aan met tien verzonnen pinguïns, al gestandaardiseerd: vijf Chinstraps en vijf Gentoos. Hij probeert ook dezelfde lijn met $w$ en $b$ drie keer zo groot, twee andere lijnen en drie metingen.
        """),
        L.task("Laat `margin` de afstand teruggeven van de lijn tot het dichtstbijzijnde voorbeeld, negatief als er een voorbeeld aan de verkeerde kant ligt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_margin():
    box_margin = L.code_box("margin", lines=6)
    box_margin
    return (box_margin,)


@app.cell(hide_code=True)
def check_margin(box_margin):
    result_margin = L.check("margin", box_margin.value)
    result_margin
    return (result_margin,)


@app.cell(hide_code=True)
def hints_margin():
    L.hints(
        "margin",
        [
            "Stap 2 van het recept, zoals in het uitgewerkte voorbeeld: eerst hoe ver elk voorbeeld aan zijn eigen kant ligt, dan de kleinste daarvan.",
            "Voor een voorbeeld aan zijn eigen kant is `y` keer zijn score positief. Welke functie van NumPy geeft daar de kleinste van? En wat maakt er een afstand van, zoals in *Geschaald op 1*?",
            "`sides = y * (X @ w + b)` en `return np.___(sides) / np.linalg.___(w)`.",
        ],
        why="Voor de tien verzonnen pinguïns is de kleinste waarde in `sides` 1,1, voor twee Chinstraps en een Gentoo; de lengte van `[1.0, 1.0]` is 1,414, dus de marge is 1,1 / 1,414 = 0,778.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_slacks():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De slacks

        `slacks(w, b, X, y)` moet de slack van elk voorbeeld teruggeven: hoe ver het van zijn rand van de straat af blijft, geteld in marges, en 0 buiten de straat aan de eigen kant, zoals stap 3 van het recept. Vervang de twee `...`: de eerste regel rekent $1 - y_i\,(w \cdot x_i + b)$ uit voor elk voorbeeld, de tweede maakt het mask van de waarden onder 0, en de regel daarna zet die op 0. De check roept `slacks(np.array([0.5, 0.5]), 0.0, X, y)` aan met dezelfde tien pinguïns, en probeert drie andere lijnen en drie metingen.
        """),
        L.task("Laat `slacks` één slack teruggeven voor elk voorbeeld."),
    ])
    return


@app.cell(hide_code=True)
def exercise_slacks():
    box_slacks = L.code_box("slacks", lines=8)
    box_slacks
    return (box_slacks,)


@app.cell(hide_code=True)
def check_slacks(box_slacks):
    result_slacks = L.check("slacks", box_slacks.value)
    result_slacks
    return (result_slacks,)


@app.cell(hide_code=True)
def hints_slacks():
    L.hints(
        "slacks",
        [
            "Stap 3 van het recept. Een voorbeeld precies op zijn rand heeft `y * (X @ w + b)` gelijk aan 1, en een slack van 0.",
            "Reken de formule voor alle voorbeelden tegelijk uit, zoals bij de marge: één getal voor elk. Dan kiest een mask de waarden onder 0, en één toewijzing via dat mask zet ze allemaal op 0, zoals in *Nieuwe woorden en tekens*.",
            "`s = 1 - y * (___)` en `to_zero = s ___ 0`.",
        ],
        why="Met `np.array([0.5, 0.5])` heeft de derde Chinstrap $y\\,(w \\cdot x + b)$ = 0,7: hij ligt op de straat, 0,3 marge van zijn rand af. De eerste ligt precies op zijn rand, op 1,0, en de vierde Gentoo erachter, op 1,1: allebei krijgen ze 0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_support():
    mo.md(r"""
    ### Oefening 3. Eén pinguïn minder
    """)
    return


@app.cell(hide_code=True)
def choice_support_cell():
    choice_support = L.choice(
        "not-a-support-vector",
        question="De breedste straat zonder pinguïn erop heeft drie support vectors. Je laat een van de andere 107 pinguïns uit de training set weg, en zoekt opnieuw de breedste straat. Wat gebeurt er?",
        options={
            "moves": "Hij verschuift een beetje: elke pinguïn trekt aan de lijn",
            "right": "Hij blijft precies waar hij is",
            "narrower": "Hij wordt smaller: er is een pinguïn minder om hem open te houden",
        },
        correct="right",
        why={
            "moves": "Zo werkt logistische regressie: elk voorbeeld telt mee in de loss. De breedste straat voelt alleen de pinguïns op zijn randen. Zie *De breedste straat* in *Het idee*.",
            "narrower": "De pinguïns buiten de straat houden hem niet open; de straat was zo breed als de drie support vectors toelieten. Zonder een van de andere is hij nog steeds de breedste. Zie *De breedste straat* in *Het idee*.",
            "right": "De drie support vectors knijpen hem nog op dezelfde plek af, en geen andere pinguïn stond in de weg: de straat blijft. Je kunt de andere 107 weglaten en vindt uit alleen de drie dezelfde straat.",
        },
    )
    choice_support
    return (choice_support,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De lijn is de decision boundary uit [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl); maar waar bij logistische regressie elk voorbeeld meetelt in de loss, voelt de breedste straat alleen zijn support vectors. De afstand tot de lijn gebruikt de lengte van een vector uit [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl), en $\tfrac{1}{2} \lVert w \rVert^2$ is de L2-straf uit [Regularisatie en weight decay](/course/open/ml-regularization?lang=nl), met $C$ in de rol van $1 / (2\lambda)$. In de volgende les, *Veel klassen uit twee: one-vs-rest en one-vs-one*, maak je van een classifier voor twee klassen, zoals deze, een voor alle drie de soorten.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_support, result_margin, result_slacks):
    L.wrap_up(result_margin, result_slacks, choice_support)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
