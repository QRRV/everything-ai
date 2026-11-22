# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-xai-tools. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-xai-tools", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Shapley-waarden](/course/open/rai-shapley-values?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sum():
    quick_sum = L.choice(
        "quick-sum",
        question="In de vorige les telden de vier Shapley-waarden van de persoon van 41 op tot 0,226. Wat is 0,226?",
        options={
            "both": "Wat education_num en age samen bijdragen aan de kans",
            "right": "De kans min het gemiddelde van de achtergrond",
            "chance": "De kans op ja van deze persoon",
        },
        correct="right",
        why={
            "right": "Goed: 0,451 min 0,225. Dat is de efficiëntie van Shapley, en elke explainer van shap in deze les houdt die vast: wat hij ook trekt of fit, zijn waarden tellen op tot de voorspelling min het gemiddelde over zijn achtergrond.",
            "both": "education_num en age krijgen +0,181 en +0,087, samen 0,268, en hours_per_week en capital_gain halen er 0,042 af. 0,226 is de kans van de persoon, 0,451, min het gemiddelde van de achtergrond, 0,225. Zie [Shapley-waarden](/course/open/rai-shapley-values?lang=nl).",
            "chance": "De kans van deze persoon is 0,451. De vier waarden verdelen het verschil tussen die kans en het gemiddelde van de achtergrond, 0,225: 0,226. Zie [Shapley-waarden](/course/open/rai-shapley-values?lang=nl).",
        },
    )
    quick_sum
    return (quick_sum,)


@app.cell(hide_code=True)
def quick_check_game():
    quick_game = L.choice(
        "quick-game",
        question="Tegenover 200 mensen met een education_num van 13 of meer kreeg education_num −0,027 in plaats van +0,181. Waarom?",
        options={
            "random": "Elke run trekt nieuwe willekeurige volgordes",
            "trained": "Het forest werd op die mensen getraind",
            "right": "De achtergrond hoort bij het spel",
        },
        correct="right",
        why={
            "right": "Goed: dezelfde persoon en hetzelfde forest, vergeleken met andere mensen. In de tools van deze les is de achtergrond één argument, en sommige maken hem kleiner met hoogstens een logbericht, of spelen een ander spel als je hem weglaat.",
            "random": "De vorige les liep alle 24 volgordes langs, dus er zat geen toeval meer in toen de achtergrond eenmaal getrokken was. Tegenover mensen met 13 of meer is 13 het laagst: de achtergrond hoort bij het spel. Zie [Shapley-waarden](/course/open/rai-shapley-values?lang=nl).",
            "trained": "Het forest bleef hetzelfde; alleen de rijen waaruit de onbekende kolommen worden ingevuld, veranderden. Tegenover mensen met 13 of meer is 13 het laagst. Zie [Shapley-waarden](/course/open/rai-shapley-values?lang=nl).",
        },
    )
    quick_game
    return (quick_game,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De bank wil het forest op alle 14 kolommen van UCI Adult trainen en iedereen in de testhelft, 24.421 mensen, Shapley-waarden geven. Precies uitgerekend, zoals in de vorige les, zijn dat $2^{14}$ = 16.384 coalities voor één persoon, en met 200 rijen in de achtergrond moet het forest voor die ene persoon 3.276.800 rijen een kans geven. Dus pak je een library. shap en InterpretML doen wat jij met de hand bouwde, veel sneller, DiCE doet wat de volgende les bouwt, en alle drie maken ze keuzes voor je: uit welke rijen ze invullen, uit hoeveel, welk spel, hoeveel steekproeven. Sommige van die keuzes staan nergens in de uitvoer, en elke keuze kan het antwoord veranderen.

    **Een voorbeeld uit het dagelijks leven.** Een camera op de automatische stand kiest de sluitertijd, de scherpte en het licht voor je. Twee camera's op automatisch maken twee verschillende foto's van dezelfde straat, en als een foto te donker is, zoek je op wat de camera koos. Doe hetzelfde met een library voor uitleg: zoek uit wat hij koos voordat je zijn getallen vertrouwt.
    """)
    return


@app.cell(hide_code=True)
def predict_kept():
    guess_kept = L.guess(
        "kept",
        question="Je geeft `shap.Explainer` de hele helft om op te trainen, 24.421 rijen, als achtergrond. Hoeveel daarvan gebruikt hij om de onbekende kolommen in te vullen?",
        options={
            "all": "Alle 24.421",
            "thousand": "1000",
            "hundred": "100",
            "ten": "10",
        },
        reveal="**100.** shap maakt van een tabel die je als achtergrond geeft een masker, `maskers.Independent(background)`, met `max_samples=100`: heeft de data meer rijen dan `max_samples`, dan neemt `shap.utils.sample` er een steekproef uit. Die trekt 100 rijen zonder terugleggen en met `random_state=0`, dus elke keer dezelfde 100, en shap meldt dat alleen in een logbericht. Dit is gelezen in de code van shap 0.54.1; de course draait shap niet.",
    )
    guess_kept
    return (guess_kept,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Drie libraries.** Deze les zet de aanroepen van drie libraries naast wat jij bouwde, en voert ze niet uit: de course installeert ze niet, en de getallen in de cellen komen uit de code van de course, uitgerekend zoals jij dat met de hand deed. De les is geschreven voor shap 0.54.1 (oktober 2026), InterpretML 0.7.8 (maart 2026; de packages `interpret` en `interpret-core`) en dice-ml 0.12 (juli 2025). Noem de versie als je een waarde rapporteert: shap veranderde standaardinstellingen in versie 0.47, en code uit een oudere tutorial kan andere getallen geven.

        **shap en zijn achtergrond.** De gewone aanroep:

        ```python
        import shap

        explainer = shap.Explainer(model, background)   # model(rows): de kans op ja voor elke rij
        explanation = explainer(rows)                   # rows: de mensen die je uitlegt
        explanation.base_values                         # het gemiddelde antwoord over de achtergrond
        explanation.values                              # een Shapley-waarde per persoon en kolom
        ```

        De achtergrond is waaruit je in de vorige les invulde. Een tabel wordt daar een **masker**, het object van shap dat de onbekende kolommen verbergt door ze in te vullen uit rijen van de achtergrond, en dat masker houdt 100 rijen over, zoals bij de gok. De documentatie geeft een vuistregel: meestal zijn ongeveer 1, 10, 100 of 1000 rijen in de achtergrond een redelijke keuze. Daarna kiest `algorithm="auto"` hoe er gerekend wordt. Voor een model dat gewoon een functie is, kiest hij `"exact"` tot 10 kolommen en `"permutation"` daarboven, dus dezelfde aanroep draait op 8 kolommen van Adult het ene algoritme en op 14 het andere. Met de vier kolommen uit de vorige les zou shap doen wat jij deed, volgens zijn code over 100 van je 200 rijen. Welke 100 het zijn, maakt uit: bij 10 trekkingen van 100 van die 200 rijen krijgt education_num van +0,164 tot +0,196, tegen +0,181 met alle 200. De schuifbalk hieronder trekt achtergronden van andere groottes.

        **Willekeurige volgordes.** Boven 10 kolommen neemt `PermutationExplainer` van shap willekeurige volgordes, zoals Štrumbelj en Kononenko deden, en middelt wat elke kolom toevoegt als die aanschuift. Volgens zijn code garandeert hij local accuracy doordat hij elke volgorde helemaal doorloopt, vooruit en achteruit (antithetic sampling): elke volgorde draait ook achterstevoren, en in elke volgorde verdelen de kolommen het hele verschil, dus de schattingen tellen altijd op. Zijn `max_evals`, hoe vaak hij het model iets mag vragen, is standaard 500. De volgordes zijn willekeurig, dus twee runs geven net andere getallen: met 10 willekeurige volgordes voor de persoon uit de vorige les krijgt education_num in 10 runs van +0,157 tot +0,205. Deze methode schrijf je in oefening 1.

        **KernelSHAP: LIME met de Shapley-kernel.** Lundberg en Lee lieten zien dat een lijn zoals die van LIME de Shapley-waarden geeft als je de onderdelen op hun manier kiest. Schrijf elke coalitie als een rij van nullen en enen, één voor elke kolom, en fit een rechte lijn van die rijen naar de waarden van de coalities, zonder straf op de weights. Laat een coalitie met $s$ van de $M$ kolommen bekend meetellen met de **Shapley-kernel**

        $$\pi(s) = \frac{M - 1}{\binom{M}{s}\, s\, (M - s)}.$$

        Voor de lege en de volle coalitie is dat oneindig: die moeten precies uitkomen, en daardoor begint de lijn bij het gemiddelde van de achtergrond en eindigt hij bij de voorspelling. Dan kun je, schrijven ze, de Shapley-waarden uit de speltheorie uitrekenen met gewogen lineaire regressie. Dat is **KernelSHAP**:

        ```python
        explainer = shap.KernelExplainer(model, background)
        values = explainer.shap_values(rows, nsamples="auto", l1_reg="num_features(10)")   # de standaard
        ```

        Over de achtergrond zegt de documentatie: bij kleine problemen kan dat de hele data zijn waarop je trainde, maar overweeg bij grotere problemen één referentiewaarde, of vat de data samen met de functie `kmeans`. Met `nsamples="auto"` vraagt hij om $2M + 2048$ coalities, maar nooit om meer dan de $2^M - 2$ tussen de lege en de volle: tot 11 kolommen zijn dat ze allemaal; boven 11 trekt hij een steekproef. Met jouw vier kolommen neemt hij alle 14. Twee standaardinstellingen veranderen het antwoord zonder het te zeggen. Sinds versie 0.47 kiest `l1_reg="num_features(10)"` een vast aantal features met het meeste gewicht: een feature selection vóór de lijn houdt hoogstens ongeveer 10 kolommen over, en de rest krijgt 0. En de coalities worden getrokken met de toevalsgetallen van NumPy, zonder seed die je kunt meegeven, dus twee runs op veel kolommen verschillen een beetje. I. Elizabeth Kumar en anderen rekenen KernelSHAP tot de interventional methoden: hij vult in uit rijen zoals ze zijn, zoals jij deed. Oefening 2 is de kernel.

        **TreeSHAP: uit de bomen zelf.** Voor forests en boosted trees heeft shap `TreeExplainer`. Lundberg en anderen noemen als eerste bijdrage van hun paper een algoritme dat in polynomiale tijd optimale uitleg uitrekent, gebaseerd op de speltheorie. Volgens hun preprint uit 2019 groeit de tijd met het aantal bomen, hun bladeren en hun diepte, niet met $2^M$; met data groeit hij ook met het aantal rijen in de achtergrond. **TreeSHAP** kan twee spellen spelen:

        ```python
        shap.TreeExplainer(forest)                     # geen data: "tree_path_dependent"
        shap.TreeExplainer(forest, data=background)    # data: "interventional", over 100 van de rijen
        ```

        De standaard, `feature_perturbation="auto"`, kiest "interventional" als je data meegeeft: het spel van de vorige les. Zonder data is er volgens de documentatie geen achtergrond nodig: hij volgt de bomen, en laat het aantal trainingsvoorbeelden dat in elk blad terechtkwam voor de achtergrond staan. Kumar en anderen rekenen TreeSHAP van die soort tot de conditional methoden. De standaard is verschoven: de preprint van Lundberg en anderen uit 2019 zegt dat TreeExplainer standaard voorwaardelijke verwachtingen uitrekent door de bomen af te lopen, de documentatie van shap zegt dat de standaard in 0.47 veranderde van "interventional" naar "auto", en in 0.54.1 hangt hij ervan af of je data meegeeft. Het uitgewerkte voorbeeld speelt beide spellen met de hand.

        **InterpretML: een model dat zichzelf uitlegt.** InterpretML bouwt de Explainable Boosting Machine uit [Additieve modellen](/course/open/rai-additive-models?lang=nl):

        ```python
        from interpret import show
        from interpret.glassbox import ExplainableBoostingClassifier

        ebm = ExplainableBoostingClassifier()          # standaard onder meer interactions="3x", random_state=42
        ebm.fit(X_train, y_train)
        show(ebm.explain_global())                     # de kromme van elke term
        show(ebm.explain_local(X_test[:5], y_test[:5]), 0)
        ebm.eval_terms(X_test[:5])                     # dezelfde waarden als de local explanation, als getallen
        ```

        `interactions="3x"` vraagt om drie keer zoveel paren van kolommen als er kolommen zijn: met de 14 kolommen van Adult 42 paren. `eval_terms` geeft per term dezelfde scores als de local explanation van `ebm.explain_local(x)`. Bij het uitleggen vult een EBM niets in en trekt hij geen steekproef: zijn local explanation lees je af aan zijn eigen termen, dus hij heeft geen achtergrond nodig en geeft elke keer dezelfde getallen. Wat hem verandert, is het model: een andere `random_state` of andere `interactions` traint een andere EBM. InterpretML verpakt ook shap: `interpret.blackbox.ShapKernel` roept KernelExplainer aan.

        **DiCE: wat er zou moeten veranderen.** DiCE, van Ramaravind K. Mothilal, Amit Sharma en Chenhao Tan, zoekt rijen dicht bij de persoon die het andere antwoord krijgen, het onderwerp van de volgende les:

        ```python
        import dice_ml

        data = dice_ml.Data(dataframe=train_table, continuous_features=["age", "hours_per_week"],
                            outcome_name="income")
        model = dice_ml.Model(model=forest, backend="sklearn")    # zonder backend: "TF1", TensorFlow 1
        dice = dice_ml.Dice(data, model, method="random")         # de standaardmethode
        found = dice.generate_counterfactuals(person, total_CFs=4, desired_class="opposite",
                                              features_to_vary=["education_num", "hours_per_week"])
        found.visualize_as_dataframe(show_only_changes=True)
        ```

        Drie keuzes veranderen wat hij vindt. `method` kiest de zoektocht: "random", "genetic", "kdtree" of "gradient", en de willekeurige begint met `random_seed=None`, dus elke run een nieuwe set. De instellingen voor nabijheid en verscheidenheid, zoals `proximity_weight`, worden genegeerd door de methoden "random" en "kdtree", ook al geeft de README van DiCE ze zelf mee aan de willekeurige methode (volgens de code stopt die er dan waarschijnlijk met een fout). En `features_to_vary="all"`, de standaard, laat DiCE alles veranderen: in het eigen voorbeeld van de paper voor Adult verandert één van vier antwoorden de opleiding in Masters, de leeftijd in 65, de burgerlijke staat in getrouwd en het geslacht in man.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **masker** | het object van shap dat de onbekende kolommen verbergt door ze in te vullen uit rijen van de achtergrond |
        | **KernelSHAP** | de manier van Lundberg en Lee om Shapley-waarden uit te rekenen voor elk model: een gewogen lineaire regressie op de coalities |
        | **Shapley-kernel** | het gewicht van een coalitie in die regressie; oneindig voor de lege en de volle coalitie |
        | **TreeSHAP** | Shapley-waarden voor bomen in polynomiale tijd, uit de bouw van de bomen |
        | **InterpretML** | de library met de Explainable Boosting Machine, en verpakkingen om andere explainers heen |
        | $M$ | het aantal kolommen dat varieert |
        | $\binom{M}{s}$ | het aantal coalities met $s$ van de $M$ kolommen bekend; in Python `math.comb(M, s)` |
        """),
        L.watch_out("De standaardinstellingen van een library zijn keuzes: welke achtergrond en hoeveel rijen, welk spel, hoeveel steekproeven, welke kolommen een waarde mogen krijgen. Zet de versie en die instellingen bij elke waarde die je rapporteert: twee aanroepen die in één ervan verschillen, kunnen voor hetzelfde model en dezelfde persoon andere waarden geven."),
    ])
    return


@app.cell(hide_code=True)
def explore_size_cell():
    explore_size = L.explore("size", "rijen in elke achtergrond", lambda v: C.size_picture(v, "nl"), start=100,
                             values=C.SIZES, role="hyper",
                             task="Schuif omlaag naar 1 en dan omhoog naar 1000. Hoe ver liggen de tien waarden van education_num uit elkaar bij elke grootte?")
    mo.vstack([
        mo.md("**Hoeveel rijen?**"),
        explore_size,
        mo.md(r"""
        De exacte Shapley-waarden van de course voor de persoon van 41, elk punt met een eigen achtergrond van zoveel rijen als de schuifbalk aangeeft, uit de helft om op te trainen, getrokken met de seeds 0 tot 9. Met één rij is de achtergrond één andere persoon, en zeggen de waarden hoe deze persoon van die ene verschilt: 2 van de 10 hadden een capital gain, van 7298 en 99.999 dollar, en capital_gain krijgt dan −0,458 en −0,531. Met 1000 rijen zijn age, hours_per_week en capital_gain het bijna eens, en geven acht van de tien trekkingen education_num tussen +0,155 en +0,162, onder de +0,181 van de 200 rijen uit de vorige les; de andere twee, +0,147 en +0,172, liggen bijna even ver uit elkaar als de uitersten bij 100 rijen. Elk punt is exact voor zijn eigen achtergrond; wat beweegt, is het spel.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo gebruik je een library voor uitleg:

    1. Schrijf de library en zijn versie op, zoals shap 0.54.1: standaardinstellingen veranderen tussen versies.
    2. Zeg welk spel hij speelt: uit welke rijen hij invult en uit hoeveel (een tabel als achtergrond van `shap.Explainer` of `TreeExplainer` wordt volgens de code hoogstens 100 rijen; `KernelExplainer` gebruikt elke rij), of, bij bomen, of je data meegaf.
    3. Zeg hoe hij rekent: exact, uit getrokken volgordes, of met een regressie op getrokken coalities. Zet de seed vast waar die er is, en draai hem twee keer waar die er niet is.
    4. Controleer dat de waarden optellen tot de voorspelling min de basiswaarde.
    5. Zoek standaardinstellingen die het antwoord veranderen, zoals `l1_reg` in KernelSHAP en `features_to_vary` in DiCE.
    6. Vergelijk het op een paar kolommen met je eigen exacte berekening.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het forest, de persoon en de 200 rijen van de achtergrond uit de vorige les, in de twee spellen van `TreeExplainer`. Een gefitte boom van scikit-learn bewaart zijn bouw in `tree_`: voor elke knoop geven `children_left` en `children_right` de twee knopen eronder (−1 bij een blad), `feature` en `threshold` geven zijn vraag, `value[node, 0, 1] / value[node, 0].sum()` is zijn deel ja, en `weighted_n_node_samples` zegt hoeveel trainingsrijen er aankwamen, elk zo vaak geteld als de eigen steekproef van de boom hem trok. `_down` roept zichzelf aan voor de knoop eronder, en loopt zo de boom af tot hij bij een blad komt. `_forest.estimators_` bevat de 100 bomen. Voer de cel uit; dat duurt een paar seconden.
    """)
    return


@app.cell
def example_trees():
    import itertools
    import math

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier

    # 1. het forest, de persoon en de achtergrond uit de vorige les
    _people = pd.read_csv("data/adult.csv.gz")
    _columns = ["age", "education_num", "hours_per_week", "capital_gain"]
    _X = _people[_columns].to_numpy().astype(float)
    _y = (_people["income"] == ">50K").to_numpy()
    _order = np.random.default_rng(0).permutation(len(_y))
    _train, _test = _order[: len(_y) // 2], _order[len(_y) // 2:]
    _forest = RandomForestClassifier(min_samples_leaf=20, random_state=0).fit(_X[_train], _y[_train])
    _person = _X[_test][5]
    _background = _X[_train][np.random.default_rng(0).choice(len(_train), 200, replace=False)]
    print("kans op ja voor deze persoon:", _forest.predict_proba([_person])[0, 1].round(3))


    # 2. met data: vul de onbekende kolommen in uit de 200 rijen, zoals in de vorige les
    def _with_data(known):
        _rows = np.where(np.array(known) == 1, _person, _background)
        return _forest.predict_proba(_rows)[:, 1].mean()


    # 3. zonder data: loop een boom af; neem bij een vraag over een onbekende kolom beide takken,
    #    elk geteld naar hoeveel trainingsrijen die kant op gingen
    def _down(tree, node, known):
        if tree.children_left[node] == -1:
            return tree.value[node, 0, 1] / tree.value[node, 0].sum()     # een blad: zijn deel ja
        _left, _right = tree.children_left[node], tree.children_right[node]
        _column = tree.feature[node]
        if known[_column] == 1:
            if _person[_column] <= tree.threshold[node]:
                return _down(tree, _left, known)
            return _down(tree, _right, known)
        _n = tree.weighted_n_node_samples
        return (_n[_left] * _down(tree, _left, known) + _n[_right] * _down(tree, _right, known)) / _n[node]


    def _without_data(known):
        _total = 0
        for _tree in _forest.estimators_:
            _total = _total + _down(_tree.tree_, 0, known)
        return _total / len(_forest.estimators_)


    # 4. de Shapley-waarden van een spel, met de formule uit de vorige les
    def _shapley(v, i):
        _total = 0
        for _known in v:
            if _known[i] == 0:
                _with_i = list(_known)
                _with_i[i] = 1
                _s, _n = sum(_known), len(_known)
                _weight = math.factorial(_s) * math.factorial(_n - _s - 1) / math.factorial(_n)
                _total = _total + _weight * (v[tuple(_with_i)] - v[_known])
        return _total


    for _name, _game in [("met data:   ", _with_data), ("zonder data:", _without_data)]:
        _v = {}
        for _known in itertools.product([0, 1], repeat=4):
            _v[_known] = _game(_known)
        _line = _name + " gemiddelde " + f"{_v[(0, 0, 0, 0)]:.3f}"
        for _i in range(4):
            _line = _line + "  " + _columns[_i] + f" {_shapley(_v, _i):+.3f}"
        print(_line)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Beide spellen eindigen bij de eigen kans van de persoon, 0,451, maar ze beginnen ergens anders: bij het gemiddelde van het forest over de 200 rijen, 0,225, of bij het gemiddelde van de bladeren van de bomen naar hun trainingsrijen, 0,238, dicht bij het deel ja in de helft om op te trainen, 0,237. education_num en age bewegen nauwelijks; hours_per_week krijgt −0,026 in plaats van −0,010, ruim twee keer zoveel. Geen van beide is fout: ze beantwoorden verschillende vragen. Met data is de vraag wat het weten van een kolom toevoegt tegenover deze 200 mensen; zonder data wat het toevoegt langs de paden van de bomen, waar de persoon bij een vraag over een onbekende kolom de kanten op gaat die de trainingsrijen in die knoop op gingen. TreeSHAP vindt de waarden van dat tweede spel door elke boom één keer af te lopen, zonder de 16 coalities langs te gaan; deze cel speelt het met de hand. (De print schrijft de getallen met een punt, zoals Python dat doet.)
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sampled():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Willekeurige volgordes

        Schrijf `sampled(v, i, orders)`, de permutatiemethode: `v` is een dict van elke coalitie, een tuple met nullen en enen, naar zijn waarde, `i` is de kolom, en `orders` is een lijst met volgordes, elk een tuple met nummers van kolommen, elke kolom één keer, in de volgorde waarin ze aanschuiven: `(1, 0)` betekent dat kolom 1 als eerste aanschuift. Zet `total` eerst op 0. Begin voor elke `order` in `orders` met `known` vol nullen, `[0] * len(order)`, en laat de kolommen `j` van de volgorde een voor een aanschuiven: neem `before`, de waarde van `tuple(known)`, zet `known[j]` op 1, en als `j` gelijk is aan `i`, tel dan de waarde van `tuple(known)` min `before` op bij `total`. Geef het gemiddelde terug, `total` gedeeld door het aantal volgordes, `len(orders)`. Een volgorde kan twee keer voorkomen, zoals bij willekeurig trekken. De check roept eerst `sampled({(0, 0): 0, (1, 0): 2, (0, 1): 4, (1, 1): 10}, 0, [(1, 0), (1, 0), (0, 1)])` aan, de pubquiz uit de vorige les met drie getrokken volgordes: Ash hoort (6 + 6 + 2) / 3 te krijgen. De tekening voert je functie uit op de 16 waarden uit de vorige les, met de eerste 1 tot 100 van 100 willekeurige volgordes.
        """),
        L.task("Schrijf `sampled(v, i, orders)`: laat voor elke volgorde de kolommen een voor een aanschuiven, tel bij `total` op wat de waarde erbij krijgt als kolom `i` aanschuift, en geef het gemiddelde over de volgordes terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sampled():
    box_sampled = L.code_box("sampled", lines=12)
    box_sampled
    return (box_sampled,)


@app.cell(hide_code=True)
def check_sampled(box_sampled):
    result_sampled = L.check("sampled", box_sampled.value)
    result_sampled
    return (result_sampled,)


@app.cell(hide_code=True)
def hints_sampled():
    L.hints(
        "sampled",
        [
            "Elke volgorde is een klein verhaal: er is niets bekend, en de kolommen schuiven een voor een aan. Alleen de stap waarin kolom `i` aanschuift telt, en die telt wat de waarde bij die stap erbij krijgt.",
            "Begin binnen `for order in orders:` met `known = [0] * len(order)`, en neem binnen `for j in order:` eerst `before = v[tuple(known)]` en zet daarna `known[j] = 1`; als `j == i`, tel dan `v[tuple(known)] - before` op bij `total`.",
            "`return total / ___`",
        ],
        why="`total = 0`, dan `for order in orders:` met `known = [0] * len(order)` en `for j in order:`: `before = v[tuple(known)]`, `known[j] = 1`, en `if j == i:` `total = total + v[tuple(known)] - before`. Na beide loops `return total / len(orders)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_kernel():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De Shapley-kernel

        Schrijf `kernel_weight(m, s)`, de Shapley-kernel van Lundberg en Lee voor een coalitie met `s` van de `m` kolommen bekend: $m - 1$ gedeeld door $\binom{m}{s}\, s\, (m - s)$. `math.comb(m, s)` geeft $\binom{m}{s}$, het aantal coalities met `s` van de `m` kolommen. Voor de lege en de volle coalitie, met `s` gelijk aan 0 of `m`, deelt de formule door 0: geef daar `math.inf` terug, oneindig in Python. De check roept eerst `kernel_weight(4, 1)` aan: 3 / (4 · 1 · 3) = 0,25. De tekening zet je gewicht uit voor 4 en voor 14 kolommen.
        """),
        L.task("Schrijf `kernel_weight(m, s)`: `math.inf` voor de lege en de volle coalitie, en anders (m − 1) gedeeld door `math.comb(m, s) * s * (m - s)`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_kernel():
    box_kernel = L.code_box("kernel", lines=8)
    box_kernel
    return (box_kernel,)


@app.cell(hide_code=True)
def check_kernel(box_kernel):
    result_kernel = L.check("kernel", box_kernel.value)
    result_kernel
    return (result_kernel,)


@app.cell(hide_code=True)
def hints_kernel():
    L.hints(
        "kernel",
        [
            "Twee gevallen: de lege en de volle coalitie wegen oneindig zwaar, en elke andere coalitie krijgt de formule.",
            "Eerst `if s == 0 or s == m:` met `return math.inf`. Daaronder de formule: `m - 1` boven de streep, en eronder het aantal coalities, `math.comb(m, s)`, keer `s` keer `m - s`.",
            "`return (m - 1) / (___ * s * (m - s))`",
        ],
        why="`if s == 0 or s == m:` `return math.inf`; dan `return (m - 1) / (math.comb(m, s) * s * (m - s))`.",
    )
    return


@app.cell(hide_code=True)
def text_after_kernel():
    mo.md(r"""
    De kernel laat de coalities met weinig of met veel bekende kolommen het zwaarst wegen: met 4 kolommen 0,25, 0,125 en 0,25, en met 14 kolommen telt een coalitie met 1 bekende 924 keer zo zwaar als een met 7. Met dit gewicht geeft de regressie op de 14 coalities van vier kolommen, tussen geen en alle, precies de vier Shapley-waarden van de vorige les. Met 14 kolommen zijn er 16.382 van zulke coalities, en neemt KernelSHAP er standaard 2076: dan trekt hij een steekproef.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_data():
    mo.md(r"""
    ### Oefening 3. Met data of zonder

    In het uitgewerkte voorbeeld gaf hetzelfde forest dezelfde persoon twee rijtjes waarden.
    """)
    return


@app.cell(hide_code=True)
def choice_data_cell():
    choice_data = L.choice(
        "data",
        question="Twee aanroepen leggen hetzelfde forest uit voor dezelfde persoon: `shap.TreeExplainer(forest)` en `shap.TreeExplainer(forest, data=background)`. Wat is er anders?",
        options={
            "right": "Het spel, en daarmee de waarden",
            "exact": "Niets: TreeSHAP is in beide gevallen exact",
            "speed": "Alleen hoe snel de waarden er zijn",
        },
        correct="right",
        why={
            "right": "Goed. Zonder data volgt TreeExplainer de bomen en laat hij de trainingsrijen die elke tak op gingen voor de achtergrond staan; met data vult hij in uit de rijen die je geeft, volgens zijn code 100 ervan. In het uitgewerkte voorbeeld met de hand: een gemiddelde van 0,238 tegen 0,225, en hours_per_week −0,026 tegen −0,010.",
            "exact": "TreeSHAP is exact voor het spel dat hij speelt, en de twee aanroepen spelen verschillende spellen: de eigen trainingsrijen van de bomen, of de rijen die je geeft. In het uitgewerkte voorbeeld met de hand krijgt hours_per_week −0,026 zonder data en −0,010 met. Zie *TreeSHAP* in *Het idee*.",
            "speed": "Ze kunnen verschillend lang duren, maar ook de waarden verschillen, omdat het spel verschilt: zonder data staan de eigen trainingsrijen van de bomen voor de achtergrond. In het uitgewerkte voorbeeld met de hand krijgt hours_per_week −0,026 zonder data en −0,010 met.",
        },
    )
    choice_data
    return (choice_data,)


@app.cell(hide_code=True)
def text_exercise_ten():
    mo.md(r"""
    ### Oefening 4. Welke kolommen krijgen een waarde?

    Een standaardinstelling van KernelSHAP bepaalt hoeveel kolommen überhaupt een waarde kunnen krijgen.
    """)
    return


@app.cell(hide_code=True)
def choice_ten_cell():
    choice_ten = L.choice(
        "ten",
        question="Je legt één persoon uit met `shap.KernelExplainer` op een tabel met 30 kolommen, met de standaardinstellingen van shap 0.54.1. Hoeveel kolommen kunnen een waarde krijgen die niet 0 is?",
        options={
            "all": "Alle 30, elk met zijn eigen deel",
            "right": "Hoogstens ongeveer 10",
            "above": "Alleen die boven 0,01",
        },
        correct="right",
        why={
            "right": "Goed. Sinds versie 0.47 is de standaard `l1_reg=\"num_features(10)\"`, die een vast aantal features met het meeste gewicht kiest: een feature selection vóór de regressie houdt hoogstens ongeveer 10 kolommen over, en de rest krijgt 0 (gelezen in de code van shap, niet uitgevoerd). Als elke kolom ertoe doet, zet `l1_reg` dan zelf en zeg wat je koos.",
            "all": "Niet met de standaardinstellingen van 0.54.1: sinds 0.47 houdt `l1_reg=\"num_features(10)\"` hoogstens ongeveer 10 kolommen over, en de rest krijgt 0. Zie *KernelSHAP* in *Het idee*.",
            "above": "Er is geen grens bij 0,01. De standaard `l1_reg=\"num_features(10)\"` houdt met een feature selection hoogstens ongeveer 10 kolommen over, en de rest krijgt 0. Zie *KernelSHAP* in *Het idee*.",
        },
    )
    choice_ten
    return (choice_ten,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De les [Shapley-waarden](/course/open/rai-shapley-values?lang=nl) rekende ze precies uit voor vier kolommen, over alle 16 coalities. De libraries komen er sneller, met getrokken volgordes, met een gewogen regressie of met de bomen zelf, en elk vraagt je om te kiezen: welke achtergrond en hoeveel rijen, data of geen data, hoeveel steekproeven, welke standaardinstellingen. De Explainable Boosting Machine uit [Additieve modellen](/course/open/rai-additive-models?lang=nl) heeft dat allemaal niet nodig, want die legt zichzelf uit. De volgende les, [Wat zou er moeten veranderen?](/course/open/rai-counterfactuals?lang=nl), bouwt met de hand wat DiCE zoekt: de kleinste verandering die het andere antwoord geeft.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_data, choice_ten, result_kernel, result_sampled):
    L.wrap_up(result_sampled, result_kernel, choice_data, choice_ten)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
