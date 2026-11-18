# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-permutation-importance. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-permutation-importance", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een korte vraag over [Soorten uitleg](/course/open/rai-kinds-of-explanation?lang=nl) en een over [Veel bomen stemmen: het random forest](/course/open/ml-random-forest?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kinds():
    quick_kinds = L.choice(
        "quick-kinds",
        question="Een methode husselt één kolom door elkaar en meet hoeveel slechter een model wordt, over alle mensen tegelijk. Ze heeft de antwoorden van het model en de echte labels nodig, niet de binnenkant van het model. Wat voor uitleg is dat?",
        options={
            "right": "Een global explanation, voor elk model",
            "local": "Een local explanation, voor één soort model",
            "inside": "Gelezen uit de eigen weights of splitsingen van het model",
        },
        correct="right",
        why={
            "right": "Goed. De methode heeft alleen nodig wat het model antwoordt en de echte labels, dus ze werkt voor elk model: model-agnostic. En ze beschrijft het model over alle mensen: een global explanation. De eigen importance van het forest, die deze les ook meet, is model-specifiek: die leest de splitsingen van de bomen.",
            "local": "De methode kijkt naar alle mensen tegelijk, dus het is een global explanation, en ze heeft alleen de antwoorden van het model en de labels nodig, dus ze werkt voor elk model. Zie [Soorten uitleg](/course/open/rai-kinds-of-explanation?lang=nl).",
            "inside": "Husselen heeft alleen nodig wat het model antwoordt, niet hoe het vanbinnen werkt: de methode is model-agnostic, en over alle mensen samen een global explanation. Zie [Soorten uitleg](/course/open/rai-kinds-of-explanation?lang=nl).",
        },
    )
    quick_kinds
    return (quick_kinds,)


@app.cell(hide_code=True)
def quick_check_forest():
    quick_forest = L.choice(
        "quick-forest",
        question="Een random forest laat elke boom groeien op een eigen bootstrap-steekproef. Wat maakt de bomen nog meer verschillend?",
        options={
            "right": "Bij elke splitsing een paar willekeurig gekozen features",
            "labels": "Elke boom leert van labels die willekeurig zijn verzonnen",
            "depth": "Elke boom krijgt een andere maximale diepte",
        },
        correct="right",
        why={
            "right": "Goed. Bij elke splitsing mag een boom van het forest alleen kiezen uit een paar willekeurig gekozen features. In scikit-learn is dat `max_features`, standaard de wortel uit het aantal features: 2 van de 4 kolommen in deze les.",
            "labels": "Elke boom leert van de echte labels. Naast een eigen bootstrap-steekproef mag elke boom bij elke splitsing alleen kiezen uit een paar willekeurig gekozen features. Zie [Veel bomen stemmen: het random forest](/course/open/ml-random-forest?lang=nl).",
            "depth": "scikit-learn laat elke boom van een forest zo diep groeien als hij kan, tenzij je een grens zet. Wat verschilt, is de bootstrap-steekproef, en bij elke splitsing een paar willekeurig gekozen features. Zie [Veel bomen stemmen: het random forest](/course/open/ml-random-forest?lang=nl).",
        },
    )
    quick_forest
    return (quick_forest,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een bank laat een random forest beslissen over leningen; net als in [Redenen bij een besluit](/course/open/rai-reason-codes?lang=nl) lezen we een voorspelling dat iemand meer dan 50.000 dollar verdient als het ja van de bank. Een toezichthouder vraagt op welke features het forest leunt. Het forest van scikit-learn heeft een antwoord klaarliggen: `feature_importances_`. In het uitgewerkte voorbeeld van deze les leert een forest van leeftijd, opleiding en uren per week, en van een vierde kolom met willekeurige getallen die niets met het inkomen te maken heeft. Het kant-en-klare antwoord geeft de willekeurige getallen het grootste deel. Zo'n getal kan een bank, een toezichthouder of een rechter op de verkeerde plek laten zoeken. Deze les meet het op een andere manier: hussel één kolom door elkaar, en kijk hoeveel slechter het model het doet bij mensen die het nog niet heeft gezien.

    **Een voorbeeld uit het dagelijks leven.** Hoeveel heeft een band zijn drummer nodig? Je kunt de drumpartij vervangen door die van een ander nummer en horen hoeveel slechter het klinkt: dat is husselen. Of je laat de band repeteren zonder drummer en hoort hoe goed hij zich redt: dat is weglaten en opnieuw trainen. Een band met twee drummers mist elk van de twee nauwelijks, en heeft toch drums nodig.
    """)
    return


@app.cell(hide_code=True)
def predict_noise():
    guess_noise = L.guess(
        "noise",
        question="Een forest met de standaardinstellingen van scikit-learn leert van age, education_num, hours_per_week en een kolom met willekeurige getallen. Zijn eigen importance, `feature_importances_`, geeft elke kolom een deel. Welke kolom krijgt het grootste deel?",
        options={
            "education": "education_num",
            "age": "age",
            "noise": "De willekeurige getallen",
        },
        reveal="**De willekeurige getallen: 48,2 procent**, bijna de helft, twee keer het deel van leeftijd (24,3). Toch kost het husselen ervan op de testhelft het forest niets. Deze les laat zien waar de twee getallen uit elkaar gaan.",
    )
    guess_noise
    return (guess_noise,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Hussel een kolom.** Leo Breiman schreef in 2001, in de publicatie over random forests, dat een forest van bomen niet met een eenvoudige uitleg van zijn werking te doorgronden is. Breiman vond een ingang: na het groeien van elke boom worden de waarden van één variabele in de out-of-bag-voorbeelden in een willekeurige volgorde gezet, en gaan die voorbeelden opnieuw door de boom. De out-of-bag-voorbeelden zijn de rijen die de bootstrap-steekproef van een boom oversloeg, en de fouten van de boom daarop worden opnieuw geteld. Na het husselen staan dezelfde waarden in de kolom, maar elke waarde staat nu in de rij van iemand anders, dus de kolom vertelt het model niets meer over de persoon. Aaron Fisher, Cynthia Rudin en Francesca Dominici (2019) maakten er een methode van voor elk model en noemden het model reliance: de loss met de gehusselde kolom gedeeld door de loss zonder. Een waarde van 1 betekent dat het model niet op de kolom leunt; bij 2 verdubbelt de loss. `permutation_importance` van scikit-learn neemt het verschil: de score vóór min de score na het husselen, gemiddeld over een paar keer husselen.

        **Het eigen getal van het forest.** Elk getraind forest van scikit-learn heeft `feature_importances_`, de mean decrease in impurity, MDI. Voor elke splitsing neemt het hoeveel zuiverder de splitsing de delen maakt, gewogen naar de mensen die hij splitst, en telt dat op per feature; per boom wordt dat een deel, en het forest neemt het gemiddelde over zijn bomen, zodat de delen samen 1 zijn. Het kost niets extra: het komt uit de data om op te trainen, terwijl de bomen groeien. De maat gaat terug op het CART-boek van Breiman, Friedman, Olshen en Stone; Jerome Friedman (2001) schreef dat de onderbouwing alleen op heuristische argumenten rust. scikit-learn waarschuwt voor twee gebreken. Ten eerste komen deze importances uit de data om op te trainen, en zeggen ze dus niet per se welke features het belangrijkst zijn om nieuwe mensen goed te voorspellen. Ten tweede bevoordelen ze features met veel verschillende waarden.

        **Waarom veel waarden winnen.** Carolin Strobl, Anne-Laure Boulesteix, Achim Zeileis en Torsten Hothorn (2007) vergeleken drie maten voor importance van de functie randomForest in R, waaronder permutation importance, en vonden in hun simulaties dat ze bij die functie alle drie onbetrouwbaar zijn, en dat de Gini importance het sterkst vertekend is. Ze geven de oorzaak: variabelen met meer mogelijke splitspunten geven eerder bij toeval een goede waarde voor het criterium, net als wanneer je veel toetsen tegelijk doet. Op de helft om op te trainen biedt de kolom met willekeurige getallen 999 plekken om te splitsen, tegen 72 voor leeftijd, 93 voor de uren en 15 voor education_num. Een boom die groeit tot elk blad zuiver is, vindt steeds weer een splitsing op de willekeurige getallen die bij toeval helpt bij de mensen van wie hij leert.

        **Meet bij mensen die het model niet heeft gezien.** scikit-learn laat hetzelfde zien met willekeurige kolommen in data over de Titanic, en voegt eraan toe dat ook permutation importance misleidt als je die meet op de data waarop een forest heeft getraind en die het uit zijn hoofd heeft geleerd: daar komen de willekeurige kolommen duidelijk hoger in de rangorde dan op de test set. Christoph Molnar raadt aan permutation importance te meten op data die niet voor het trainen zijn gebruikt, om te optimistische uitkomsten te vermijden, vooral bij modellen die overfitten.

        **Weglaten stelt een andere vraag.** Breiman zag het al bij twee gelijke variabelen: wie ze elk apart husselt, ziet de fout bij allebei even veel stijgen, maar als de ene al in het model zit, maakt de andere de fout niet kleiner. Het model opnieuw trainen zonder een feature, wat Jing Lei en anderen leave one covariate out (LOCO) noemen en Molnar leave one feature out (LOFO), vraagt, in de woorden van Molnar: hoeveel slechter voorspelt het model zonder deze feature, gegeven de andere features? Husselen vraagt hoeveel dit ene getrainde model op de kolom leunt. Kolommen met dezelfde informatie delen hun importance; volgens scikit-learn geeft het husselen van een van de twee een lagere importance voor allebei, terwijl ze misschien wel belangrijk zijn. Giles Hooker, Lucas Mentch en Siyu Zhou (2021) waarschuwen voor de andere kant: husselen maakt rijen die niet kunnen bestaan, zoals zwangere mannen, en bij flexibele modellen kunnen zulke methoden features die samenhangen veel te zwaar laten wegen.

        **Wat het je niet vertelt.** Permutation importance zegt volgens scikit-learn niet hoeveel een feature op zichzelf voorspelt, maar hoe belangrijk die feature is voor één bepaald model. Molnar: het zegt niet hoe de feature de voorspelling beïnvloedt, alleen hoeveel hij de loss raakt. En husselen is willekeurig; Molnar weer: als je het herhaalt, kunnen de uitkomsten flink verschillen. Hussel dus meer dan één keer, en kijk naar de spreiding.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **feature importance** | een getal per feature voor hoeveel een model erop leunt; er zijn verschillende manieren om het te meten |
        | **permutation importance** | hoeveel slechter een model het doet op data die het niet heeft gezien als één kolom door elkaar wordt gehusseld; Fisher en anderen noemen het model reliance |
        | **impurity importance (MDI)** | hoeveel de splitsingen van een boom op een feature de Gini impurity verlagen op de data om op te trainen, gewogen naar de mensen die ze splitsen, als deel; een forest neemt het gemiddelde over zijn bomen |
        | **leave one feature out (LOFO)** | het model opnieuw trainen zonder de feature, en meten hoeveel slechter het wordt |
        | `n_repeats` | hoe vaak `permutation_importance` van scikit-learn elke kolom husselt; het geeft het gemiddelde en de spreiding |
        """),
        L.watch_out("Permutation importance beschrijft één getraind model, niet de wereld. scikit-learn waarschuwt dat een feature die voor een slecht model weinig uitmaakt, voor een goed model heel belangrijk kan zijn. En even goede modellen kunnen op verschillende features leunen, zoals in [Veel goede modellen](/course/open/rai-rashomon?lang=nl): Fisher, Rudin en Dominici meten daarom ook hoe ver de importance van een feature uiteenloopt over alle modellen van één soort die goed voorspellen."),
    ])
    return


@app.cell(hide_code=True)
def explore_leaf_cell():
    explore_leaf = L.explore("leaf", "mensen per blad, minstens", lambda v: C.leaf_picture(v, "nl"), start=1,
                             values=C.LEAVES, role="hyper",
                             task="Schuif van 1 naar 50. Wat gebeurt er met de willekeurige getallen in de linker grafiek, en in de rechter? En met hoe vaak het forest goed zit op de testhelft?")
    mo.vstack([
        mo.md("**Twee importances, als de bladeren groter worden**"),
        explore_leaf,
        mo.md(r"""
        Een forest van 100 bomen op de helft van UCI Adult om op te trainen, met leeftijd, opleiding, uren en een kolom met willekeurige getallen. De schuif zet `min_samples_leaf`, het kleinste aantal mensen dat een blad mag bevatten; 1 is de standaard van scikit-learn, bomen die groeien tot elk blad zuiver is. Links de eigen importance van het forest, rechts permutation importance op de testhelft.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om te meten op welke features een model leunt:

    1. Verdeel de data in een deel om op te trainen en een deel om op te testen, en train het model op het eerste deel.
    2. Meet hoe vaak het model goed zit op het deel om op te testen.
    3. Hussel één kolom van het testdeel door elkaar, zodat elke persoon de waarde van iemand anders krijgt, en meet opnieuw.
    4. De importance van die kolom is hoeveel slechter het model het deed. Hussel een paar keer en neem het gemiddelde.
    5. Doe hetzelfde voor elke kolom. Waar twee kolommen dezelfde informatie dragen, hussel ze ook samen, of laat er een weg en train opnieuw: elk beantwoordt zijn eigen vraag.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De drie kolommen uit [Redenen bij een besluit](/course/open/rai-reason-codes?lang=nl), en een vierde: `integers(0, 1000, size=n)` trekt n gehele getallen van 0 tot en met 999. `.to_numpy()` maakt van kolommen van een tabel een NumPy-array, en `np.column_stack` zet kolommen naast elkaar in één array. De helften zijn die van de lessen hiervoor: `//` deelt en houdt het gehele deel. `RandomForestClassifier(random_state=0)` heeft de standaardinstellingen van scikit-learn: 100 bomen, die elk groeien tot hun bladeren zuiver zijn. `.score` geeft hoe vaak een model goed zit, en `feature_importances_` is de eigen importance van het forest. `permutation_importance(model, X, y, n_repeats=5, random_state=0)` husselt elke kolom vijf keer en geeft `importances_mean`, de gemiddelde daling van de accuracy, en `importances_std`, de spreiding daarvan. Dan education_num twee keer: `_X[:, 1]` is die kolom, `.copy()` kopieert een array zodat het veranderen van de kopie het origineel laat zoals het is, `_both[:, [1, 4]] = ...` vervangt twee kolommen tegelijk, en `[:, [0, 2, 3, 4]]` houdt alleen die kolommen. De index `np.random.default_rng(0).permutation(len(_test))` zet de mensen van de testhelft in één gehusselde volgorde voor beide kopieën, zodat elke persoon ze allebei van dezelfde andere persoon krijgt. Dit forest heeft `min_samples_leaf=20`: elk blad bevat minstens 20 mensen, zodat het niet elke persoon van de helft om op te trainen uit zijn hoofd kan leren. Voer de cel uit. Dat duurt ongeveer een halve minuut.
    """)
    return


@app.cell
def example_forest():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance

    # 1. drie kolommen van UCI Adult en een kolom met willekeurige getallen, in twee helften; een forest zoals standaard
    _people = pd.read_csv("data/adult.csv.gz")
    _noise = np.random.default_rng(2).integers(0, 1000, size=len(_people))
    _X = np.column_stack([_people[["age", "education_num", "hours_per_week"]].to_numpy(), _noise])
    _y = (_people["income"] == ">50K").to_numpy()
    _order = np.random.default_rng(0).permutation(len(_y))
    _train, _test = _order[: len(_y) // 2], _order[len(_y) // 2:]
    _forest = RandomForestClassifier(random_state=0).fit(_X[_train], _y[_train])

    # 2. hoe vaak het goed zit
    print("goed, helft om op te trainen:", round(_forest.score(_X[_train], _y[_train]), 4))
    print("goed, testhelft:", round(_forest.score(_X[_test], _y[_test]), 4))

    # 3 en 4. de eigen importance; dan elke kolom vijf keer gehusseld, op de testhelft en op de helft om op te trainen
    print("eigen importance:", _forest.feature_importances_.round(3))
    _shuffled = permutation_importance(_forest, _X[_test], _y[_test], n_repeats=5, random_state=0)
    print("gehusseld, testhelft:", _shuffled.importances_mean.round(4))
    print("spreiding:", _shuffled.importances_std.round(4))
    _learned = permutation_importance(_forest, _X[_train], _y[_train], n_repeats=5, random_state=0)
    print("gehusseld, helft om op te trainen:", _learned.importances_mean.round(4))

    # 5. education_num twee keer: elke kopie husselen, beide samen husselen, een of beide weglaten en opnieuw trainen
    _X5 = np.column_stack([_X, _X[:, 1]])
    _leafy = RandomForestClassifier(min_samples_leaf=20, random_state=0).fit(_X5[_train], _y[_train])
    _right = _leafy.score(_X5[_test], _y[_test])
    print("met een kopie, goed:", round(_right, 4))
    _each = permutation_importance(_leafy, _X5[_test], _y[_test], n_repeats=5, random_state=0)
    print("elk gehusseld:", _each.importances_mean.round(4))
    _both = _X5[_test].copy()
    _both[:, [1, 4]] = _both[np.random.default_rng(0).permutation(len(_test))][:, [1, 4]]
    print("beide samen gehusseld:", round(_right - _leafy.score(_both, _y[_test]), 4))
    _one = RandomForestClassifier(min_samples_leaf=20, random_state=0).fit(_X5[_train][:, [0, 2, 3, 4]], _y[_train])
    print("een weggelaten, opnieuw getraind:", round(_right - _one.score(_X5[_test][:, [0, 2, 3, 4]], _y[_test]), 4))
    _none = RandomForestClassifier(min_samples_leaf=20, random_state=0).fit(_X5[_train][:, [0, 2, 3]], _y[_train])
    print("beide weggelaten, opnieuw getraind:", round(_right - _none.score(_X5[_test][:, [0, 2, 3]], _y[_test]), 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met de standaardinstellingen zit het forest goed bij 99,6 procent van de helft om op te trainen, die het uit zijn hoofd heeft geleerd, en bij 76,0 procent van de testhelft: nauwelijks beter dan altijd nee zeggen, 75,9 procent. Zijn eigen importance geeft de willekeurige getallen 0,482, bijna de helft. Op de testhelft kost het husselen daarvan niets: −0,0002, dus −0,02 procentpunt, ruim binnen de spreiding van 0,0017; education_num husselen kost 4,4 procentpunt, leeftijd 2,9 en de uren 1,3. Op de helft om op te trainen lijkt elke kolom belangrijk, ook de willekeurige getallen: 20,7 procentpunt, omdat het forest ze daar uit zijn hoofd heeft geleerd. Dan education_num twee keer. Apart gehusseld kost elke kopie maar 1,9 en 1,4 procentpunt, omdat het forest de andere nog heeft; samen gehusseld 6,1 procentpunt, ongeveer wat education_num een forest zonder kopie kost bij dezelfde grootte van de bladeren, 6,2 op de schuif. Weggelaten en opnieuw getraind kost één kopie 0,1 procentpunt en kosten ze samen 3,3: minder dan beide husselen, omdat een forest dat zonder education_num wordt getraind het meeste haalt uit de kolommen die het nog heeft.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_importance():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een kolom met de hand husselen

        Het vak bevat `model`, de boom met diepte 2 uit [Veel goede modellen](/course/open/rai-rashomon?lang=nl): ja bij een opleiding boven 12,5 en een leeftijd boven 33,5. Schrijf `importance(rows, labels, j, order)`: hoeveel minder vaak `model` goed zit als kolom `j` door elkaar is gehusseld. `rows` bevat één lijst per persoon, `labels` of elke persoon meer dan 50.000 dollar verdient, en `order` de mensen in een gehusselde volgorde: persoon `i` krijgt kolom `j` van persoon `order[i]`. `list(rows[i])` maakt een kopie van een rij, zodat je de kopie kunt veranderen en `rows` blijft zoals het is. De check roept `importance` eerst aan met zes mensen, `j = 1` en `order = [1, 0, 3, 4, 5, 2]`, en daarna met andere mensen, kolommen en volgordes. De tekening voert je functie uit op de 24.421 mensen van de testhelft.
        """),
        L.task("Schrijf `importance(rows, labels, j, order)`: goed vóór, min goed met kolom `j` van persoon `order[i]`, gedeeld door het aantal mensen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_importance():
    box_importance = L.code_box("importance", lines=19)
    box_importance
    return (box_importance,)


@app.cell(hide_code=True)
def check_importance(box_importance):
    result_importance = L.check("importance", box_importance.value)
    result_importance
    return (result_importance,)


@app.cell(hide_code=True)
def hints_importance():
    L.hints(
        "importance",
        [
            "Tel twee dingen in één loop over de mensen: hoe vaak `model` goed zit bij de rij zoals hij is, en hoe vaak bij een kopie van de rij waarin kolom `j` van persoon `order[i]` komt. Het verschil, gedeeld door het aantal mensen, is de importance.",
            "Loop met `for i in range(len(rows)):`. Tel 1 op bij `right` als `model(rows[i]) == labels[i]`. Maak `row = list(rows[i])`, zet daarin de waarde die persoon `order[i]` in kolom `j` heeft, en tel 1 op bij `shuffled` als `model(row) == labels[i]`.",
            "`row[j] = rows[___][j]`",
        ],
        why="`right = 0` en `shuffled = 0`, dan `for i in range(len(rows)):` met `if model(rows[i]) == labels[i]:` dat 1 optelt bij `right`, `row = list(rows[i])`, `row[j] = rows[order[i]][j]`, en `if model(row) == labels[i]:` dat 1 optelt bij `shuffled`. Na de loop `return (right - shuffled) / len(rows)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_impurity():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Het eigen getal van het forest, voor één boom

        Waar komt `feature_importances_` vandaan? Het vak bevat een boom met diepte 3, gegroeid op de helft om op te trainen, zoals scikit-learn hem bewaart in `tree_`. Voor elke knoop is `feature` de kolom waarop hij splitst, zijn `left` en `right` de knopen eronder, is `impurity` zijn Gini impurity, en `people` hoeveel mensen van de helft om op te trainen bij hem komen. Een blad heeft `-1` in `left` en `right`, en `-2` in `feature`. Wat een splitsing doet, is de impurity van de knoop maal zijn mensen, min hetzelfde voor de twee knopen eronder: hoeveel zuiverder de splitsing de delen maakt, gewogen naar de mensen. Vul de loop aan, zodat `gain` die daling per kolom opgeteld bevat. De laatste regels maken van `gain` delen; de tekening laat de jouwe zien.
        """),
        L.task("Vervang de `...`: tel voor elke knoop met `left[n] != -1` zijn daling op bij `gain[feature[n]]`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_impurity():
    box_impurity = L.code_box("impurity", lines=21)
    box_impurity
    return (box_impurity,)


@app.cell(hide_code=True)
def check_impurity(box_impurity):
    result_impurity = L.check("impurity", box_impurity.value)
    result_impurity
    return (result_impurity,)


@app.cell(hide_code=True)
def hints_impurity():
    L.hints(
        "impurity",
        [
            "Alleen de knopen die splitsen tellen mee: hun `left[n]` is niet `-1`. Reken voor zo'n knoop uit hoeveel de splitsing de impurity verlaagt, gewogen naar de mensen, en tel dat op bij de kolom waarop de knoop splitst.",
            "Noem binnen `if left[n] != -1:` de twee knopen eronder `a = left[n]` en `b = right[n]`. Reken `drop` uit met `people` en `impurity` van `n`, `a` en `b`, en tel het op bij `gain[feature[n]]`.",
            "`drop = people[n] * impurity[n] - people[a] * impurity[a] - people[b] * ___`",
        ],
        why="`if left[n] != -1:` met `a = left[n]`, `b = right[n]`, `drop = people[n] * impurity[n] - people[a] * impurity[a] - people[b] * impurity[b]` en `gain[feature[n]] = gain[feature[n]] + drop`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_train():
    mo.md(r"""
    ### Oefening 3. Welke helft?
    """)
    return


@app.cell(hide_code=True)
def choice_train_cell():
    choice_train = L.choice(
        "train",
        question="In het uitgewerkte voorbeeld kost het husselen van de willekeurige getallen het forest 20,7 procentpunt op de helft om op te trainen, en niets op de testhelft. Welke van de twee zegt of het forest ze gebruikt voor nieuwe mensen?",
        options={
            "right": "De testhelft: mensen die het niet heeft gezien",
            "train": "De helft om op te trainen: daar leerde het",
            "both": "Allebei even goed: husselen gaat overal hetzelfde",
        },
        correct="right",
        why={
            "right": "Goed. Op de helft om op te trainen heeft het forest elke persoon uit zijn hoofd geleerd, ook de willekeurige getallen, dus husselen doet daar pijn. Voor nieuwe mensen vertellen ze het niets, en husselen kost niets. Molnar raadt aan het te meten op data die niet voor het trainen zijn gebruikt.",
            "train": "Daar heeft het forest de willekeurige getallen uit zijn hoofd geleerd: het zit bij 99,6 procent van die helft goed. Voor nieuwe mensen vertellen ze het niets, zoals de testhelft laat zien. Zie *Meet bij mensen die het model niet heeft gezien* in *Het idee*.",
            "both": "Het husselen is hetzelfde, maar de mensen niet: het forest heeft de helft om op te trainen uit zijn hoofd geleerd, ook de willekeurige getallen, en de testhelft heeft het nooit gezien. Zie *Meet bij mensen die het model niet heeft gezien* in *Het idee*.",
        },
    )
    choice_train
    return (choice_train,)


@app.cell(hide_code=True)
def text_exercise_copy():
    mo.md(r"""
    ### Oefening 4. Weglaten
    """)
    return


@app.cell(hide_code=True)
def choice_copy_cell():
    choice_copy = L.choice(
        "copy",
        question="Met education_num twee keer in de tabel kost het weglaten van één kopie en opnieuw trainen het forest 0,1 procentpunt. Wat zegt dat?",
        options={
            "right": "De andere kopie heeft het ook en neemt het over",
            "useless": "Opleiding helpt niet om het inkomen te voorspellen",
            "unused": "Het forest gebruikte die kopie nooit",
        },
        correct="right",
        why={
            "right": "Goed. Opnieuw trainen vraagt wat een feature toevoegt naast de andere, en de andere kopie bevat dezelfde informatie. Beide weglaten kost 3,3 procentpunt, en beide samen husselen 6,1: opleiding doet ertoe, en de twee kopieën delen het.",
            "useless": "Laat beide kopieën weg en het forest wordt 3,3 procentpunt slechter; hussel ze samen en het wordt 6,1 procentpunt slechter. Opleiding helpt; de andere kopie neemt het alleen over. Zie *Weglaten stelt een andere vraag* in *Het idee*.",
            "unused": "Het forest gebruikte allebei: education_num husselen kost het 1,9 procentpunt, de kopie 1,4. Opnieuw getraind zonder een van de twee haalt het forest alles uit de andere kopie. Zie *Weglaten stelt een andere vraag* in *Het idee*.",
        },
    )
    choice_copy
    return (choice_copy,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Soorten uitleg](/course/open/rai-kinds-of-explanation?lang=nl) heb je uitleg ingedeeld: permutation importance is een global explanation en model-agnostic, terwijl de eigen importance van het forest alleen voor bomen werkt. Het forest is dat uit [Veel bomen stemmen: het random forest](/course/open/ml-random-forest?lang=nl), en de bootstrap-steekproeven ervan gaven Breiman de eerste versie van deze maat: elke boom werd getest op de rijen die zijn steekproef oversloeg. De volgende les, [Hoe een voorspelling meebeweegt met een feature](/course/open/rai-pdp-ice?lang=nl), vraagt wat deze les openlaat: niet hoeveel een feature ertoe doet, maar welke kant de voorspelling op gaat als de feature verandert.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_copy, choice_train, result_importance, result_impurity):
    L.wrap_up(result_importance, result_impurity, choice_train, choice_copy)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
