# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-pdp-ice. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-pdp-ice", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Welke features doen ertoe?](/course/open/rai-permutation-importance?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_which():
    quick_which = L.choice(
        "quick-which",
        question="Als je education_num op de testhelft door elkaar husselt, kost dat het forest 4,4 procentpunt accuracy. Wat vertelt dat getal je?",
        options={
            "up": "Dat meer opleiding meer ja's geeft",
            "right": "Hoeveel het forest erop leunt, niet welke kant op",
            "world": "Hoeveel opleiding in de wereld aan inkomen toevoegt",
        },
        correct="right",
        why={
            "right": "Goed. Permutation importance zegt hoeveel slechter het forest wordt zonder de informatie in de kolom, niet of een hogere waarde de voorspelling omhoog of omlaag duwt. Molnar: feature importance zegt niet hoe de feature de voorspelling beïnvloedt, alleen hoeveel hij de loss verandert. Deze les laat zien welke kant op.",
            "up": "4,4 procentpunt zegt hoeveel het forest op opleiding leunt, niet welke kant het de voorspelling op duwt: dat kan zelfs van persoon tot persoon verschillen. Deze les tekent welke kant op. Zie [Welke features doen ertoe?](/course/open/rai-permutation-importance?lang=nl).",
            "world": "Permutation importance beschrijft één getraind model, niet de wereld: volgens scikit-learn laat het zien hoe belangrijk een feature is voor één bepaald model. Zie [Welke features doen ertoe?](/course/open/rai-permutation-importance?lang=nl).",
        },
    )
    quick_which
    return (quick_which,)


@app.cell(hide_code=True)
def quick_check_copies():
    quick_copies = L.choice(
        "quick-copies",
        question="Twee kolommen bevatten dezelfde informatie, zoals education_num twee keer. Je husselt elk ervan apart. Wat laat permutation importance zien?",
        options={
            "copy": "De kopie krijgt alle importance",
            "right": "Elke kopie lijkt minder belangrijk dan ze samen zijn",
            "double": "Elke kopie krijgt twee keer de importance die bij hem hoort",
        },
        correct="right",
        why={
            "right": "Goed. Hussel je één kopie, dan heeft het forest de andere nog: in het uitgewerkte voorbeeld van die les kostte de ene kopie 1,9 procentpunt, de andere 1,4, en beide samen 6,1. Features die samenhangen geven ook in deze les problemen.",
            "copy": "Beide kopieën krijgen een klein deel: apart gehusseld kostte de ene kopie 1,9 procentpunt, de andere 1,4, en beide samen 6,1, omdat het forest terugvalt op de kopie die het nog heeft. Zie [Welke features doen ertoe?](/course/open/rai-permutation-importance?lang=nl).",
            "double": "Eén kopie husselen doet minder pijn, niet meer: het forest heeft de andere nog. De ene kopie kostte 1,9 procentpunt, de andere 1,4, beide samen 6,1. Zie [Welke features doen ertoe?](/course/open/rai-permutation-importance?lang=nl).",
        },
    )
    quick_copies
    return (quick_copies,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Welke features doen ertoe?](/course/open/rai-permutation-importance?lang=nl) leerde de toezichthouder van de bank dat het forest op opleiding en leeftijd leunt; net als eerder is een voorspelling van meer dan 50.000 dollar verdienen het ja van de bank. De volgende vraag is welke kant op. Stijgt de kans op een ja met de leeftijd, en blijft hij stijgen? Voor iedereen op dezelfde manier? Permutation importance kan dat niet zeggen: het meet hoeveel slechter het model wordt, niet waar de voorspelling heen gaat. Deze les vraagt het rechtstreeks aan het model: geef iedereen dezelfde leeftijd, kijk wat het zegt, en doe dat voor elke leeftijd. Het gemiddelde over alle mensen is een partial dependence plot; één lijn per persoon laat zien wie anders reageert.

    **Een voorbeeld uit het dagelijks leven.** Een bakker vraagt zich af wat 50 cent meer voor een brood zou doen. Vraag het alle klanten en neem het gemiddelde, en je hoort: een paar broden minder per week. Vraag het ze een voor een, en de meesten maakt het niets uit, terwijl een paar helemaal wegblijven. Het gemiddelde antwoord is de partial dependence; de antwoorden een voor een zijn wat deze les ICE curves noemt.
    """)
    return


@app.cell(hide_code=True)
def predict_steps():
    guess_steps = L.guess(
        "steps",
        question="Een boom met diepte 2 vraagt eerst naar opleiding en dan naar leeftijd: bij 33,5 voor een opleiding tot 12,5, bij 28,5 daarboven. Geef iedereen leeftijd 17, dan 18, tot 90, en neem het gemiddelde van wat de boom zegt: dat gemiddelde heeft twee sprongen, bij 29 en bij 34. Hoeveel sprongen heeft de lijn van één persoon?",
        options={
            "two": "Twee, net als het gemiddelde",
            "one": "Eén",
            "none": "Geen: de lijn van één persoon is vlak",
        },
        reveal="**Eén.** Iemand met een opleiding tot 12,5 komt alleen de vraag over 33,5 tegen, en maakt een sprong bij 34; iemand boven 12,5 maakt een sprong bij 29. Het gemiddelde heeft twee sprongen omdat het beide soorten mensen mengt: niemand heeft een lijn zoals het gemiddelde. Oefening 1 tekent ze.",
    )
    guess_steps
    return (guess_steps,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Zet een feature vast, neem het gemiddelde van de voorspellingen.** Jerome Friedman introduceerde partial dependence plots in 2001, om naar een model met veel inputs te kijken via één of twee ervan tegelijk. Kies een feature $S$ en een waarde $v$. Geef elke rij van de data de waarde $v$ voor $S$, laat de andere kolommen $C$ zoals ze zijn, vraag het model $f$, en neem het gemiddelde over de $N$ rijen:

        $$\mathrm{PD}_S(v) = \frac{1}{N}\sum_{i=1}^{N} f(v,\ x_{C,i})$$

        waarbij $x_{C,i}$ de andere kolommen van rij $i$ zijn. Doe dat voor elke waarde op een grid en teken de lijn: de partial dependence plot, PDP. Volgens Friedman kun je partial dependence functions gebruiken om de modellen van elke voorspelmethode te helpen begrijpen, hoe ondoorzichtig die ook is, en dan het best met één of twee features tegelijk. Friedman gaf ook een voorwaarde: het gemiddelde kan een bruikbare samenvatting zijn zolang de vorm van de curve niet te sterk afhangt van de waarden van de andere features.

        **Eén lijn per persoon.** Alex Goldstein, Adam Kapelner, Justin Bleich en Emil Pitkin (2015) gaan uit van die voorwaarde. Features werken op elkaar in als wat de ene feature doet afhangt van de waarde van een andere. Als features flink op elkaar inwerken, schrijven ze, kan het verband per rij verschillen, en dan kan een gemiddelde curve zoals de PDP verbergen hoe ingewikkeld het verband in het model is. Hun antwoord is de lijn van elke rij te houden: een individual conditional expectation curve, ICE curve. Ze merken op dat de PDP het gemiddelde is van de N ICE curves, en dus een bewerking achteraf. Friedman had al iets vergelijkbaars voorgesteld: reken de curve uit voor een paar willekeurig gekozen rijen, om te zien hoeveel hij varieert.

        **Als het gemiddelde vlak is.** Goldstein en collega's laten het gevaar zien met verzonnen data: $Y = 0{,}2X_1 - 5X_2 + 10X_2 \cdot 1(X_3 \ge 0) + E$, waarbij $1(X_3 \ge 0)$ 1 is als $X_3 \ge 0$ en anders 0, $E$ ruis is, en $X_1$, $X_2$, $X_3$ gelijkmatig getrokken zijn tussen −1 en 1. Ze trainen een model op 1000 rijen, en de partial dependence op $X_2$ komt er vlak uit. Toch hangt $X_2$ duidelijk samen met $Y$, schrijven ze: het middelen in de PDP houdt dat alleen uit het zicht. Voor rijen met $X_3 \ge 0$ heeft de lijn van $X_2$ een helling van $-5 + 10 = +5$; voor de andere rijen $-5$. Ongeveer de helft van de rijen valt aan elke kant, en in het gemiddelde heffen de hellingen elkaar op. Je rekent het uit in oefening 2.

        **Knijp de lijnen samen.** Als de lijnen op verschillende hoogtes beginnen, zie je slecht hoeveel elke lijn beweegt. Goldstein en collega's knijpen dan alle lijnen samen in één waarde, door van elke lijn zijn waarde daar af te trekken: een centered ICE plot, c-ICE. De laagste of de hoogste waarde kiezen gaf bij hen de best leesbare plots; Molnar vindt het laagste uiteinde van de feature een goede keuze. `PartialDependenceDisplay.from_estimator` van scikit-learn doet het met `centered=True`, bij de eerste waarde van de grid. De y-as van zo'n plot is dan een verandering, geen kans meer.

        **Rijen die niet bestaan.** Om de curve bij 17 jaar uit te rekenen, krijgt iedereen leeftijd 17, ook mensen die gepromoveerd zijn. Features hangen samen als hun waarden met elkaar meegaan, zoals leeftijd en opleiding; scikit-learn waarschuwt dat PDP's en ICE curves ervan uitgaan dat de feature die je bekijkt onafhankelijk is van de andere features, wat in de praktijk vaak niet zo is; bij features die samenhangen maak je dan onzinnige datapunten. Het voorbeeld van Molnar: bij een lengte van 200 cm kan het middelen over het gewicht een gewicht onder de 50 kg meenemen, wat voor iemand van 2 meter niet realistisch is. Daniel Apley en Jingyu Zhu (2020) schrijven dat partial dependence plots foute uitkomsten kunnen geven als features sterk samenhangen, omdat het model dan moet voorspellen voor combinaties van waarden ver buiten het gebied van de training data. Waar geen echte rijen zijn, kan het model alleen gokken. Laat dus de data onder de curve zien. Volgens Molnar kun je zonder de verdeling te veel lezen in stukken waar bijna geen data zijn. Goldstein en collega's zetten op elke lijn een stip bij de eigen waarde van de rij.

        **Wat het je niet vertelt.** Molnar: het verband is causaal voor het model, omdat het model de uitkomst uitrekent uit de features, maar niet per se voor de echte wereld. Een curve die stijgt met opleiding zegt dat het forest vaker ja zegt als je de opleiding in een rij verhoogt; hij zegt niet dat meer school iemands inkomen verhoogt. En een curve laat één of twee features tegelijk zien: meer dan twee kun je volgens Molnar in de praktijk niet zinvol tekenen.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **partial dependence plot (PDP)** | voor elke waarde van één feature: geef elke rij die waarde, vraag het model en neem het gemiddelde; getekend als een lijn over de waarden |
        | **ICE curve** | individual conditional expectation: hetzelfde voor één rij, zonder gemiddelde; de PDP is het gemiddelde van de ICE curves |
        | **centered ICE (c-ICE)** | ICE curves min hun waarde in één punt, meestal het laagste, zodat ze allemaal bij 0 beginnen |
        | **grid** | de waarden van de feature waarbij de curves worden uitgerekend |
        | **op elkaar inwerken** | wat de ene feature doet, hangt af van de waarde van een andere; de ICE curves lopen dan niet naast elkaar |
        | $1(X_3 \ge 0)$ | 1 als $X_3 \ge 0$, en anders 0 |
        | `kind` | in `partial_dependence` van scikit-learn: `"average"` voor de PDP, `"individual"` voor de ICE curves, `"both"` voor allebei |
        """),
        L.watch_out("Een partial dependence plot laat zien wat het model doet als je één kolom van een rij verandert, niet wat er in de wereld met iemand gebeurt. En waar de curve door rijen loopt die niet voorkomen, zoals gepromoveerde zeventienjarigen, gokt het model: kijk waar de echte mensen zitten voor je de curve leest."),
    ])
    return


@app.cell(hide_code=True)
def explore_education_cell():
    explore_education = L.explore("education", "education_num", lambda v: C.level_picture(v, "nl"), start=9,
                                  values=C.EDUCATION, role="data",
                                  task="Schuif van 1 naar 16. Hoeveel stijgen de dunne lijnen van 25 tot 45 jaar bij mensen met weinig opleiding, en bij mensen met een diploma van hbo of universiteit? En het gestippelde gemiddelde van iedereen?")
    mo.vstack([
        mo.md("**ICE curves voor leeftijd, per opleiding**"),
        explore_education,
        mo.md(r"""
        Het forest van het uitgewerkte voorbeeld: 100 bomen met bladeren van minstens 20 mensen, gegroeid op de helft om op te trainen. Dunne lijnen: wat het zegt voor 12 mensen van de testhelft met de gekozen education_num, als hun leeftijd op elke leeftijd van 17 tot 90 wordt gezet. Dikke lijn: hun gemiddelde. Gestippeld: het gemiddelde over de 1000 mensen van het uitgewerkte voorbeeld, de partial dependence plot. education_num 9 is een Amerikaans middelbareschooldiploma (HS-grad), 13 een bachelor.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om te zien hoe de voorspelling van een model meebeweegt met één feature:

    1. Kies de feature en de grid: elke waarde die de feature aanneemt, of waarden op gelijke afstand tussen twee percentielen.
    2. Kopieer voor elke rij en elke waarde op de grid de rij, zet de feature op die waarde en vraag het model. Elke rij geeft een ICE curve.
    3. Neem bij elke waarde op de grid het gemiddelde van de curves: dat is de partial dependence.
    4. Teken de curves met hun gemiddelde. Waar ze uitwaaieren, zoek je de feature die ze sorteert; knijp ze samen bij de laagste waarde om te vergelijken hoeveel elke curve beweegt.
    5. Geef aan waar de echte rijen zitten, en lees de curve niet waar de waarde van de feature niet samen met de rest van de rij voorkomt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het forest uit [Welke features doen ertoe?](/course/open/rai-permutation-importance?lang=nl) met bladeren van minstens 20 mensen, op de drie kolommen uit [Redenen bij een besluit](/course/open/rai-reason-codes?lang=nl), met dezelfde helften. `.astype(float)` maakt van gehele getallen kommagetallen, en daar vraagt `partial_dependence` om. `partial_dependence(model, X, [0], kind="both")` neemt kolom 0, leeftijd, als feature, en geeft drie dingen: `grid_values`, de leeftijden op de grid; `average`, de partial dependence bij elk ervan; en `individual`, de ICE curve van elke rij van `X`. Elk komt als lijst met één plaats per feature, vandaar de `[0]`. scikit-learn vraagt het model naar elke rij bij elke leeftijd, dus het voorbeeld neemt de eerste 1000 mensen van de testhelft; `PartialDependenceDisplay.from_estimator(model, X, [0], kind="both")`, dat de plot tekent met matplotlib, tekent hoogstens 1000 ICE curves (`subsample=1000`), tenzij je om meer vraagt. Standaard heeft de grid 100 waarden op gelijke afstand tussen het 5e en het 95e percentiel van de feature, of elke waarde van de feature als dat er minder dan 100 zijn. `np.isin(a, values)` zegt voor elk element van `a` of het een van `values` is, `np.allclose(a, b)` of twee arrays gelijk zijn op afronding na, `_lines[:, _ages == 45]` neemt van elke lijn de kolom bij 45 jaar, `[:, 0]` maakt van die ene kolom een platte array, kolom 0 van `_lines` is de eerste leeftijd op de grid, 17, en `&` tussen twee voorwaarden klopt waar ze allebei kloppen. Voer de cel uit. Dat duurt een paar seconden.
    """)
    return


@app.cell
def example_pdp():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import partial_dependence

    # 1. drie kolommen van UCI Adult als kommagetallen, in twee helften; een forest, bladeren van minstens 20 mensen
    _people = pd.read_csv("data/adult.csv.gz")
    _X = _people[["age", "education_num", "hours_per_week"]].to_numpy().astype(float)
    _y = (_people["income"] == ">50K").to_numpy()
    _order = np.random.default_rng(0).permutation(len(_y))
    _train, _test = _order[: len(_y) // 2], _order[len(_y) // 2:]
    _forest = RandomForestClassifier(min_samples_leaf=20, random_state=0).fit(_X[_train], _y[_train])
    _some = _X[_test][:1000]

    # 2 en 3. elke leeftijd op de grid voor alle 1000 mensen: één lijn per persoon, en hun gemiddelde
    _pd = partial_dependence(_forest, _some, [0], kind="both")
    _ages, _average, _lines = _pd["grid_values"][0], _pd["average"][0], _pd["individual"][0]
    print("grid:", _ages[0], "tot", _ages[-1], "met", len(_ages), "leeftijden")
    print("gemiddelde bij 20, 30, ..., 90:", _average[np.isin(_ages, [20, 30, 40, 50, 60, 70, 80, 90])].round(3))
    print("het gemiddelde van de lijnen:", np.allclose(_average, _lines.mean(axis=0)))

    # 4. hoeveel elke lijn stijgt van 25 tot 45 jaar, per opleiding
    _rise = _lines[:, _ages == 45][:, 0] - _lines[:, _ages == 25][:, 0]
    for _low, _high in [(1, 9), (10, 12), (13, 16)]:
        _them = (_some[:, 1] >= _low) & (_some[:, 1] <= _high)
        print("opleiding", _low, "tot", _high, "-", _them.sum(), "mensen, stijging:", _rise[_them].mean().round(3))

    # 5. wie er in het gemiddelde zit bij 17 jaar
    _degree = _some[:, 1] >= 13
    print("met opleiding 13 of meer:", _degree.sum(), "van de", len(_some))
    print("bij 17 jaar, met:", _lines[_degree, 0].mean().round(3))
    print("bij 17 jaar, zonder:", _lines[_some[:, 1] < 13, 0].mean().round(3))
    _young = _X[:, 0] == 17
    print("17 jaar in de tabel:", _young.sum(), "- met 13 of meer:", (_young & (_X[:, 1] >= 13)).sum())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Bij deze 1000 mensen neemt leeftijd 62 waarden aan, van 17 tot 90, dus de grid bevat elk ervan. Het gemiddelde stijgt van 0,038 bij 20 jaar tot 0,354 bij 50, en zakt na 70 tot ongeveer 0,28; het is precies het gemiddelde van de 1000 lijnen. Maar de lijnen stijgen niet allemaal even hard. Van 25 tot 45 jaar stijgt een lijn gemiddeld met 0,154 bij de 452 mensen met een opleiding tot 9, met 0,204 bij de 296 met 10 tot 12, en met 0,476 bij de 252 met 13 of meer. Het forest heeft geleerd dat leeftijd meer uitmaakt voor mensen met meer opleiding, en het gemiddelde, dat met 0,25 stijgt, verbergt dat. Dan het linkeruiteinde. Bij 17 jaar telt het gemiddelde die 252 mensen met een opleiding van 13 of meer mee, terwijl geen van de 595 zeventienjarigen in de hele tabel zoveel opleiding heeft. Voor hen zegt het forest bij 17 jaar 0,102, voor de anderen 0,016: een gok over mensen die niet voorkomen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ice():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De lijn van één persoon

        Het vak bevat `model`, een boom met diepte 2, gegroeid op de helft om op te trainen, met in elk van zijn vier bladeren het deel ja, afgerond. Schrijf `ice(row, j, grid)`: de ICE curve van één persoon, een lijst met wat `model` zegt voor `row` met kolom `j` om de beurt op elke waarde van `grid`. `list(row)` maakt een kopie, zodat je de kopie kunt veranderen en `row` laat zoals hij is. De check roept eerst `ice([30, 10, 40], 0, [20, 30, 40, 50])` aan, en dan andere rijen, kolommen en grids. De tekening voert je functie uit voor 40 mensen van de testhelft, bij elke leeftijd van 17 tot 90.
        """),
        L.task("Schrijf `ice(row, j, grid)`: voor elke waarde `v` in `grid`, wat `model` zegt voor een kopie van `row` met kolom `j` op `v`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_ice():
    box_ice = L.code_box("ice", lines=20)
    box_ice
    return (box_ice,)


@app.cell(hide_code=True)
def check_ice(box_ice):
    result_ice = L.check("ice", box_ice.value)
    result_ice
    return (result_ice,)


@app.cell(hide_code=True)
def hints_ice():
    L.hints(
        "ice",
        [
            "Maak een lege lijst voor de lijn. Loop de waarden van `grid` een voor een af: maak voor elke waarde een kopie van de rij waarin kolom `j` die waarde heeft, en voeg toe wat `model` zegt voor de kopie.",
            "Begin met `line = []` en `for v in grid:`. Maak binnen de loop `changed = list(row)`, zet `v` in kolom `j` van `changed`, en voeg `model(changed)` met `append` toe aan `line`. Geef na de loop `line` terug.",
            "`changed[___] = v`",
        ],
        why="`line = []`, dan `for v in grid:` met `changed = list(row)`, `changed[j] = v` en `line.append(model(changed))`. Na de loop `return line`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_average():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een vlak gemiddelde

        Het vak bevat het voorbeeld van Goldstein en collega's, met hun formule als model, zonder de ruis: voor een rij `[x1, x2, x3]` geeft het 0,2·x1 − 5·x2, plus 10·x2 als x3 ≥ 0. `random.seed(0)` zorgt dat de willekeurige getallen elke keer dezelfde zijn, en `random.uniform(-1, 1)` trekt een getal tussen −1 en 1, dus `rows` bevat 1000 rijen zoals de hunne. Vul de loop in, zodat `average` de partial dependence op X2, kolom 1, bevat bij elke waarde van `grid`: geef voor elke waarde elke rij die waarde in kolom 1, en neem het gemiddelde van wat `model` zegt. De tekening laat je gemiddelde zien onder de lijnen van 40 rijen.
        """),
        L.task("Vervang de `...`: tel voor elke `v` op wat `model` zegt voor elke rij met kolom 1 op `v`, en voeg het totaal gedeeld door `len(rows)` toe."),
    ])
    return


@app.cell(hide_code=True)
def exercise_average():
    box_average = L.code_box("average", lines=25)
    box_average
    return (box_average,)


@app.cell(hide_code=True)
def check_average(box_average):
    result_average = L.check("average", box_average.value)
    result_average
    return (result_average,)


@app.cell(hide_code=True)
def hints_average():
    L.hints(
        "average",
        [
            "Dit is de formule uit *Het idee*: voor elke waarde van de grid het gemiddelde over alle rijen van wat het model zegt als kolom 1 die waarde heeft. Je hebt dus een loop over de rijen nodig binnen de loop over de grid, en een totaal dat bij elke waarde op 0 begint.",
            "Begin binnen `for v in grid:` met `total = 0`, dan `for row in rows:` met `changed = list(row)`, kolom 1 van `changed` op `v`, en `total = total + model(changed)`. Voeg na de loop over de rijen het gemiddelde toe aan `average`.",
            "`average.append(total / ___)`",
        ],
        why="`total = 0`, dan `for row in rows:` met `changed = list(row)`, `changed[1] = v` en `total = total + model(changed)`; na die loop `average.append(total / len(rows))`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_flat():
    mo.md(r"""
    ### Oefening 3. Een vlakke lijn
    """)
    return


@app.cell(hide_code=True)
def choice_flat_cell():
    choice_flat = L.choice(
        "flat",
        question="Een collega tekent de partial dependence plot van één feature, en die is vlak. Kan je collega concluderen dat het model die feature negeert?",
        options={
            "scale": "Ja, als hij kansen laat zien",
            "right": "Nee: stijgende en dalende lijnen kunnen elkaar opheffen",
            "flat": "Ja: als het gemiddelde vlak is, dan is elke lijn dat ook",
        },
        correct="right",
        why={
            "right": "Goed. In het voorbeeld van Goldstein en collega's, waarvan je in oefening 2 het gemiddelde nam, is de partial dependence op X2 bijna vlak, terwijl elke lijn een helling van +5 of −5 heeft. Teken de ICE curves voor je iets concludeert uit een vlak gemiddelde.",
            "scale": "De schaal doet er hier niet toe: in oefening 2 geeft het model gewone getallen, en het gemiddelde over X2 is vlak terwijl elke lijn steil is. Zie *Als het gemiddelde vlak is* in *Het idee*.",
            "flat": "Een gemiddelde kan vlak zijn terwijl geen enkele lijn dat is: in oefening 2 stijgt ongeveer de helft van de lijnen met 10 en dalen de andere met 10, en je gemiddelde beweegt nauwelijks. Zie *Als het gemiddelde vlak is* in *Het idee*.",
        },
    )
    choice_flat
    return (choice_flat,)


@app.cell(hide_code=True)
def text_exercise_absurd():
    mo.md(r"""
    ### Oefening 4. Het linkeruiteinde
    """)
    return


@app.cell(hide_code=True)
def choice_absurd_cell():
    choice_absurd = L.choice(
        "absurd",
        question="In het uitgewerkte voorbeeld loopt het gemiddelde bij 17 jaar over 1000 mensen, van wie er 252 een opleiding van 13 of meer hebben. Geen van de 595 zeventienjarigen in de tabel heeft zoveel opleiding. Wat betekent dat voor het linkeruiteinde van de curve?",
        options={
            "few": "Het rust op te weinig mensen om te vertrouwen",
            "right": "Het bestaat deels uit rijen die nooit voorkomen",
            "causal": "Het laat zien wat een diploma doet voor een zeventienjarige",
        },
        correct="right",
        why={
            "right": "Goed. Leeftijd op 17 zetten laat ieders opleiding staan, dus een kwart van de rijen zijn zeventienjarigen met een diploma, en zo iemand heeft het forest nooit gezien: voor hen zegt het 0,102, voor de anderen 0,016. scikit-learn: bij features die samenhangen maak je onzinnige datapunten. Lees de curve waar de echte mensen zitten.",
            "few": "Het gemiddelde loopt bij elke leeftijd over 1000 mensen, ook bij 17. Het probleem is welke rijen: een kwart ervan zijn zeventienjarigen met een diploma, en die komen in de tabel niet voor. Zie *Rijen die niet bestaan* in *Het idee*.",
            "causal": "Een curve laat zien wat het model doet, niet wat er in de wereld gebeurt, en hier gokt het model: geen enkele zeventienjarige in de tabel heeft zoveel opleiding. Zie *Rijen die niet bestaan* en *Wat het je niet vertelt* in *Het idee*.",
        },
    )
    choice_absurd
    return (choice_absurd,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Welke features doen ertoe?](/course/open/rai-permutation-importance?lang=nl) mat hoeveel het forest op een feature leunt; deze les tekent welke kant de voorspelling op beweegt, en voor wie. Allebei beschrijven ze het hele model, alleen uit zijn antwoorden: het zijn global explanations, en ze zijn model-agnostic, zoals ingedeeld in [Soorten uitleg](/course/open/rai-kinds-of-explanation?lang=nl). Het forest is dat uit [Veel bomen stemmen: het random forest](/course/open/ml-random-forest?lang=nl), en de boom met diepte 2 stelt zijn vragen zoals in [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl). De volgende les, [Een klein model dat een groot model nadoet](/course/open/rai-global-surrogate?lang=nl), vat het hele forest samen in één kleine boom die je kunt lezen, en meet hoe vaak die het met het forest eens is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_absurd, choice_flat, result_average, result_ice):
    L.wrap_up(result_ice, result_average, choice_flat, choice_absurd)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
