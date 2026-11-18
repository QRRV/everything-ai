# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-lime. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-lime", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een klein model dat een groot model nadoet](/course/open/rai-global-surrogate?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_fidelity():
    quick_fidelity = L.choice(
        "quick-fidelity",
        question="Een boom met diepte 3 groeit op de antwoorden van het forest. Wat meet zijn fidelity op de testhelft?",
        options={
            "right": "Hoe vaak hij zegt wat het forest zegt",
            "accuracy": "Hoe vaak hij goed zit met wie meer verdient",
            "leaves": "Hoeveel van zijn bladeren ja zeggen",
        },
        correct="right",
        why={
            "right": "Goed: fidelity vergelijkt de kopie met het model dat hij nadoet, niet met de werkelijkheid, op rijen waarvan hij niet leerde. In deze les wordt opnieuw een kopie gemaakt, maar die hoeft het alleen dicht bij één persoon met het forest eens te zijn.",
            "accuracy": "Dat is accuracy, die met de werkelijkheid vergelijkt. Fidelity vergelijkt de kopie met het forest: hoe vaak ze hetzelfde antwoord geven. Zie [Een klein model dat een groot model nadoet](/course/open/rai-global-surrogate?lang=nl).",
            "leaves": "Fidelity gaat over antwoorden, niet over de vorm van de boom: het deel van de rijen waarop de kopie zegt wat het forest zegt. Zie [Een klein model dat een groot model nadoet](/course/open/rai-global-surrogate?lang=nl).",
        },
    )
    quick_fidelity
    return (quick_fidelity,)


@app.cell(hide_code=True)
def quick_check_group():
    quick_group = L.choice(
        "quick-group",
        question="De kopie met diepte 3 is het met het forest eens bij 98,8% van de mensen met een opleiding tot 12, en bij 86,3% van wie 13 of meer heeft. Je moet het antwoord van het forest uitleggen voor één persoon met opleiding 13. Hoeveel helpt de kopie?",
        options={
            "fully": "Helemaal: hij is het eens bij 95,7% van iedereen",
            "right": "Minder: hij zit bij één op de zeven naast",
            "none": "Niet: een kopie zegt over niemand iets",
        },
        correct="right",
        why={
            "right": "Goed: bij mensen met een opleiding van 13 of meer geeft de kopie bij 13,7% een ander antwoord dan het forest, ongeveer één op de zeven. Een kopie voor iedereen kan net naast zitten waar jouw persoon staat. Deze les doet het forest alleen dicht bij die ene persoon na.",
            "fully": "95,7% geldt voor iedereen samen. Bij mensen met een opleiding van 13 of meer is de kopie het eens bij 86,3%, dus hij zit naast bij ongeveer één op de zeven zoals deze persoon. Zie [Een klein model dat een groot model nadoet](/course/open/rai-global-surrogate?lang=nl).",
            "none": "Bij mensen met een opleiding van 13 of meer is de kopie het nog steeds bij 86,3% eens met het forest: hij helpt, maar zit naast bij ongeveer één op de zeven zoals deze persoon. Zie [Een klein model dat een groot model nadoet](/course/open/rai-global-surrogate?lang=nl).",
        },
    )
    quick_group
    return (quick_group,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De toezichthouder van de bank heeft nu een klacht op het bureau. Iemand van 41, met een bachelordiploma (education_num 13), die 40 uur per week werkt, kreeg nee: het forest geeft een kans van 0,475 dat deze persoon meer dan 50.000 dollar verdient, net onder de helft. De persoon vraagt welke van de drie features het zwaarst woog. De kopie uit de vorige les vat het forest samen voor iedereen, en bij mensen met een diploma zit hij naast bij ongeveer één op de zeven. Doe het forest dus opnieuw na, maar alleen dicht bij deze persoon: maak veel rijen, vraag het forest naar elk ervan, en trek een rechte lijn waarin rijen dicht bij de persoon het zwaarst tellen. Dat is LIME, en de hellingen van die lijn zeggen hoe de kans van het forest meebeweegt dicht bij deze ene persoon. Hoe dichtbij "dichtbij" is, blijkt een keuze, en het antwoord verandert mee.

    **Een voorbeeld uit het dagelijks leven.** Om te weten wat een pan soep nog nodig heeft, proef je lepels met kleine veranderingen: een snufje zout erbij, een druppel citroen, een beetje water. Wat je leert, geldt voor deze pan, niet voor elke soep. En hoe groot je veranderingen zijn, maakt uit: met een snufje leer je iets over deze pan; met een kop zout leer je alleen dat een kop zout te veel is.
    """)
    return


@app.cell(hide_code=True)
def predict_order():
    guess_order = L.guess(
        "order",
        question="Met de eigen instelling van lime voor hoe dichtbij rijen moeten liggen, kernel width 1,3, weegt education_num voor deze persoon het zwaarst. Met een width van 0,2 tellen alleen rijen heel dicht bij de persoon mee. Welke feature weegt dan het zwaarst?",
        options={
            "age": "age",
            "education": "education_num, net als eerst",
            "hours": "hours_per_week",
        },
        reveal="**hours_per_week, in 9 van de 10 runs.** Heel dicht bij de persoon springt de kans van het forest mee met de uren: van 0,394 bij 39 uur naar 0,610 bij 42 uur. Met width 1,3 weegt education_num het zwaarst in alle 10 runs: gemiddeld 0,144, tegen 0,108 voor hours_per_week en 0,107 voor age. Het forest is hetzelfde; alleen hoe dichtbij \"dichtbij\" is, veranderde.",
    )
    guess_order
    return (guess_order,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Eén voorspelling, één klein model.** Marco Tulio Ribeiro, Sameer Singh en Carlos Guestrin (2016) noemden hun methode LIME, voor Local Interpretable Model-agnostic Explanations. Hun doel: een model vinden dat je kunt lezen en dat het model dat je uitlegt plaatselijk trouw nadoet. Plaatselijk trouw is fidelity, zoals in de vorige les, maar alleen dicht bij één voorbeeld: het kleine model is een local surrogate. Ze wegen twee dingen tegen elkaar af: hoe slecht het kleine model het grote model nadoet in de buurt van het voorbeeld, waarbij rijen zwaarder tellen naarmate ze dichterbij liggen, en hoe ingewikkeld het is. In hun versie is het kleine model een rechte lijn over een paar features. Ze waarschuwen dat plaatselijke fidelity geen fidelity overal betekent, en dat er misschien geen trouwe uitleg bestaat als het model zelfs dicht bij de voorspelling sterk niet-lineair is.

        **Wat het vond.** Ribeiro en anderen trainden een classifier op 20 afbeeldingen, met de hand zo gekozen dat op alle foto's van wolven sneeuw lag en op die van husky's niet. Van 27 studenten na hun bachelor, met minstens één vak machine learning op dat niveau, vertrouwden er 10 de classifier voordat ze uitleg zagen, en 12 noemden sneeuw als iets wat hij misschien gebruikte; na de uitleg van LIME vertrouwden er 3 hem en noemden 25 de sneeuw. De auteurs zeggen er zelf bij dat hun groep klein was. In een andere proef sorteerde een classifier teksten over christendom en atheïsme met 94% accuracy op achtergehouden data, terwijl het woord "Posting" in 22% van de voorbeelden om op te trainen stond, en daarvan 99% in de klasse atheïsme. Op een nieuwe verzameling webpagina's haalde hij 57,3%, en een opgeschoonde versie 69,0%, al haalde die opgeschoonde versie op de oorspronkelijke testdata minder: 88,6% tegen 94,0%.

        **Hoe dichtbij is dichtbij?** Elke verzonnen rij telt mee naar zijn nabijheid tot de persoon, $\exp(-d^2 / (2w^2))$: $d$ is de afstand van de rij tot de persoon, met elke kolom in standaardafwijkingen zoals in [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl), en $w$ is de kernel width. Een rij op de plek van de persoon telt 1; een rij ver weg bijna 0. Het lime-package neemt 0,75 keer de wortel uit het aantal kolommen, 1,3 bij drie kolommen. (Het paper schrijft $\exp(-d^2/w^2)$; het package neemt daar de wortel uit, en dat is de formule hierboven.) Een goede regel ervoor is er niet. Christoph Molnar noemt de keuze van de buurt een onopgelost probleem als je LIME op tabellen gebruikt. In hun analyse van LIME op tabellen, voor een model dat zelf een rechte lijn is, laten Damien Garreau en Ulrike von Luxburg zien dat bij een heel grote width alle voorbeelden meetellen, dat de lijn bij een te kleine width alleen een constante wordt die alle belangrijke informatie mist, en dat slecht gekozen instellingen LIME belangrijke features kunnen laten missen.

        **Waar de rijen vandaan komen.** Het paper maakt zijn rijen door een willekeurig deel van het voorbeeld te houden, zoals een paar woorden van een tekst. Bij een tabel beslist het package. Standaard knipt het elke kolom in vier bakjes bij zijn kwartielen, de waarden op een kwart, de helft en driekwart van de gesorteerde rijen om op te trainen, en trekt het elke kolom uit diezelfde rijen; een weight betekent dan "in hetzelfde bakje als deze persoon". Zonder bakjes (`discretize_continuous=False`) trekt het elke kolom willekeurig rond het gemiddelde van die kolom, de meeste waarden dichtbij en weinig ver weg, met de eigen spreiding van de kolom: een normale verdeling. Met beide instellingen worden de rijen niet rond de persoon getrokken, tenzij je `sample_around_instance=True` meegeeft; Molnar wijst erop dat ze uit het zwaartepunt van de data om op te trainen komen, en noemt dat een probleem. De persoon komt binnen via de nabijheid. De LIME van deze les doet wat het package zonder bakjes doet, behalve dat hij een gewone rechte lijn fit waar het package ridge regression fit, een rechte lijn die zijn weights kleiner houdt. Bij de width 1,3 van lime, waar de rijen even zwaar tellen als 1965 rijen op de plek van de persoon, geven de twee hier dezelfde weights, op drie decimalen. Bij kleine widths trekt ridge de weights naar 0: bij width 0,1 geeft hij 0,004, 0,004 en 0,013, dicht bij de constante die Garreau en von Luxburg beschrijven.

        **Nog een keer.** Ribeiro en anderen noemen hun methode tamelijk robuust tegen toeval in de steekproef, omdat de rijen gewogen worden. Anderen zijn het daar niet mee eens. Molnar merkt uit eigen ervaring dat er een andere uitleg kan uitkomen als je het trekken van de rijen herhaalt. Giorgio Visani en vier collega's draaiden LIME 10 keer voor één kredietaanvrager, met een model op meer kolommen dan hier; hun maat voor hoe goed dezelfde 7 belangrijkste features terugkwamen was 89,44% bij de ene instelling (kernel width 3, ridge-straf 1) en 14,17% bij een andere (width 1,3, straf 0,001). En David Alvarez-Melis en Tommi Jaakkola vonden op een verzonnen dataset met twee kolommen dat LIME en SHAP, de methode van de volgende les, voor een klein neural network bij sommige inputs die vlak bij elkaar liggen flink verschillende uitleg geven.

        **Met opzet misleid.** De rijen van LIME hoeven niet op echte mensen te lijken: in het uitgewerkte voorbeeld liggen 324 van de 5000 buiten wat de tabel bevat, zoals mensen jonger dan 17. Dylan Slack, Sophie Hilgard, Emily Jia, Sameer Singh en Himabindu Lakkaraju (2020) maakten daar gebruik van. Hun model beslist voor echte mensen alleen op ras, en voor rijen die verzonnen lijken op een kolom die niets betekent; een random forest houdt de twee uit elkaar. Op data over verdachten (COMPAS) zette LIME daarna bij 100% van de achtergehouden mensen de betekenisloze kolom bovenaan in plaats van ras. Kernel widths van 0,15 tot 0,95 keer de wortel uit het aantal kolommen hielpen niet: de aanvallen bleven even goed werken. Net als deze les gebruikten ze lime zonder bakjes. Molnar: wie LIME gebruikt, kan de uitleg zo sturen dat vooroordelen verborgen blijven.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **LIME** | een methode die één voorspelling uitlegt: maak veel rijen, vraag het model ernaar, en fit een rechte lijn waarin rijen dicht bij het voorbeeld het zwaarst tellen; de weights van die lijn zijn de uitleg |
        | **local surrogate** | een model dat je kunt lezen en dat een black box alleen dicht bij één voorbeeld nadoet |
        | **kernel width** | tot hoe ver van het voorbeeld rijen nog meetellen: klein, en alleen rijen heel dichtbij tellen; groot, en alle rijen tellen bijna even zwaar |
        | $\exp(-d^2 / (2w^2))$ | de nabijheid van een rij op afstand $d$ bij kernel width $w$: 1 bij het voorbeeld, bijna 0 ver weg |
        | `predict_proba` | in scikit-learn: voor elke rij de kans op elke klasse; kolom 1 is de kans op ja |
        | `sample_weight` | in `fit` van scikit-learn: hoe zwaar elke rij telt; een rij met 0,5 telt half |
        """),
        L.watch_out("Een weight van LIME is een helling van een rechte lijn dicht bij één persoon, gemaakt uit willekeurige rijen en een kernel width die jij koos. Het is niet de reden van het forest, het verandert met de width en van run tot run, en een model kan zo gebouwd zijn dat het voor LIME verbergt wat het doet. Draai LIME een paar keer en met een paar widths, en vertrouw alleen wat blijft staan."),
    ])
    return


@app.cell(hide_code=True)
def explore_width_cell():
    explore_width = L.explore("width", "kernel width", lambda v: C.width_picture(v, "nl"), start=1.3,
                              values=C.WIDTHS, role="hyper",
                              task="Schuif van width 0,1 naar 10. Bij welke widths weegt hours_per_week het zwaarst, en bij welke zijn de 10 runs het oneens met elkaar?")
    mo.vstack([
        mo.md("**De weights van LIME voor één persoon, per kernel width**"),
        explore_width,
        mo.md(r"""
        De LIME van de course, gemaakt zoals in het uitgewerkte voorbeeld verderop, voor de persoon van 41 met education_num 13 die 40 uur werkt: 10 runs, elk met zijn eigen 5000 willekeurige rijen. Elk punt is de weight die één run aan een kolom geeft; het korte streepje is hun gemiddelde. Een weight zegt hoeveel de kans op ja van de lijn stijgt bij één standaardafwijking meer in die kolom: 13,7 jaar leeftijd, 2,6 niveaus opleiding, 12,5 uur per week.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo leg je één voorspelling uit met LIME:

    1. Neem de rij die je uitlegt, en het antwoord van het model daarop.
    2. Maak veel rijen waarop het model antwoord kan geven, en vraag het model naar elk ervan.
    3. Geef elke rij een nabijheid, uit zijn afstand tot de rij die je uitlegt en een kernel width.
    4. Fit een rechte lijn door de rijen, elke rij geteld naar zijn nabijheid: de weights zijn de uitleg.
    5. Lees een weight als hoeveel het antwoord dicht bij deze rij meebeweegt, per standaardafwijking van zijn kolom.
    6. Draai het opnieuw met andere willekeurige rijen en andere widths, en vertrouw alleen wat blijft staan.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het forest uit de vorige lessen, en de persoon op plaats 5 van de testhelft. `.predict_proba(rows)[:, 1]` geeft voor elke rij de kans op ja van het forest. `_mean` en `_std` bevatten het gemiddelde en de standaardafwijking van elke kolom van de helft om op te trainen: `.mean(axis=0)` en `.std(axis=0)` werken kolom voor kolom. `_lime(seed, width, person)` is één run voor één persoon. `.normal(size=(5000, 3))` van de random generator geeft 5000 rijen van 3 getallen, elk getrokken rond 0 met spreiding 1, dus `_mean + z * _std` zijn rijen rond het gemiddelde van elke kolom; `z` zijn dezelfde rijen in standaardafwijkingen. `(person - _mean) / _std` is de persoon, ook in standaardafwijkingen, en `np.linalg.norm(..., axis=1)` geeft de lengte van elke rij, hier zijn afstand tot de persoon. `LinearRegression().fit(z, chance, sample_weight=closeness)` fit de rechte lijn, elke rij geteld naar zijn nabijheid, en `.coef_` bevat de drie weights. Voer de cel uit. Dat duurt een paar seconden.
    """)
    return


@app.cell
def example_lime():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LinearRegression

    # 1. het forest uit de vorige lessen, en de persoon die je uitlegt
    _people = pd.read_csv("data/adult.csv.gz")
    _columns = ["age", "education_num", "hours_per_week"]
    _X = _people[_columns].to_numpy().astype(float)
    _y = (_people["income"] == ">50K").to_numpy()
    _order = np.random.default_rng(0).permutation(len(_y))
    _train, _test = _order[: len(_y) // 2], _order[len(_y) // 2:]
    _forest = RandomForestClassifier(min_samples_leaf=20, random_state=0).fit(_X[_train], _y[_train])
    _person = _X[_test][5]
    print("persoon:", _person, " kans op ja:", _forest.predict_proba([_person])[0, 1].round(3))
    _mean, _std = _X[_train].mean(axis=0), _X[_train].std(axis=0)

    def _lime(seed, width, person):
        # 2. 5000 verzonnen rijen, elke kolom getrokken rond zijn gemiddelde, en de kans op ja van het forest
        z = np.random.default_rng(seed).normal(size=(5000, 3))
        chance = _forest.predict_proba(_mean + z * _std)[:, 1]
        # 3. de persoon in standaardafwijkingen, de afstand van elke rij daartoe, en hoe zwaar de rij telt
        d = np.linalg.norm(z - (person - _mean) / _std, axis=1)
        closeness = np.exp(-d ** 2 / (2 * width ** 2))
        # 4. een rechte lijn door de rijen, elke rij geteld naar zijn nabijheid: de weights zijn de uitleg
        line = LinearRegression().fit(z, chance, sample_weight=closeness)
        return line.coef_, closeness.sum()

    # 5. de eigen width van lime bij drie kolommen
    _weights, _total = _lime(0, 0.75 * 3 ** 0.5, _person)
    print("width 1.3: de rijen tellen even zwaar als", _total.round(1), "rijen op de plek van de persoon")
    for _j in range(3):
        print("  weight van", _columns[_j], _weights[_j].round(3))

    # 6. andere willekeurige rijen en widths, en als hoeveel rijen op de plek van de persoon ze tellen
    for _width in [0.1, 0.2, 0.75 * 3 ** 0.5, 10]:
        for _seed in range(3):
            _weights, _total = _lime(_seed, _width, _person)
            print("width", round(_width, 1), " seed", _seed, " weights", _weights.round(3),
                  " rijen", _total.round(1))

    # 7. een heel brede kernel voor een andere persoon: de eerste van de testhelft
    _other = _X[_test][0]
    print("andere persoon:", _other, " width 10, seed 0, weights", _lime(0, 10, _other)[0].round(3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met de width van lime tellen de 5000 rijen even zwaar als 1965,1 rijen op de plek van de persoon, en education_num weegt het zwaarst: 0,144, dus dicht bij deze persoon stijgt de kans op ja van de lijn met 0,144 bij 2,6 niveaus meer opleiding. hours_per_week en age volgen met 0,106 en 0,105, zo dicht bij elkaar dat age in 4 van de 10 runs van de schuifbalk vóór hours_per_week komt. Met width 0,2 weegt hours_per_week het zwaarst in alle drie de runs hier, en met width 0,1 tellen de rijen even zwaar als maar ongeveer 2 rijen op de plek van de persoon, en springt de weight van education_num van run tot run van 0,081 naar 0,199 naar 0,242. Met width 10 telt bijna elke rij volledig mee: de weights zijn 0,082, 0,111 en 0,073, en in de eerste run zijn ze op drie decimalen gelijk voor een andere persoon: iemand van 47 met education_num 10 die 45 uur werkt. Dan legt LIME deze persoon niet meer uit, maar fit hij één lijn voor iedereen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_closeness():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Hoe zwaar telt een rij?

        Schrijf `closeness(distances, width)`: `distances` is een lijst met afstanden tot de persoon, in standaardafwijkingen, en de functie geeft een lijst terug met de nabijheid van elke afstand `d`, $\exp(-d^2 / (2w^2))$ met $w$ de `width`. `math.exp(x)` is $e^x$, en `d ** 2` is $d^2$. De check roept eerst `closeness([0, 1, 2], 1)` aan, daarna andere lijsten. De tekening laat je functie zien voor drie widths, en telt hem op over de 5000 rijen uit het uitgewerkte voorbeeld.
        """),
        L.task("Schrijf `closeness(distances, width)`: voeg voor elke afstand `d` in `distances` zijn nabijheid toe aan een lijst `out`, en geef `out` terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_closeness():
    box_closeness = L.code_box("closeness", lines=10)
    box_closeness
    return (box_closeness,)


@app.cell(hide_code=True)
def check_closeness(box_closeness):
    result_closeness = L.check("closeness", box_closeness.value)
    result_closeness
    return (result_closeness,)


@app.cell(hide_code=True)
def hints_closeness():
    L.hints(
        "closeness",
        [
            "Een rij op de plek van de persoon, op afstand 0, telt 1; hoe verder weg, hoe minder. Reken de formule uit voor elke afstand, en verzamel de uitkomsten in een lijst.",
            "Begin met `out = []` en loop langs de lijst met `for d in distances:`. Voeg voor elke `d` de waarde van `math.exp(...)` met de formule erin toe; geef na de loop de lijst terug.",
            "`out.append(math.exp(-d ** 2 / (2 * ___)))`",
        ],
        why="`out = []`, dan `for d in distances:` met `out.append(math.exp(-d ** 2 / (2 * width ** 2)))`. Na de loop `return out`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_slope():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De helling dicht bij de persoon

        De lijn van LIME heeft een helling voor elke kolom. Met één kolom kun je die met de hand uitrekenen. Met $c_i$ de nabijheid van rij $i$, en $\bar x$ en $\bar y$ de gemiddelden van `xs` en `ys` waarin elke rij telt naar zijn nabijheid:

        $$\text{helling} = \frac{\sum_i c_i\,(x_i - \bar x)(y_i - \bar y)}{\sum_i c_i\,(x_i - \bar x)^2}$$

        Het vak rekent de twee gemiddelden al uit, `mean_x` en `mean_y`. Maak `local_slope(xs, ys, closeness)` af: tel de teller van de breuk op in `up` en de noemer in `down`, en loop daarbij langs de rijen met `i`; geef `up / down` terug. De check roept eerst `local_slope([1, 2, 3, 4], [1, 3, 2, 4], [1, 1, 1, 0])` aan, waarin de laatste rij 0 telt. De tekening voert je functie uit op 5000 rijen voor alleen hours_per_week, met de leeftijd en de opleiding van de persoon vast: voor width 0,3 en 1,3, en als elke rij volledig meetelt.
        """),
        L.task("Vervang de `...`: begin `up` en `down` bij 0, tel in een loop over `i` voor elke rij zijn term van de teller op bij `up` en zijn term van de noemer bij `down`, en geef `up / down` terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_slope():
    box_slope = L.code_box("slope", lines=15)
    box_slope
    return (box_slope,)


@app.cell(hide_code=True)
def check_slope(box_slope):
    result_slope = L.check("slope", box_slope.value)
    result_slope
    return (result_slope,)


@app.cell(hide_code=True)
def hints_slope():
    L.hints(
        "slope",
        [
            "De teller vermenigvuldigt voor elke rij zijn nabijheid met hoe ver zijn x van `mean_x` ligt en hoe ver zijn y van `mean_y` ligt; de noemer gebruikt hoe ver zijn x ligt, in het kwadraat. Tel elk op over alle rijen.",
            "Begin met `up = 0` en `down = 0`, en dan `for i in range(len(xs)):`. Tel in de loop de term van de teller van de rij op bij `up` en de term van de noemer bij `down`; geef na de loop `up / down` terug.",
            "`up = up + closeness[i] * (xs[i] - mean_x) * ___`",
        ],
        why="`up = 0` en `down = 0`, dan `for i in range(len(xs)):` met `up = up + closeness[i] * (xs[i] - mean_x) * (ys[i] - mean_y)` en `down = down + closeness[i] * (xs[i] - mean_x) ** 2`. Na de loop `return up / down`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_wide():
    mo.md(r"""
    ### Oefening 3. Een heel brede kernel
    """)
    return


@app.cell(hide_code=True)
def choice_wide_cell():
    choice_wide = L.choice(
        "wide",
        question="Met kernel width 10 telt bijna elke rij van de 5000 volledig mee. Wat zegt LIME dan over de persoon van 41 uit het uitgewerkte voorbeeld?",
        options={
            "right": "Bijna wat het voor ieder ander zegt",
            "exact": "De eigen redenen van het forest voor deze persoon",
            "none": "Niets: geen rij ligt dichtbij genoeg",
        },
        correct="right",
        why={
            "right": "Goed. Met width 10 hangt de nabijheid nauwelijks af van de afstand, dus de lijn wordt aan alle rijen even zwaar gefit: in de eerste run van het uitgewerkte voorbeeld zijn de weights 0,082, 0,111 en 0,073, voor deze persoon en ook voor iemand van 47 met education_num 10 die 45 uur werkt. Volgens Garreau en von Luxburg tellen bij een heel grote width alle voorbeelden mee. De uitleg is dan een global explanation, één lijn voor iedereen.",
            "exact": "De weights van LIME zijn hellingen van een rechte lijn die op de antwoorden van het forest gefit is, nooit de eigen redenen van het forest. En met width 10 wordt die lijn aan alle rijen even zwaar gefit, dus hij is voor iedereen bijna hetzelfde. Zie *Hoe dichtbij is dichtbij?* in *Het idee*.",
            "none": "Een bredere kernel laat rijen zwaarder tellen, niet lichter: met width 10 telt een rij op 3 standaardafwijkingen nog 0,956. De lijn wordt dan aan alle rijen even zwaar gefit, voor iedereen bijna hetzelfde. Zie *Hoe dichtbij is dichtbij?* in *Het idee*.",
        },
    )
    choice_wide
    return (choice_wide,)


@app.cell(hide_code=True)
def text_exercise_fooled():
    mo.md(r"""
    ### Oefening 4. Een model dat iets verbergt
    """)
    return


@app.cell(hide_code=True)
def choice_fooled_cell():
    choice_fooled = L.choice(
        "fooled",
        question="Een model beslist voor echte aanvragers alleen op ras, en voor rijen die niet echt lijken op een kolom die niets betekent. LIME legt het antwoord voor één echte aanvrager uit. Wat laat het zien?",
        options={
            "race": "Ras, omdat het model daarop beslist",
            "right": "De kolom die niets betekent",
            "both": "Allebei, elk ongeveer half zo zwaar",
        },
        correct="right",
        why={
            "right": "Goed. LIME vraagt het model naar verzonnen rijen, en die hoeven niet op echte mensen te lijken: in het uitgewerkte voorbeeld liggen 324 van de 5000 buiten wat de tabel bevat. Dus antwoordt het model LIME met de kolom die niets betekent, en LIME meldt die kolom. Bij het model van Slack en anderen gebeurde dat voor 100% van de mensen die ze in COMPAS uitlegden.",
            "race": "Het model gebruikt ras alleen voor rijen die echt lijken. De rijen van LIME zijn verzonnen, en het model ziet dat: het antwoordt ze met de andere kolom, dus die kolom ziet LIME. Zie *Met opzet misleid* in *Het idee*.",
            "both": "LIME ziet alleen de antwoorden van het model op zijn eigen verzonnen rijen, en op de meeste daarvan gebruikt het model geen ras. Zie *Met opzet misleid* in *Het idee*.",
        },
    )
    choice_fooled
    return (choice_fooled,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Een klein model dat een groot model nadoet](/course/open/rai-global-surrogate?lang=nl) deed het forest na voor iedereen; LIME doet het na dicht bij één persoon, en is een local explanation in de indeling van [Soorten uitleg](/course/open/rai-kinds-of-explanation?lang=nl). De afstand is die uit [Lengte, afstand en cosine similarity](/course/open/math-norms-cosine?lang=nl), gemeten in standaardafwijkingen zoals in [Waarom gradient descent van geschaalde features houdt](/course/open/ml-feature-scaling?lang=nl), en de lijn is de lineaire regressie uit [Lineaire regressie met veel features](/course/open/ml-linear-regression-numpy?lang=nl), waarin sommige rijen zwaarder tellen dan andere. De volgende les, [Shapley values](/course/open/rai-shapley-values?lang=nl), legt ook één voorspelling uit, maar verdeelt die over de features met een regel die niet van een kernel width afhangt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fooled, choice_wide, result_closeness, result_slope):
    L.wrap_up(result_closeness, result_slope, choice_wide, choice_fooled)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
