# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-fairness-benchmarks. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-fairness-benchmarks", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [AI-claims kritisch lezen](/course/open/res-reading-claims?lang=nl) en [De beslissing aanpassen: een threshold per groep](/course/open/rai-fair-thresholds?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_benchmark():
    quick_benchmark = L.choice(
        "quick-benchmark",
        question="Een model scoort 0,90 op een benchmark van 500 taken, de hoogste score tot nu toe. Wat laat die score zien?",
        options={
            "all": "Hoe het doet op elke taak van die soort",
            "right": "Hoe het doet op die taken, op die manier gescoord",
            "best": "Dat het in die taak beter is dan alle andere modellen",
        },
        correct="right",
        why={
            "all": "Een benchmark is een vaste set testtaken. De score zegt hoe het model het deed op die taken, op die manier gescoord; of dat ook geldt voor andere taken van dezelfde soort, zegt de benchmark niet. Zie *Het idee* in [AI-claims kritisch lezen](/course/open/res-reading-claims?lang=nl).",
            "right": "Eén getal, op die taken, op die manier gescoord: dat is wat er gemeten is. Bij een fairness-benchmark is dat niet anders.",
            "best": "De hoogste score tot nu toe betekent dat het model het best deed op die taken, op die manier gescoord, van de modellen die erop gemeten zijn. Beter in de taak in het algemeen is niet gemeten. Zie *Het idee* in [AI-claims kritisch lezen](/course/open/res-reading-claims?lang=nl).",
        },
    )
    quick_benchmark
    return (quick_benchmark,)


@app.cell(hide_code=True)
def quick_check_threshold():
    quick_threshold = L.choice(
        "quick-threshold",
        question="Na een zoektocht krijgt een model een threshold van 0,38 voor vrouwen en 0,52 voor mannen. Wat veranderde er aan het model zelf?",
        options={
            "weights": "De weights, door opnieuw te trainen",
            "right": "Niets: alleen de beslissing veranderde",
            "labels": "De labels waarvan het leerde, voor sommige vrouwen",
        },
        correct="right",
        why={
            "weights": "Het model werd niet opnieuw getraind, en geen weight veranderde: alleen de thresholds waarmee de kansen vergeleken worden. Zie *Het idee* in [De beslissing aanpassen: een threshold per groep](/course/open/rai-fair-thresholds?lang=nl).",
            "right": "Het model geeft elke persoon dezelfde kans als eerst; alleen de threshold waarmee die kans vergeleken wordt, hangt af van de groep.",
            "labels": "Labels veranderen is massaging, voor het trainen. Een threshold per groep komt na het trainen en verandert geen label en geen weight van het model. Zie *Het idee* in [De beslissing aanpassen: een threshold per groep](/course/open/rai-fair-thresholds?lang=nl).",
        },
    )
    quick_threshold
    return (quick_threshold,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In de vorige lessen rekende je het deel met een ja, de recall, gewichten en thresholds zelf uit, in een paar regels Python. In de praktijk doen libraries dat rekenwerk, en twee bekende zijn Fairlearn en AIF360. En als een paper zegt dat een nieuwe methode de fairness van modellen verbetert, laat het dat vaak zien op dezelfde paar tabellen: UCI Adult, COMPAS en German Credit. Om zo'n claim te kunnen lezen, moet je weten wat de libraries uitrekenen, wat die tabellen zijn, en hoeveel één resultaat erop zegt. De demo laat zien hoeveel de toename door één methode verandert als je andere mensen uit dezelfde tabel neemt.

    **Een voorbeeld uit het dagelijks leven.** Een autofabrikant geeft op hoeveel brandstof een auto verbruikt, gemeten op een vaste testrit. Elke auto rijdt dezelfde rit, dus je kunt auto's vergelijken. Maar je eigen ritten, in de winter, in de heuvels of met een volle auto, kosten meer, en de auto die de testrit wint, wint niet altijd op jouw wegen. Een fairness-benchmark is zo'n testrit: voor elke methode dezelfde, maar niet de weg waarop de methode gebruikt gaat worden.
    """)
    return


@app.cell(hide_code=True)
def predict_spread():
    guess_spread = L.guess(
        "spread",
        question="Op de hele tabel bracht reweighing het deel met een ja van de vrouwen gedeeld door dat van de mannen van 0,28 naar 0,57. Neem nu 1000 mensen uit de tabel, zoveel als German Credit heeft, train op 700 van hen en meet op de andere 300, tien keer met andere mensen. Met hoeveel brengt reweighing die verhouding elke keer omhoog, denk je?",
        options={
            "same": "Elke keer met ongeveer 0,29, zoals op de hele tabel",
            "right": "Met alles tussen ongeveer 0,07 en 0,51",
            "worse": "In sommige steekproeven daalt de verhouding",
        },
        reveal="**Met alles tussen ongeveer 0,07 en 0,51.** Van 300 mensen om te testen zijn er maar ongeveer 100 vrouwen, en zonder gewichten krijgen er daarvan maar 5 tot 13 een ja: één vrouw meer of minder verschuift de verhouding met ongeveer 0,05. De demo hieronder laat het zien.",
    )
    guess_spread
    return (guess_spread,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Twee libraries.** **Fairlearn** is een library voor Python om fairness te meten en te verbeteren, onder de MIT-licentie; de nieuwste versie, 0.15.0, kwam uit in oktober 2026. **AIF360**, AI Fairness 360, noemt zichzelf een uitbreidbare opensource-library met technieken uit het onderzoek om bias in machine learning-modellen op te sporen en te verminderen, in de hele levensloop van een AI-toepassing. Het valt onder de Apache 2.0-licentie, en de nieuwste versie, 0.6.1, is van april 2024. AIF360 heeft reweighing als `Reweighing`, en Fairlearn heeft threshold optimization als `ThresholdOptimizer`. De kern is wat je zelf hebt gebouwd. `MetricFrame` van Fairlearn, volgens de documentatie een verzameling maten per groep, rekent een maat uit per groep, in `by_group`, en voor iedereen, in `overall`. `selection_rate` van Fairlearn, het selectiepercentage, is het deel met een ja: het deel van de voorspellingen dat de gunstige uitkomst is. Deze course installeert Fairlearn niet, dus de code hieronder is alleen om te lezen. Het is het voorbeeld uit de documentatie van Fairlearn:

    ```python
    from fairlearn.metrics import MetricFrame, selection_rate

    y_true = [1, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    y_pred = [0, 1, 1, 1, 1, 0, 0, 0, 1, 1]
    sex = ["Female", "Female", "Female", "Female", "Female", "Male", "Male", "Male", "Male", "Male"]
    frame = MetricFrame(metrics={"selection_rate": selection_rate},
                        y_true=y_true, y_pred=y_pred, sensitive_features=sex)
    frame.by_group        # Female 0.8, Male 0.4
    frame.difference()    # 0.4: de grootste min de kleinste
    frame.ratio()         # 0.5: de kleinste gedeeld door de grootste
    ```

    Het selectiepercentage heeft `y_true` niet nodig; Fairlearn vraagt er toch om, zodat alle maten op dezelfde manier aangeroepen worden, ook maten zoals de recall, die het wel nodig hebben. Oefening 1 rekent `by_group` van dit voorbeeld met de hand uit.

    **Dezelfde naam, andere getallen.** Lees welke formule een tool gebruikt. `demographic_parity_difference` van Fairlearn is het verschil tussen het grootste en het kleinste selectiepercentage van de groepen, dus nooit negatief, en `demographic_parity_ratio` deelt het kleinste door het grootste, dus nooit boven 1. AIF360 vraagt je welke groep privileged is en welke unprivileged. Zijn `statistical_parity_difference` is het deel met een ja van de unprivileged groep min dat van de privileged groep, en zijn `disparate_impact` deelt het ene door het andere. Volgens die formules geeft het voorbeeld hierboven, met de vrouwen als unprivileged groep, 0,8 min 0,4, dus 0,4, en 0,8 gedeeld door 0,4, dus 2: dezelfde voorspellingen geven een ratio van 0,5 in de ene library en 2 in de andere. En geen van beide libraries beslist wat fair is. De handleiding van Fairlearn zegt over de eigen methoden dat ze wiskundig garanderen hoe dicht ze een maat voor unfairness bij nul brengen, maar dat de uitkomst daarmee nog niet fair is.

    **De tabellen die iedereen gebruikt.** Een **fairness-benchmark** is een vaste tabel met een protected attribute, zoals geslacht of ras, en een taak, waarop methoden voor fairness vergeleken worden. Alessandro Fabris en collega's gingen 558 papers na uit belangrijke tijdschriften en conferenties met woorden over fairness in de titel, van 2014 tot juni 2021: UCI Adult werd 118 keer gebruikt, COMPAS 81 keer en German Credit 35 keer. Volgens hen werden de drie waarschijnlijk de standaard doordat vroege, veelgelezen papers en publicaties ze gebruikten. Elke tabel heeft een eigen probleem.

    - **UCI Adult** ken je: 48.842 mensen uit een Amerikaanse enquête van 1994, en het label zegt of iemand meer dan 50.000 dollar verdient. Frances Ding en collega's wijzen erop dat die inkomensgrens in 1994 bij het 0,76-kwantiel van alle inkomens in de Verenigde Staten lag, bij het 0,88-kwantiel onder zwarte mensen en bij het 0,89-kwantiel onder vrouwen. Bijna alle vrouwen verdienen dus minder, en een nee is voor hen meestal goed. Wat daaruit volgt, noemen ze ongewoon: een standaardmodel zit vaker goed bij vrouwen en bij zwarte mensen dan bij iedereen samen, terwijl modellen het vaak juist slechter doen bij groepen die achtergesteld werden. Het model van de vorige lessen ook: op de test set zit het goed bij 0,916 van de vrouwen en bij 0,794 van de mannen. Toen ze de inkomensgrens verschoven, veranderden zowel de accuracy als de verschillen tussen de groepen flink.
    - **German Credit** bevat 1000 leningen uit 1973 tot 1975 van een bank in Zuid-Duitsland, 700 goede en 300 slechte. Ulrike Grömping schrijft dat de slechte leningen sterk oververtegenwoordigd zijn: in werkelijkheid was ongeveer 5 procent slecht. Papers gebruiken de tabel om vrouwen en mannen te vergelijken, maar de tabel weet niet wie een vrouw is. De codetabel wekt de indruk dat één kolom geslacht en burgerlijke staat samen bevat, maar volgens Grömping klopt die codetabel niet: ongetrouwde mannen en vrouwen die niet ongetrouwd zijn, hebben dezelfde code, 2, dus het geslacht is er niet uit terug te halen. In 2019 publiceerde Grömping een verbeterde versie, South German Credit. Papers leidden het geslacht er toch uit af, onder andere de preprint uit 2018 van Sorelle Friedler en collega's, die hieronder terugkomt. Fabris en collega's vinden dat experimenten die een geslacht uit German Credit halen, op zijn best als onbetrouwbaar moeten gelden.
    - **COMPAS** ken je uit [COMPAS en een sollicitatietool](/course/open/rai-case-compas-amazon?lang=nl): verdachten in Broward County, Florida, uit 2013 en 2014. Het label zegt of iemand opnieuw gearresteerd werd, niet of iemand opnieuw een misdrijf pleegde, en waar de politie kijkt, beïnvloedt wie er gearresteerd wordt. Michelle Bao en collega's raden onderzoekers in fairness aan om tabellen uit het strafrecht, zoals COMPAS, niet te gebruiken als algemeen voorbeeld uit de echte wereld om een nieuwe methode of maat voor fairness te tonen of te benchmarken. Ze voegen eraan toe dat ze niet pleiten om data uit het strafrecht helemaal te vermijden.
    - **Folktables**, van Ding en collega's uit 2021, is een nieuwer antwoord: tabellen uit de American Community Survey, een jaarlijkse enquête die naar ongeveer 3,5 miljoen Amerikaanse huishoudens gaat, voor alle vijftig staten en Puerto Rico en de jaren 2014 tot en met 2018, met taken zoals inkomen en werk. De inkomenstaak, ACSIncome, heeft alleen al voor 2018 1.664.500 mensen, en de inkomensgrens kies je zelf. Het blijft één land: de makers noemen het zelf een duidelijke beperking dat alles over de Verenigde Staten gaat.

    **Eén tabel, één splitsing, één uitkomst.** Ding en collega's vergeleken methoden tussen staten: trainen op de ene staat en testen op een andere gaf meestal onvoorspelbare uitkomsten, en accuracy en fairness konden beide kanten op veranderen. Friedler en collega's vergeleken methoden voor fairness op Adult, German Credit, een tabel die Ricci heet en twee tabellen van COMPAS, elk met tien willekeurige splitsingen in mensen om op te trainen en mensen om te testen. Geen enkele methode deed het op alle tabellen beter dan de andere, in fairness en in accuracy tegelijk, en zelfs één methode gaf over de splitsingen heen enorm verschillende uitkomsten. Hun advies: rapporteer hoe goed en hoe stabiel een methode is over een redelijk aantal willekeurige splitsingen.

    **Probeer het.** De cel hieronder doet wat Friedler en collega's aanraden, voor reweighing, op steekproeven zo klein als German Credit. Hij neemt tien keer 1000 mensen uit UCI Adult, met `np.random.default_rng(_seed).permutation` en de seeds 0 tot en met 9: 700 om op te trainen en 300 om te testen. `_ratio` traint het model van de vorige lessen op de rijen op de plaatsen `train`, zonder gewichten of met reweighing zoals in [De data aanpassen: reweighing](/course/open/rai-reweighing?lang=nl), en geeft op de rijen `test` hoe vaak vrouwen een ja krijgen gedeeld door hoe vaak mannen er een krijgen. Voer de cel uit; dat duurt een seconde of twee.
    """)
    return


@app.cell
def demo_samples():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    # 48.842 mensen uit een Amerikaanse enquête van 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _rich = _people["income"] == ">50K"
    _poor = _people["income"] == "<=50K"
    _women = _people["sex"] == "Female"
    _men = _people["sex"] == "Male"

    # de kolommen van het model zonder sex, zoals in de vorige lessen
    _words = ["workclass", "marital_status", "occupation", "relationship", "race"]
    _columns = ["age", "education_num", "hours_per_week"] + _words
    _X = pd.get_dummies(_people[_columns], columns=_words, dtype=float)
    _X = _X / _X.max()


    def _ratio(train, test, reweigh):
        # train op de rijen op de plaatsen train, met of zonder reweighing, en meet op de rijen test:
        # hoe vaak vrouwen een ja krijgen, gedeeld door hoe vaak mannen er een krijgen
        weight = None
        if reweigh:
            weight = 0
            for sex in [_women.iloc[train], _men.iloc[train]]:
                for label in [_rich.iloc[train], _poor.iloc[train]]:
                    expected = sex.sum() * label.sum() / len(train)
                    weight = weight + expected / (sex * label).sum() * (sex * label)
        model = LogisticRegression(max_iter=1000)
        model.fit(_X.iloc[train], _rich.iloc[train], sample_weight=weight)
        yes = model.predict_proba(_X.iloc[test])[:, 1] > 0.5
        return yes[_women.iloc[test]].mean() / yes[_men.iloc[test]].mean()


    # tien steekproeven van 1000 mensen, zoveel als German Credit heeft: 700 om op te trainen, 300 om te testen
    for _seed in range(10):
        _pick = np.random.default_rng(_seed).permutation(len(_people))[:1000]
        _plain = _ratio(_pick[:700], _pick[700:], False)
        _reweighed = _ratio(_pick[:700], _pick[700:], True)
        print("seed", _seed, "| zonder gewichten", round(_plain, 2),
              "| met reweighing", round(_reweighed, 2))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.picture_samples({"title": "Eén methode, tien steekproeven van 1000 mensen",
                                   "whole": "hele tabel", "seed": "seed",
                                   "xlabel": "deel met een ja van vrouwen gedeeld door dat van mannen", "decimal": ",",
                                   "key": "open: zonder gewichten, gevuld: met reweighing. Bij 1 zouden vrouwen even vaak een ja krijgen als mannen. De bovenste rij is de hele tabel, uit De data aanpassen: reweighing.",
                                   "alt": "Elf rijen, elk met een open stip zonder gewichten en een gevulde stip met reweighing. Hele tabel: 0,28 en 0,57. Seeds 0 tot en met 9 zonder gewichten: 0,45, 0,36, 0,24, 0,56, 0,32, 0,55, 0,56, 0,31, 0,34, 0,22; met reweighing: 0,72, 0,57, 0,46, 0,63, 0,64, 0,78, 0,75, 0,82, 0,6, 0,67."})),
        mo.md(r"""
        Dezelfde tabel, dezelfde methode en hetzelfde model: alleen de mensen verschillen. Zonder gewichten ligt de verhouding tussen 0,22 en 0,56; met reweighing tussen 0,46 en 0,82. Bij seed 3 bracht reweighing de verhouding maar van 0,56 naar 0,63, bij seed 7 van 0,31 naar 0,82. Op de hele tabel ging ze van 0,28 naar 0,57. De oorzaak zijn de kleine aantallen. Van de 300 mensen om te testen zijn er 90 tot 109 vrouwen, en zonder gewichten krijgen er daarvan maar 5 tot 13 een ja: één vrouw meer of minder verschuift de verhouding met ongeveer 0,05. En van de 700 mensen om op te trainen zijn er maar 21 tot 33 vrouwen die meer verdienen, dus ook de gewichten zelf verschillen per steekproef. Hier bracht reweighing de verhouding in alle tien de steekproeven omhoog; met hoeveel, kan één splitsing van zo'n kleine tabel je niet vertellen. Een paper dat één splitsing van zo'n kleine tabel rapporteert, kan er net zo ver naast zitten.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **fairness-benchmark** | een vaste tabel met een protected attribute en een taak, zoals UCI Adult, COMPAS of German Credit, waarop methoden voor fairness vergeleken worden |
        | **Fairlearn** | een library voor fairness in Python, onder de MIT-licentie; zijn `MetricFrame` rekent een maat per groep uit |
        | **AIF360** | AI Fairness 360, een library voor fairness in Python en R, onder de Apache 2.0-licentie; je geeft aan welke groep privileged is |
        | **Folktables** | tabellen uit een jaarlijkse Amerikaanse enquête, per staat en per jaar, als nieuwere fairness-benchmark voor inkomen, werk en meer |
        """),
        L.watch_out("Een methode die op deze tabellen wint, heeft laten zien dat ze op deze tabellen wint. Fabris en collega's concluderen over Adult, COMPAS en German Credit dat je sceptisch moet zijn over grote claims voor de echte wereld die op experimenten met deze tabellen steunen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een uitkomst over fairness te lezen, uit een paper of uit een tool:

    1. Welke tabel: uit welk jaar en land, met hoeveel mensen, en wat meet het label? Kan de tabel überhaupt zien wie bij welke groep hoort?
    2. Welke formule: een verschil of een ratio, de grootste tegenover de kleinste, of de unprivileged groep tegenover de privileged groep?
    3. Hoeveel splitsingen, en hoe ver liggen de uitkomsten uit elkaar? Eén splitsing van een kleine tabel zegt weinig.
    4. Op welke andere tabellen, en op data zoals de data waarover het model gaat beslissen?
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Acht verzonnen mensen in twee groepen, a en b, en voor elk het antwoord van het model, 1 bij een ja. `_people` en `_yes` beginnen voor beide groepen op 0. De loop gaat de mensen langs op hun plaats `_i`, van 0 tot en met 7: `_g` is de groep van persoon `_i`, `_people[_g]` telt één persoon meer in die groep, en `_yes[_g]` telt het antwoord op. Daarna bevat `_rate` het selectiepercentage per groep, en de laatste twee regels rekenen het verschil en de ratio van Fairlearn uit.
    """)
    return


@app.cell
def example_rates():
    # acht verzonnen mensen: hun groep en het antwoord van het model, 1 bij een ja
    _group = ["a", "a", "b", "a", "b", "b", "a", "b"]
    _answer = [1, 1, 0, 0, 1, 0, 1, 0]

    _people = {"a": 0, "b": 0}
    _yes = {"a": 0, "b": 0}
    for _i in range(len(_group)):
        _g = _group[_i]
        _people[_g] = _people[_g] + 1
        _yes[_g] = _yes[_g] + _answer[_i]
    print("mensen:", _people, "| ja:", _yes)

    _rate = {"a": _yes["a"] / _people["a"], "b": _yes["b"] / _people["b"]}
    print("selectiepercentage:", _rate)
    print("verschil, de grootste min de kleinste:", _rate["a"] - _rate["b"])
    print("ratio, de kleinste gedeeld door de grootste:", round(_rate["b"] / _rate["a"], 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Groep a heeft 3 keer ja bij 4 mensen, een selectiepercentage van 0,75; groep b 1 bij 4, 0,25. Fairlearn zou een verschil van 0,5 en een ratio van 0,33 geven. AIF360 zou, met b als unprivileged groep, een statistical parity difference van 0,25 min 0,75, dus −0,5, en een disparate impact van 0,33 geven. Oefening 1 doet hetzelfde voor het voorbeeld uit de documentatie van Fairlearn.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rates():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Het voorbeeld van Fairlearn met de hand

        `sex` en `y_pred` zijn de tien mensen van het voorbeeld in *Het idee*. `people` en `yes` beginnen voor beide geslachten op 0, en de loop gaat de mensen langs op hun plaats `i`. Vervang de `...` door twee regels, zoals in het uitgewerkte voorbeeld: tel 1 op bij het aantal in `people` van het geslacht van persoon `i`, en tel het antwoord op bij het aantal in `yes`. De regel na de loop rekent `rate` uit, het selectiepercentage per geslacht. De check probeert ook andere mensen.
        """),
        L.task("Maak `rate`: per geslacht het deel van de mensen met een ja, zoals `MetricFrame` het in `by_group` geeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rates():
    box_rates = L.code_box("rates", lines=11)
    box_rates
    return (box_rates,)


@app.cell(hide_code=True)
def check_rates(box_rates):
    result_rates = L.check("rates", box_rates.value)
    result_rates
    return (result_rates,)


@app.cell(hide_code=True)
def hints_rates():
    L.hints(
        "rates",
        [
            "Doe wat het uitgewerkte voorbeeld deed: per persoon één meer in `people` voor het eigen geslacht, en het antwoord opgeteld bij `yes` voor het eigen geslacht.",
            "Het geslacht van persoon `i` is `sex[i]`, en het antwoord is `y_pred[i]`. De eerste regel is dus `people[sex[i]] = people[sex[i]] + 1`.",
            "`yes[sex[i]] = yes[sex[i]] + ___`.",
        ],
        why="`people[sex[i]] = people[sex[i]] + 1` en `yes[sex[i]] = yes[sex[i]] + y_pred[i]`. Dat geeft `{'Female': 0.8, 'Male': 0.4}`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_spread():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Hoe ver de toenames uit elkaar liggen

        `plain` en `reweighed` bevatten de tien verhoudingen van de demo, afgerond: hoe vaak vrouwen een ja krijgen gedeeld door hoe vaak mannen er een krijgen, zonder gewichten en met reweighing. `gain` begint als lege lijst, en de loop gaat de steekproeven langs op hun plaats `i`. Vervang de eerste `...` door één regel die aan `gain` toevoegt met hoeveel reweighing steekproef `i` omhoog bracht, en de tweede door de spreiding: de grootste toename min de kleinste, met `max` en `min`. De check probeert ook andere getallen.
        """),
        L.task("Vind `spread`: hoe ver de grootste en de kleinste toename door reweighing uit elkaar liggen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_spread():
    box_spread = L.code_box("spread", lines=9)
    box_spread
    return (box_spread,)


@app.cell(hide_code=True)
def check_spread(box_spread):
    result_spread = L.check("spread", box_spread.value)
    result_spread
    return (result_spread,)


@app.cell(hide_code=True)
def hints_spread():
    L.hints(
        "spread",
        [
            "Een toename is de verhouding met reweighing min de verhouding zonder, voor dezelfde steekproef. De spreiding is de grootste toename min de kleinste.",
            "In de loop: `gain.append(reweighed[i] - plain[i])`. Na de loop is `max(gain)` de grootste toename.",
            "`spread = max(gain) - ___`.",
        ],
        why="`gain.append(reweighed[i] - plain[i])` in de loop, en `spread = max(gain) - min(gain)` erna. Dat geeft 0,44.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_german():
    mo.md(r"""
    ### Oefening 3. Een uitkomst op German Credit
    """)
    return


@app.cell(hide_code=True)
def choice_german_cell():
    choice_german = L.choice(
        "german",
        question="Een paper vergelijkt hoe vaak vrouwen en mannen in German Credit een lening krijgen, en laat een nieuwe methode zien die de twee dichter bij elkaar brengt. Wat vraag je eerst?",
        options={
            "model": "Welk model het paper erop trainde",
            "right": "Hoe de tabel ziet wie een vrouw is",
            "size": "Of 1000 leningen genoeg zijn om een model te trainen",
        },
        correct="right",
        why={
            "model": "Het model doet ertoe, maar er komt een vraag voor: German Credit lijkt alleen een kolom voor geslacht te hebben. Grömping liet zien dat ongetrouwde mannen en vrouwen die niet ongetrouwd zijn dezelfde code hebben. Zie *Het idee*.",
            "right": "German Credit kan het niet zien: ongetrouwde mannen en vrouwen die niet ongetrouwd zijn, hebben dezelfde code. Wie op deze tabel vrouwen en mannen vergelijkt, vergelijkt groepen die geen vrouwen en mannen zijn.",
            "size": "1000 is weinig, en de demo liet zien hoeveel uitkomsten dan per splitsing verschillen. Maar eerst: de tabel kan niet zien wie een vrouw is, omdat ongetrouwde mannen en vrouwen die niet ongetrouwd zijn dezelfde code hebben. Zie *Het idee*.",
        },
    )
    choice_german
    return (choice_german,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [De data aanpassen: reweighing](/course/open/rai-reweighing?lang=nl) en [De beslissing aanpassen: een threshold per groep](/course/open/rai-fair-thresholds?lang=nl) bouwden twee methoden voor bias mitigation met de hand. Fairlearn en AIF360 hebben ze kant-en-klaar, soms met een andere formule achter dezelfde naam. En de gewone manier om te laten zien dat zo'n methode werkt, op UCI Adult, COMPAS of German Credit, zegt minder dan het lijkt: de tabellen zijn oud, kunnen niet altijd zien wie bij een groep hoort, of meten iets anders dan hun naam zegt, en één splitsing van een kleine tabel kan heel andere uitkomsten geven. [AI-claims kritisch lezen](/course/open/res-reading-claims?lang=nl) vroeg wat er precies gemeten is; vraag bij een uitkomst over fairness ook op welke tabel, met welke formule en over hoeveel splitsingen. De volgende les, *Project: meet de bias van je eigen classifier*, brengt alles samen op een model van jezelf.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_german, result_rates, result_spread):
    L.wrap_up(result_rates, result_spread, choice_german)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
