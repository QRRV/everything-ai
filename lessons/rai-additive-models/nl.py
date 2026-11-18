# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-additive-models. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-additive-models", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een korte vraag over [Glass-box models](/course/open/rai-glass-box-models?lang=nl) en een over [Boom na boom op wat overblijft: gradient boosting](/course/open/ml-gradient-boosting?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_odds():
    quick_odds = L.choice(
        "quick-odds",
        question="In een logistische regressie heeft opleiding de weight 0,345. Wat doet één opleidingsniveau meer met de odds op een ja, als de andere features gelijk blijven?",
        options={
            "add": "Het telt 0,345 op bij de odds",
            "right": "Het vermenigvuldigt ze met $e^{0{,}345}$",
            "chance": "Het vermenigvuldigt de kans op een ja met 1,412",
        },
        correct="right",
        why={
            "add": "De 0,345 komt bij de logit, de log van de odds. De odds zelf worden vermenigvuldigd met $e^{0{,}345}$, ongeveer 1,412. Zie [Glass-box models](/course/open/rai-glass-box-models?lang=nl).",
            "right": "Goed: de weight telt op bij de logit, dus de odds worden vermenigvuldigd met $e^{0{,}345}$, ongeveer 1,412. In deze les lees je een hele curve op dezelfde manier: aan wat ze bij de score optelt.",
            "chance": "1,412 is de factor op de odds, niet op de kans; wat hij met een kans doet, hangt af van waar je begint. Zie [Glass-box models](/course/open/rai-glass-box-models?lang=nl).",
        },
    )
    quick_odds
    return (quick_odds,)


@app.cell(hide_code=True)
def quick_check_residual():
    quick_residual = L.choice(
        "quick-residual",
        question="Wat leert bij gradient boosting elke nieuwe kleine boom?",
        options={
            "labels": "De labels zelf, helemaal opnieuw",
            "right": "Wat de bomen ervoor nog fout deden",
            "half": "Elke keer een willekeurige helft van de features",
        },
        correct="right",
        why={
            "labels": "Elke boom leert wat nog fout is: de residuen van de bomen ervoor. Zie [Boom na boom op wat overblijft: gradient boosting](/course/open/ml-gradient-boosting?lang=nl).",
            "right": "Goed: elke boom leert wat de bomen ervoor nog fout doen, en een deel van wat hij voorspelt, de learning rate, komt erbij. In deze les krijgt elke boom bovendien maar één kolom.",
            "half": "Willekeurig features kiezen doet een random forest. Bij boosting leert elke boom de residuen van de bomen ervoor. Zie [Boom na boom op wat overblijft: gradient boosting](/course/open/ml-gradient-boosting?lang=nl).",
        },
    )
    quick_residual
    return (quick_residual,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Glass-box models](/course/open/rai-glass-box-models?lang=nl) gaf de logistische regressie leeftijd één weight: elk levensjaar vermenigvuldigt de odds op meer dan 50.000 dollar met 1,047, van 25 naar 26 net zo goed als van 70 naar 71. Eén weight per feature is één rechte lijn op de log-odds. Als wat je voorspelt eerst stijgt en dan weer daalt, kan een rechte lijn dat niet laten zien, en aan de weight zie je het niet. Een generalized additive model geeft elke feature een eigen curve, die mag buigen, en telt de curves op. Elke curve kun je tekenen, dus het model blijft een glass box.

    **Een voorbeeld uit het dagelijks leven.** De verzendkosten van een pakket: één tabel voor het gewicht, één voor de afmetingen, één voor de bestemming. Je zoekt elk deel op en telt ze op. De tabel voor het gewicht mag buigen, zodat de eerste kilo meer kost dan elke kilo daarna, en toch zie je wat elk deel kost, omdat de delen alleen worden opgeteld en nooit gemengd.
    """)
    return


@app.cell(hide_code=True)
def predict_seventy():
    guess_seventy = L.guess(
        "seventy",
        question="Van de 50-jarigen in UCI Adult verdient 43,1 procent meer dan 50.000 dollar. Hoe zit dat bij de 70-jarigen?",
        options={
            "more": "Meer, ongeveer 55 procent",
            "same": "Ongeveer evenveel",
            "fewer": "Minder, ongeveer 19 procent",
        },
        reveal="**Minder: 18,8 procent.** Van de 25-jarigen is het 6,4 procent. Het aandeel stijgt met de leeftijd en daalt daarna weer. De logistische regressie uit [Glass-box models](/course/open/rai-glass-box-models?lang=nl) kan dat niet laten zien: haar weight voor leeftijd, 0,046, zegt dat elk jaar de odds met 1,047 vermenigvuldigt, ook na je vijftigste. In deze aandelen zitten ook opleiding en uren; de curves van deze les houden die gelijk.",
    )
    guess_seventy
    return (guess_seventy,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Eén curve per feature.** Trevor Hastie en Robert Tibshirani (1986) introduceerden generalized additive models: ze vervangen de lineaire vorm ∑ βjXj door een som van gladde functies ∑ sj(Xj). Een lineair model telt per feature een weight keer die feature op. Een additief model telt per feature een curve op, afgelezen bij de waarde van die feature, een curve in elke vorm die de data vragen. Bij een ja of nee is de som een score op de log-odds, net als bij een logistische regressie, en de sigmoid maakt er een kans van. Yin Lou, Rich Caruana en Johannes Gehrke (2012) noemen de curves shape functions. Omdat die zo ingewikkeld mogen zijn als nodig, voorspellen GAM's volgens hen beter dan eenvoudige lineaire modellen; en omdat ze geen interacties tussen features bevatten, kunnen gebruikers ze makkelijk lezen.

        **De curves leren met boosting.** Lou en anderen leren de curves op de manier die je kent uit [Boom na boom op wat overblijft: gradient boosting](/course/open/ml-gradient-boosting?lang=nl), met één verschil: elke kleine boom ziet maar één feature. In elke ronde van boosting gaan ze alle features om de beurt langs. Wat een boom optelt, hoort dan bij de curve van zijn feature. In hun proeven werkte gradient boosting met bomen van 2 tot 4 bladeren, elk gemaakt met bagging, op de meeste datasets het best; een boom met diepte 2 heeft hoogstens vier bladeren.

        **Explainable Boosting Machines.** De library InterpretML (Harsha Nori en anderen, 2019) noemt haar versie een Explainable Boosting Machine, een snelle uitvoering van het GA2M-algoritme (GA2M neemt ook enkele paren van features mee; zie *Let op* hieronder). Die leert op dezelfde manier: de boosting traint steeds op één feature tegelijk, om de beurt, met een heel lage learning rate, zodat de volgorde van de features niet uitmaakt. En ze voorspelt door op te tellen: elke functie werkt als een opzoektabel per feature, de bijdragen worden gewoon opgeteld en gaan dan door een linkfunctie, bij een ja of nee de sigmoid. Volgens de documentatie voorspellen EBM's vaak even goed als de beste black boxes, terwijl ze helemaal te lezen blijven. Let op het woord vaak. Op UCI Adult maten Lou en anderen een foutpercentage van 16,04 voor een logistische regressie, 14,85 voor een random forest en ongeveer 14,6 voor hun gebooste additieve modellen.

        **De curves lezen.** Molnar: wat een gladde curve betekent, zie je pas als je naar de curve kijkt. Je leest haar aan de verschillen, zoals je een weight leest: in het uitgewerkte voorbeeld geeft leeftijd 50 een score die 2,17 hoger is dan bij leeftijd 25, dus worden de odds $e^{2{,}17}$, ongeveer 8,8 keer zo groot, met opleiding en uren gelijk. Net als een weight zegt een curve wat het model uit de data leerde, niet wat oorzaak en gevolg is.

        **De verrassing met astma.** In de jaren negentig leerde een systeem met regels uit data over patiënten met een longontsteking dat astma hun risico verlaagt. Rich Caruana en anderen (2015) vertellen waarom: patiënten met astma die met een longontsteking binnenkwamen, werden meestal niet alleen in het ziekenhuis opgenomen, maar meteen op de intensive care. Die zorg was zo goed dat hun risico om aan de longontsteking te overlijden lager werd dan dat van de bevolking als geheel. Modellen die op die data leren, leren dus ten onrechte dat astma het risico verlaagt, terwijl mensen met astma juist een veel hoger risico hebben als ze niet worden opgenomen. Op een van de datasets van toen voorspelde een neural network beter dan een logistische regressie, maar het werd niet gebruikt: als het systeem met regels had geleerd dat astma het risico verlaagt, dan had het neural network dat zeker ook geleerd, en omdat het niet te lezen was, was het moeilijk te weten welke andere problemen er misschien ook gerepareerd moesten worden. In 2015 vond hun additieve model met paren, op data van 14.199 patiënten met een longontsteking, hetzelfde patroon terug. Omdat je elke term kunt tekenen, konden ze het zien en herstellen: door de term weg te halen, of door met kennis van experts de grafiek opnieuw te tekenen, zodat de risicoscore bij astma positief is in plaats van negatief.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **generalized additive model** (GAM) | een model dat per feature één curve optelt, elk afgelezen bij de waarde van de feature; bij een ja of nee is de som een score op de log-odds |
        | **Explainable Boosting Machine** (EBM) | het additieve model van InterpretML, waarvan de curves worden geleerd door kleine bomen te boosten, één feature tegelijk; standaard neemt het ook enkele paren van features mee |
        | $f_j(x_j)$ | wat de curve van feature $j$ bij de waarde $x_j$ optelt bij de score |
        """),
        L.watch_out("Een EBM is niet alleen één curve per feature: standaard neemt ze ook paren van features mee. Lou en anderen (2013) noemen zo'n model GA2M, een generalized additive model plus interacties, met een klein aantal termen voor paren; een paar is een tabel over twee features, die je nog steeds kunt tekenen. En als features samengaan, zoals iemands leeftijd en het aantal jaren werkervaring, zijn de curves moeilijker te lezen: Hastie en Tibshirani waarschuwen dat dit voor de curves van een additief model misschien nog erger is dan voor de weights van een lineair model."),
    ])
    return


@app.cell(hide_code=True)
def explore_rounds_cell():
    explore_rounds = L.explore("rounds", "rondes", lambda n: C.rounds_picture(n, "nl"), start=1, values=C.STOPS,
                               role="hyper",
                               task="Schuif het aantal rondes van 1 naar 100. Na hoeveel rondes zit het model vaker goed dan altijd nee zeggen, en waar komt de curve van leeftijd het hoogst uit?")
    mo.vstack([
        mo.md("**De curves, ronde na ronde**"),
        explore_rounds,
        mo.md(r"""
        Het model uit het uitgewerkte voorbeeld, op de 48.842 mensen van UCI Adult: wat elke curve optelt bij de score, op één schaal. Elke ronde voegt drie kleine bomen toe, één per kolom. Eerst zijn de curves bijna vlak en zegt het model nog tegen iedereen nee; daarna krijgen ze hun vorm.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo leer je een additief model met boosting, bij een ja of nee:

    1. Laat de score van iedereen beginnen bij de log-odds van het aandeel ja: `np.log(share / (1 - share))`.
    2. Neem de features één voor één. Voor elke feature: de kans `p = 1 / (1 + np.exp(-score))`, een boom met diepte 2 die `y - p` leert van die ene kolom, en `rate` keer wat hij voorspelt opgeteld bij de score.
    3. Herhaal stap 2 veel rondes, en bewaar elke boom met zijn feature.
    4. De curve van een feature is `rate` keer de som van wat zijn bomen voorspellen, bij elke waarde van de feature.
    5. Voorspellen voor een persoon: de start, plus elke curve bij de waarde van die persoon, dan de sigmoid.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De drie kolommen uit [Redenen bij een besluit](/course/open/rai-reason-codes?lang=nl), nu elk met een curve. `np.log` is de natuurlijke logaritme, die `np.exp` ongedaan maakt. `people[[name]]`, met twee paar haakjes, houdt één kolom over als tabel, de vorm waar een boom van leert; `.reshape(-1, 1)` maakt van een rij getallen zo'n kolom. Bij een ja of nee leren de bomen `y - p`, het label min de kans: dat zegt welke kant op en hoe ver de kans ernaast zit, en de score schuift die kant op. Die verschillen in kans zijn klein naast de score, dus elke stap telt de helft op van wat een boom voorspelt in plaats van een tiende. InterpretML gaat zorgvuldiger te werk: met bagging, en in de instellingen die de paper over InterpretML aanraadt met een learning rate van 0,01 en 5000 epochs, rondes langs alle features. Voer de cel uit: hij leert de curves, print van elke curve een paar punten, en de kans van de aanvrager van 30 uit [Redenen bij een besluit](/course/open/rai-reason-codes?lang=nl). Dat duurt een seconde of twee.
    """)
    return


@app.cell
def example_gam():
    import numpy as np
    import pandas as pd
    from sklearn.tree import DecisionTreeRegressor

    # 48.842 mensen uit een Amerikaans onderzoek van 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _names = ["age", "education_num", "hours_per_week"]
    _y = (_people["income"] == ">50K").to_numpy()

    # 1. elke score begint bij de log-odds van het aandeel dat meer verdient
    _start = np.log(_y.mean() / (1 - _y.mean()))
    _score = np.zeros(len(_y)) + _start
    _trees = []
    # 3. stap 2, 100 rondes
    for _r in range(100):
        # 2. één kolom tegelijk: een kleine boom leert wat nog fout is, de helft ervan komt erbij
        for _name in _names:
            _column = _people[[_name]].to_numpy()
            _p = 1 / (1 + np.exp(-_score))
            _tree = DecisionTreeRegressor(max_depth=2, random_state=0).fit(_column, _y - _p)
            _score = _score + 0.5 * _tree.predict(_column)
            _trees.append((_name, _tree))
    print("start", round(_start, 3))
    print("goed", round(np.mean((_score >= 0) == _y), 3))

    # 4. de curve van elke kolom, bij elke waarde van 0 tot de grootste
    _curves = []
    for _name in _names:
        _values = np.arange(_people[_name].max() + 1).reshape(-1, 1)
        _curve = np.zeros(len(_values))
        for _tree_name, _tree in _trees:
            if _tree_name == _name:
                _curve = _curve + 0.5 * _tree.predict(_values)
        _curves.append(_curve)
    print("leeftijd 25, 50, 70:", _curves[0][25].round(2), _curves[0][50].round(2), _curves[0][70].round(2))
    print("opleiding 9, 13, 16:", _curves[1][9].round(2), _curves[1][13].round(2), _curves[1][16].round(2))
    print("uren 20, 40, 60:", _curves[2][20].round(2), _curves[2][40].round(2), _curves[2][60].round(2))

    # 5. de aanvrager van 30 met opleidingsniveau 9, die 40 uur per week werkt
    _s = _start + _curves[0][30] + _curves[1][9] + _curves[2][40]
    print("kans", round(1 / (1 + np.exp(-_s)), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Elke score begint bij −1,157, de log-odds van de 23,9 procent die meer verdient. Na 100 rondes van drie bomen zit het model bij 80,0 procent van de mensen goed; de logistische regressie uit [Glass-box models](/course/open/rai-glass-box-models?lang=nl) bij 78,9 procent, de boom met diepte 2 bij 78,7, altijd nee zeggen bij 76,1. De curve van leeftijd ligt het laagst rond 20, stijgt tot haar hoogste punt bij 50, 0,76, en daalt daarna: 0,51 bij 70. De curve van opleiding klimt in treden, met grote bij 13, een bachelor, en bij 15. De curve van de uren stijgt tot 50 uur per week en daalt daarna een beetje. Voor de aanvrager van 30 met opleidingsniveau 9 die 40 uur werkt, tellen de curves −0,28, −0,56 en −0,15 op bij de start, en de kans op een ja is 0,104.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_chance():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Tel de curves op

        Schrijf `chance(person, start, curves)`: de kans op een ja voor één persoon. `person` bevat één waarde per feature, `start` is de score waarmee iedereen begint, en `curves` bevat één curve per feature, dus `curves[i][person[i]]` is wat feature `i` voor deze persoon optelt. De check roept eerst `chance((2, 1), -1.0, [[0.0, 0.5, 1.5], [-0.5, 0.5]])` aan: twee features met kleine verzonnen curves. Daarna probeert hij andere personen, curves en aantallen features; de tekening voert je functie uit met de curves van het uitgewerkte voorbeeld.
        """),
        L.task("Schrijf `chance(person, start, curves)`: de start plus elke curve bij de waarde van de persoon, door de sigmoid."),
    ])
    return


@app.cell(hide_code=True)
def exercise_chance():
    box_chance = L.code_box("chance", lines=10)
    box_chance
    return (box_chance,)


@app.cell(hide_code=True)
def check_chance(box_chance):
    result_chance = L.check("chance", box_chance.value)
    result_chance
    return (result_chance,)


@app.cell(hide_code=True)
def hints_chance():
    L.hints(
        "chance",
        [
            "Begin bij `start`. Tel voor elke feature op wat haar curve geeft bij de waarde van de persoon. Maak van de som dan een kans, zoals een logistische regressie doet.",
            "Loop over de features met `for i in range(len(person)):`. Feature `i` heeft de waarde `person[i]`, en haar curve is `curves[i]`: welk getal daaruit heb je nodig? Na de loop komt de sigmoid, `1 / (1 + np.exp(-score))`.",
            "`score = score + curves[i][___]`",
        ],
        why="`score = start`, dan `for i in range(len(person)):` met daarin `score = score + curves[i][person[i]]`, en na de loop `return 1 / (1 + np.exp(-score))`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_round():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Eén ronde, één kolom tegelijk

        In het vak ontbreken alleen drie regels in de loop over de drie kolommen. Vul ze in: de kans `p` uit de score zoals die nu is, een boom met diepte 2 en `random_state=0` die `y - p` leert van deze ene kolom, en de nieuwe score. De check vergelijkt `score` na één ronde en probeert ook een andere learning rate; de tekening laat zien wat elk van je bomen optelt.
        """),
        L.task("Vervang de drie `...`: `p`, een boom op `column` die `y - p` leert, en de nieuwe `score`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_round():
    box_round = L.code_box("round", lines=18)
    box_round
    return (box_round,)


@app.cell(hide_code=True)
def check_round(box_round):
    result_round = L.check("round", box_round.value)
    result_round
    return (result_round,)


@app.cell(hide_code=True)
def hints_round():
    L.hints(
        "round",
        [
            "Drie stappen per kolom, zoals in één ronde gradient boosting: wat nog fout is, een kleine boom die dat leert, en een deel van wat de boom voorspelt opgeteld bij de score.",
            "De kans komt uit de score via de sigmoid. De boom is `DecisionTreeRegressor(max_depth=2, random_state=0)`, getraind met `.fit(column, y - p)`.",
            "`score = score + rate * tree.predict(___)`",
        ],
        why="`p = 1 / (1 + np.exp(-score))`, `tree = DecisionTreeRegressor(max_depth=2, random_state=0).fit(column, y - p)` en `score = score + rate * tree.predict(column)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_read():
    mo.md(r"""
    ### Oefening 3. Een curve lezen
    """)
    return


@app.cell(hide_code=True)
def choice_read_cell():
    choice_read = L.choice(
        "read",
        question="In het uitgewerkte voorbeeld stijgt de curve van leeftijd van 25 tot haar hoogste punt bij 50 en daalt daarna. Wat zegt dat?",
        options={
            "cause": "Wie 50 wordt, gaat meer verdienen",
            "right": "Met de rest gelijk scoort het model leeftijd 50 het hoogst",
            "most": "De meeste 50-jarigen in de tabel verdienen meer dan 50.000 dollar",
        },
        correct="right",
        why={
            "cause": "Een curve is wat het model uit de tabel leerde, geen oorzaak. Mensen van 50 verschillen van mensen van 25 op veel manieren die niet in de drie kolommen staan. Zie *De curves lezen* in *Het idee*.",
            "right": "Met dezelfde opleiding en uren telt de curve van leeftijd bij 50 het meest op bij de score: 2,17 meer dan bij 25. Ze zegt wat het model uit de tabel leerde, niet dat leeftijd het inkomen veroorzaakt; de volgende oefening laat zien hoe een curve kan misleiden.",
            "most": "Van de 50-jarigen in de tabel verdient 43,1 procent meer: niet de meesten. De curve geeft wat leeftijd optelt bij de score, met opleiding en uren gelijk; ze is geen aandeel van mensen. Zie *De curves lezen* in *Het idee*.",
        },
    )
    choice_read
    return (choice_read,)


@app.cell(hide_code=True)
def text_exercise_treated():
    mo.md(r"""
    ### Oefening 4. Nog een verrassing
    """)
    return


@app.cell(hide_code=True)
def choice_treated_cell():
    choice_treated = L.choice(
        "treated",
        question="Caruana en anderen bedenken een tweede geval: misschien krijgen zwangere vrouwen met een longontsteking ook een stevige behandeling die hun risico lager maakt dan dat van de bevolking als geheel. Wat zou een model dan uit zulke data leren?",
        options={
            "right": "Dat zwangerschap het risico op overlijden verlaagt",
            "raises": "Dat zwangerschap het risico op overlijden verhoogt",
            "nothing": "Niets: een model leert alleen wat artsen het vertellen",
        },
        correct="right",
        why={
            "right": "Goed. De data laten zien wie overleed, na de behandeling die ze kregen. Als zwangere patiënten steviger werden behandeld en minder vaak overleden, leert het model dat zwangerschap het risico verlaagt, net als bij astma. In een additief model kun je die term zien, een arts om raad vragen, en de term weghalen of opnieuw tekenen.",
            "raises": "Zonder die behandeling is hun risico misschien hoger, maar het model ziet alleen wie overleed na de zorg die ze kregen. Met een stevige behandeling overleden er minder, dus leert het model dat zwangerschap het risico verlaagt, net als bij astma. Zie *De verrassing met astma* in *Het idee*.",
            "nothing": "Een model leert uit de data, niet uit wat artsen zeggen: wat in de tabel samengaat met minder sterfgevallen, ziet eruit als een lager risico. Met een stevige behandeling leert het model dat zwangerschap het risico verlaagt, net als bij astma. Zie *De verrassing met astma* in *Het idee*.",
        },
    )
    choice_treated
    return (choice_treated,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Glass-box models](/course/open/rai-glass-box-models?lang=nl) las je een weight aan wat hij bij de logit optelt; hier heeft elke feature een hele curve, en die lees je op dezelfde manier, aan de verschillen. De curves werden geleerd zoals in [Boom na boom op wat overblijft: gradient boosting](/course/open/ml-gradient-boosting?lang=nl), één kolom tegelijk. De volgende les, [Veel goede modellen](/course/open/rai-rashomon?lang=nl), vraagt waarom een model dat je kunt lezen vaak ongeveer even goed voorspelt als een model dat je niet kunt lezen. Later in deze module tekent [Hoe een voorspelling meebeweegt met een feature](/course/open/rai-pdp-ice?lang=nl) een curve zoals deze voor elk model, ook een black box, achteraf gemaakt in plaats van uit het model gelezen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_read, choice_treated, result_chance, result_round):
    L.wrap_up(result_chance, result_round, choice_read, choice_treated)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
