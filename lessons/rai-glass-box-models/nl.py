# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-glass-box-models. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-glass-box-models", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Soorten uitleg](/course/open/rai-kinds-of-explanation?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_black():
    quick_black = L.choice(
        "quick-black",
        question="Je kunt een model inputs geven en zijn outputs zien, maar niets van hoe het van het een naar het ander komt. Hoe heet zo'n model?",
        options={
            "right": "Een black box",
            "agnostic": "Een model-agnostic model",
            "global": "Een global explanation",
        },
        correct="right",
        why={
            "right": "Een black box: je ziet wat erin gaat en wat eruit komt. Alleen een model-agnostic methode, die niets dan inputs en outputs nodig heeft, kan hem nog uitleggen.",
            "agnostic": "Model-agnostic zeg je van een methode die uitlegt, niet van een model: een methode die alleen inputs en outputs nodig heeft. Het model dat ze hier uitlegt, is een black box. Zie [Soorten uitleg](/course/open/rai-kinds-of-explanation?lang=nl).",
            "global": "Een global explanation beschrijft hoe een model zich in het algemeen gedraagt; het is geen soort model. Een model dat je alleen van buiten kunt bekijken, is een black box. Zie [Soorten uitleg](/course/open/rai-kinds-of-explanation?lang=nl).",
        },
    )
    quick_black
    return (quick_black,)


@app.cell(hide_code=True)
def quick_check_design():
    quick_design = L.choice(
        "quick-design",
        question="Welke van deze uitleg lees je uit het model zelf, zonder dat er achteraf iets gemaakt wordt?",
        options={
            "nearby": "Een klein lineair model rond één voorspelling",
            "right": "De vragen van een beslisboom",
            "shuffle": "Eén kolom schudden",
        },
        correct="right",
        why={
            "nearby": "Een klein model dat rond één voorspelling wordt aangepast, maak je achteraf, door het getrainde model vragen te stellen: een post-hoc explanation. Zie [Soorten uitleg](/course/open/rai-kinds-of-explanation?lang=nl).",
            "right": "De vragen zijn de boom zelf: je leest ze, er wordt niets achteraf gemaakt. Dat is interpretability by design, het onderwerp van deze les.",
            "shuffle": "Een kolom schudden en kijken hoeveel slechter het model wordt, doe je achteraf, door het getrainde model vragen te stellen. Zie [Soorten uitleg](/course/open/rai-kinds-of-explanation?lang=nl).",
        },
    )
    quick_design
    return (quick_design,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    COMPAS is een commerciële score die voorspelt of iemand opnieuw wordt opgepakt; je kent hem misschien uit [COMPAS en een sollicitatietool](/course/open/rai-case-compas-amazon?lang=nl). Hoe hij werkt, is niet openbaar: Cynthia Rudin (2019) noemt zo'n model een black box omdat het eigendom van een bedrijf is. Rudin zette er een model naast dat iedereen kan lezen: drie regels, op leeftijd en het aantal eerdere misdrijven, waarvan één ook op geslacht. De vraag: als drie regels het werk kunnen doen, waarom zou je dan een geheim model vertrouwen? Een model waarvan een mens elke stap kan volgen, heet een glass box: er valt achteraf niets uit te leggen, want je kunt het lezen.

    **Een voorbeeld uit het dagelijks leven.** Een belastingaangifte leidt je regel voor regel naar wat je moet betalen: je kunt elke stap nagaan, en als een regel fout is, zie je welke. Een ervaren boekhouder komt misschien op hetzelfde bedrag uit ervaring, maar hoe, kun je niet volgen.
    """)
    return


@app.cell(hide_code=True)
def predict_three():
    guess_three = L.guess(
        "three",
        question="Rudin vergeleek COMPAS met de drie regels op data uit Broward County in Florida. Hoe goed voorspelden de drie regels wie opnieuw zou worden opgepakt?",
        options={
            "much": "Veel slechter dan COMPAS",
            "little": "Iets slechter",
            "same": "Ongeveer even goed",
        },
        reveal="**Ongeveer even goed.** Volgens Rudin voorspelt COMPAS recidive even goed als het heel eenvoudige model van drie regels, met alleen leeftijd en het aantal eerdere misdrijven. Op de data uit Broward County in Florida hebben beide modellen een vergelijkbare true en false positive rate en true en false negative rate. Waarom een eenvoudig model het even goed kan doen als een ingewikkeld model, is het onderwerp van [Veel goede modellen](/course/open/rai-rashomon?lang=nl), verderop in deze module.",
    )
    guess_three
    return (guess_three,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een glass box.** InterpretML, een library van Harsha Nori en anderen (2019), gebruikt dat woord. Die deelt methoden in: glassbox-modellen, modellen voor machine learning die ontworpen zijn om leesbaar te zijn, zoals lineaire modellen, rule lists en generalized additive models, en blackbox-technieken die bestaande systemen uitleggen. Volgens de documentatie is de uitleg van een glassbox-model exact en voor mensen te lezen, terwijl de uitleg van een blackbox-model meestal een benadering is. Andere bronnen zeggen interpretable model, transparant model of white box; het boek van Molnar zegt het liefst interpretable model. Deze course zegt glass-box model.

        **Met de hand volgen.** Zachary Lipton (2016) geeft twee toetsen. De eerste: een mens moet de input en de parameters van het model kunnen nemen en in redelijke tijd elke berekening kunnen doorlopen die tot een voorspelling leidt. De tweede: elk deel van het model, elke input, parameter en berekening, is intuïtief uit te leggen, en daarvoor moet elke input op zichzelf te begrijpen zijn. Een weight voor een kolom die `f3` heet, haalt de eerste toets en zakt voor de tweede.

        **Een lineair model lezen.** Molnar: als een feature één eenheid stijgt en alle andere features gelijk blijven, stijgt de voorspelling met de weight van die feature. Een logistische regressie telt in plaats daarvan op tot een logit, en de logit is de log van de odds, zoals in [Van score naar kans: de sigmoid](/course/open/ml-sigmoid?lang=nl). Eén eenheid meer vermenigvuldigt de odds dus met $e^{w}$. De weights tellen op; hun effect op de odds vermenigvuldigt, en op de kans is het geen van beide. Molnar: de uitleg van de weights is vermenigvuldigend en niet optellend.

        **Een boom lezen.** Elk pad van boven naar een blad is een regel: als feature A kleiner is dan grens c EN ..., dan is de voorspelling het gemiddelde van de voorbeelden in dat blad. Bij een boom die ja of nee zegt, is dat gemiddelde het deel ja in het blad, en het blad zegt ja als dat deel boven de helft ligt. Een boom met diepte $d$ heeft hoogstens $2^d$ bladeren, dus een beslisboom is heel goed te lezen, zolang hij kort is, schrijft Molnar. Ook waarschuwt Molnar dat bomen nogal instabiel zijn: een iets andere tabel kan een andere boom laten groeien.

        **Hoe klein is klein genoeg?** Rudin: mensen kunnen hoogstens 7±2 dingen tegelijk vasthouden. Rudin verwijst naar *The magical number seven, plus or minus two* van George Miller (1956) en *The magical mystery four* van Nelson Cowan (2010); zoals de titel van Cowan al zegt, komt nieuwer werk lager uit, rond de vier. Lipton voegt toe dat geen soort model vanzelf een glass box is: lineaire modellen, systemen met regels en beslisbomen zijn het geen van alle uit zichzelf. Diepe bomen, lange lijsten met regels en modellen met duizenden features kunnen volgens Lipton minder doorzichtig zijn dan een betrekkelijk compact neural network.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **glass-box model** | een model dat door zijn ontwerp te lezen is, zoals een korte beslisboom of een lineair model met weinig features, zodat een mens elke stap van input naar output kan volgen |
        | $e^{w}$ | de factor waarmee één eenheid meer van een feature de odds vermenigvuldigt, in een logistische regressie met weight $w$ |
        """),
        L.watch_out("Een weight is niet het effect van zijn feature op zich. Hij geldt als alle andere features gelijk blijven, en hij verandert met de andere features die in het model zitten. Het voorbeeld van Lipton: het verband tussen griepvaccinatie en griep kan positief of negatief uitvallen, afhankelijk van of er ook features in zitten voor hoge leeftijd, jonge kinderen of een zwakke afweer. En een glass box kan net zo goed iets verkeerds leren; je ziet het alleen makkelijker, een verhaal dat de volgende les vertelt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo ga je na of een mens een model kan volgen:

    1. Tel de delen: weights, vragen, bladeren. Een handvol is te volgen, duizenden niet.
    2. Ga na of elke input op zichzelf iets betekent, met een naam en een eenheid.
    3. Loop één voorspelling door. Een lineair model: vermenigvuldigen en optellen. Een logistische regressie: lees ook $e^{w}$ als de factor op de odds. Een boom: volg de vragen naar een blad.
    4. Lees elke weight met zijn eenheid, en met de andere features gelijk.
    """)
    return


@app.cell(hide_code=True)
def picture_tree():
    mo.vstack([
        mo.md("**Een glass box, helemaal getekend**"),
        mo.Html(C.whole_tree("nl")),
        mo.md(r"""
        De boom uit het uitgewerkte voorbeeld, op de 48.842 mensen van UCI Adult. Twee vragen bepalen elke voorspelling. Maar één blad zegt ja: een opleiding boven 12,5 en ouder dan 29,5, waar 56,5 procent meer dan 50.000 dollar verdient. Links vraagt de boom naar de leeftijd hoewel beide bladeren nee zeggen: de vraag maakt de bladeren zuiverder, 4 procent tegenover 22,6 procent, zonder te veranderen wat ze zeggen.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Twee glass boxes op de drie kolommen van [Redenen bij een besluit](/course/open/rai-reason-codes?lang=nl): de logistische regressie uit die les, en een beslisboom met diepte 2 zoals in [Vraag na vraag: de beslisboom](/course/open/ml-decision-tree?lang=nl). `.to_numpy()` maakt van de drie kolommen een NumPy-array, met een rij per persoon. Voer de cel uit: hij print de weights en hun factoren op de odds, de boom zoals `export_text` hem schrijft, en hoe vaak elk model goed zit. Dat duurt een seconde of twee.
    """)
    return


@app.cell
def example_glass():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, export_text

    # 48.842 mensen uit een Amerikaans onderzoek van 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _names = ["age", "education_num", "hours_per_week"]
    _X = _people[_names].to_numpy()
    _y = _people["income"] == ">50K"

    # glass box 1: een logistische regressie, één weight per kolom
    _model = LogisticRegression(max_iter=1000)
    _model.fit(_X, _y)
    _weights = _model.coef_[0].round(3)
    print("weights", _weights)
    print("factoren op de odds", np.exp(_weights).round(3))
    print("logistische regressie goed", round(np.mean(_model.predict(_X) == _y), 3))

    # glass box 2: een beslisboom met hoogstens twee vragen op elk pad
    _tree = DecisionTreeClassifier(max_depth=2, random_state=0)
    _tree.fit(_X, _y)
    print(export_text(_tree, feature_names=_names))
    print("boom goed", round(np.mean(_tree.predict(_X) == _y), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Eén opleidingsniveau meer vermenigvuldigt de odds op meer dan 50.000 dollar met 1,412, één levensjaar meer met 1,047, één uur per week meer met 1,043, telkens met de andere twee gelijk. De logistische regressie zit bij 78,9 procent van de mensen goed, de boom bij 78,7 procent; altijd nee zeggen zou bij 76,1 procent goed zitten. Geen van beide is een sterk model, maar je kunt ze allebei helemaal lezen: drie weights en een bias, of drie vragen en vier bladeren. Dat maakt ze tot glass boxes, niet hoe goed ze voorspellen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_follow():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Volg de boom met de hand

        Het vak begint met de boom uit het uitgewerkte voorbeeld, als commentaar. Schrijf de functie `says_yes(age, education_num, hours_per_week)`: hij volgt de boom met `if` en `else`, zoals je met de hand zou doen, en geeft `True` of `False` terug, zoals het blad zegt. Elke `|` verder naar rechts is één vraag dieper; een regel met `<=` leidt naar het deel waar de vraag klopt. De check roept je functie aan voor de aanvrager uit [Redenen bij een besluit](/course/open/rai-reason-codes?lang=nl), 30 jaar, opleidingsniveau 9 en 40 uur, en voor twaalf andere mensen; de tekening probeert hem op iedereen in de tabel.
        """),
        L.task("Schrijf `says_yes(age, education_num, hours_per_week)`: volg de vragen van de boom en geef terug wat zijn blad zegt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_follow():
    box_follow = L.code_box("follow", lines=23)
    box_follow
    return (box_follow,)


@app.cell(hide_code=True)
def check_follow(box_follow):
    result_follow = L.check("follow", box_follow.value)
    result_follow
    return (result_follow,)


@app.cell(hide_code=True)
def hints_follow():
    L.hints(
        "follow",
        [
            "Begin bij de bovenste regel van de boom: één vraag met twee delen. In elk deel komt nog één vraag, en dan een blad.",
            "Schrijf `if education_num <= 12.5:`, en daarbinnen een tweede `if` over `age` met `return False` in beide delen. Onder `else:` komt de andere vraag over `age`, met een eigen grens.",
            "`if age <= 29.5:` met `return False`, en onder `else:` `return ___`",
        ],
        why="`if education_num <= 12.5:` met daarbinnen `if age <= 30.5: return False` en `else: return False`; dan `else:` met `if age <= 29.5: return False` en `else: return True`, elke `return` op een eigen regel. Korter: `return education_num > 12.5 and age > 29.5`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_odds():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Vier opleidingsniveaus

        `weights` bevat de weights van de logistische regressie: leeftijd, opleiding, uren. Twee mensen zijn even oud en werken evenveel uren; de een heeft een middelbareschooldiploma (opleidingsniveau 9), de ander een bachelor (niveau 13). Reken `factor` uit: hoeveel keer zo groot de odds op een ja zijn voor de tweede. De check probeert ook andere weights.
        """),
        L.task("Reken `factor` uit: hoeveel keer de odds op een ja groeien van opleidingsniveau 9 naar 13."),
    ])
    return


@app.cell(hide_code=True)
def exercise_odds():
    box_odds = L.code_box("odds", lines=7)
    box_odds
    return (box_odds,)


@app.cell(hide_code=True)
def check_odds(box_odds):
    result_odds = L.check("odds", box_odds.value)
    result_odds
    return (result_odds,)


@app.cell(hide_code=True)
def hints_odds():
    L.hints(
        "odds",
        [
            "Eén niveau meer vermenigvuldigt de odds met $e$ tot de macht van de weight. Vier niveaus meer vermenigvuldigen vier keer.",
            "De weight van opleiding is `weights[1]`. Vier niveaus tellen vier keer die weight op bij de logit; de odds veranderen met `np.exp` van wat erbij komt.",
            "`factor = np.exp(weights[1] * ___)`",
        ],
        why="`factor = np.exp(weights[1] * 4)`, of `np.exp(weights[1]) ** 4`: ongeveer `3.975`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_glass():
    mo.md(r"""
    ### Oefening 3. Welke kan een mens volgen?

    Zet elk model in zijn groep: kan een mens een voorspelling stap voor stap volgen, zoals het model nu is?
    """)
    return


@app.cell(hide_code=True)
def sort_glass_cell():
    sort_glass = L.sort("glass")
    sort_glass
    return (sort_glass,)


@app.cell(hide_code=True)
def check_glass(sort_glass):
    result_glass = L.check("glass", sort_glass)
    result_glass
    return (result_glass,)


@app.cell(hide_code=True)
def hints_glass():
    L.hints(
        "glass",
        [
            "Twee vragen voor elk: hoeveel delen heeft het, en betekent elk deel op zich iets?",
            "Drie modellen zijn klein en hebben delen met een betekenis. Een boom met diepte 20 kan 2 tot de macht 20 bladeren hebben, meer dan een miljoen.",
            "Een mens kan volgen: de korte boom, de punten en \\_\\_\\_.",
        ],
        why="**Een mens kan het volgen:** de korte boom, de drie weights met een naam, de punten. **Niet zoals het nu is:** de diepe boom, de 5000 weights, de vier gemaakte kolommen, het kleine netwerk.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_scale():
    mo.md(r"""
    ### Oefening 4. Weights vergelijken
    """)
    return


@app.cell(hide_code=True)
def choice_scale_cell():
    choice_scale = L.choice(
        "scale",
        question="In de logistische regressie heeft opleiding de weight 0,345 en leeftijd 0,046. Telt opleiding 7,5 keer zo zwaar als leeftijd?",
        options={
            "times": "Ja, zijn weight is 7,5 keer zo groot als die van leeftijd",
            "right": "Nee, een niveau en een jaar zijn andere eenheden",
            "less": "Nee, leeftijd telt zwaarder dan opleiding",
        },
        correct="right",
        why={
            "times": "Elke weight geldt per eenheid van zijn eigen feature: één opleidingsniveau, één levensjaar. Molnar raadt af om weights van features met verschillende schalen te vergelijken. In maanden gemeten zou leeftijd een weight krijgen die 12 keer zo klein is, zonder dat het model verandert. Zie *Een lineair model lezen* in *Het idee*.",
            "right": "Een weight geldt per eenheid: per opleidingsniveau, per levensjaar. In maanden gemeten zou leeftijd een weight krijgen die 12 keer zo klein is, en aan het model zou niets veranderd zijn. Molnar raadt af om weights van features met verschillende schalen te vergelijken.",
            "less": "Ook dat zeggen de weights niet. Elke weight geldt per eenheid van zijn eigen feature, een opleidingsniveau of een levensjaar, en die eenheden kun je niet vergelijken. Zie *Een lineair model lezen* in *Het idee*.",
        },
    )
    choice_scale
    return (choice_scale,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Soorten uitleg](/course/open/rai-kinds-of-explanation?lang=nl) las je een uitleg uit het model zelf; bij een glass box is die uitleg het hele model. In [Redenen bij een besluit](/course/open/rai-reason-codes?lang=nl) gebruikte je de weights om redenen te geven. De volgende les, [Additieve modellen](/course/open/rai-additive-models?lang=nl), houdt een glass box maar laat elke feature buigen: een curve per feature in plaats van een rechte lijn. Die les vertelt ook het verhaal van een glass box die iets gevaarlijks leerde, en hoe de makers dat konden zien. [Veel goede modellen](/course/open/rai-rashomon?lang=nl) vraagt waarom een eenvoudig model vaak even goed voorspelt als een ingewikkeld model.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_scale, result_follow, result_glass, result_odds):
    L.wrap_up(result_follow, result_odds, result_glass, choice_scale)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
