# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-project-audit-explain. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("rai-project-audit-explain", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [Project: meet de bias van je eigen classifier](/course/open/rai-project-audit-bias?lang=nl) en een over [Wat zou er moeten veranderen?](/course/open/rai-counterfactuals?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_why():
    quick_why = L.choice(
        "quick-why",
        question="In stap één gaf je `report` de vrouwen in de test set een deel met een ja van 0,070 en de mannen 0,248. Wat vertelt die tabel je over het waarom?",
        options={
            "right": "Nog niets: hij telt uitkomsten per groep",
            "sex": "Dat de classifier naar geslacht kijkt",
            "size": "Dat er minder vrouwen in de test set zitten",
        },
        correct="right",
        why={
            "right": "Een audit meet wie een ja krijgt, per groep. Waarom de uitkomsten verschillen, zegt de tabel niet. Deze les vraagt het de classifier zelf.",
            "sex": "De classifier krijgt de kolom sex helemaal niet, en geeft vrouwen toch veel minder vaak een ja. De tabel telt uitkomsten per groep; waarom ze verschillen, zegt hij niet. Zie *Het idee* in [Project: meet de bias van je eigen classifier](/course/open/rai-project-audit-bias?lang=nl).",
            "size": "Er zijn minder vrouwen, 5421 tegen 10.860 mannen, maar een deel met een ja is een deel binnen elke groep, dus de grootte van een groep verklaart het niet. De tabel telt uitkomsten; waarom ze verschillen, zegt hij niet. Zie [Project: meet de bias van je eigen classifier](/course/open/rai-project-audit-bias?lang=nl).",
        },
    )
    quick_why
    return (quick_why,)


@app.cell(hide_code=True)
def quick_check_distance():
    quick_distance = L.choice(
        "quick-distance",
        question="Met de mediane absolute afwijkingen als stappen was 81 worden de dichtstbijzijnde counterfactual van de aanvrager van 30; met de standaardafwijkingen als stappen opleidingsniveau 16. Waar lag dat aan?",
        options={
            "right": "Aan de afstand: die kiest wat dichtbij is",
            "model": "Het model werd tussendoor opnieuw getraind",
            "person": "De twee gebruikten een andere aanvrager",
        },
        correct="right",
        why={
            "right": "Hetzelfde model en dezelfde aanvrager geven met een andere afstand een andere dichtstbijzijnde counterfactual. In deze les mag een counterfactual maar één kolom veranderen, de uren per week, en die keuze van wat mag veranderen doet hetzelfde werk: schrijf hem op.",
            "model": "Het model bleef hetzelfde, en de aanvrager ook; alleen de stappen waarin de afstand telt, veranderden. Zie [Wat zou er moeten veranderen?](/course/open/rai-counterfactuals?lang=nl).",
            "person": "Het was dezelfde aanvrager van 30, en hetzelfde model; alleen de stappen waarin de afstand telt, veranderden. Zie [Wat zou er moeten veranderen?](/course/open/rai-counterfactuals?lang=nl).",
        },
    )
    quick_distance
    return (quick_distance,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        In [Project: meet de bias van je eigen classifier](/course/open/rai-project-audit-bias?lang=nl) heb je je classifier gemeten op de test set: hij geeft 7,0% van de vrouwen een ja en 24,8% van de mannen, een impact ratio van 0,28, terwijl hij de kolom sex nooit krijgt. Die audit liet zien waar de uitkomsten verschillen, niet waarom. Sindsdien leerde deze module je een model naar het waarom te vragen: op welke kolommen het leunt, hoe zijn antwoord meebeweegt met één kolom, wat het antwoord voor één persoon omhoog of omlaag duwt, en wat iemand zou moeten veranderen. Vandaag stel je die vragen aan dezelfde classifier, bewaar je drie nieuwe functies in je project, en schrijf je het deel van je auditrapport over de uitleg: wat die laat zien over de bias die je mat, en wat niet.

        **Een voorbeeld uit het dagelijks leven.** Een winkel ziet de verkoop in één maand met een vijfde dalen. Een uitsplitsing per afdeling laat zien dat bijna de hele daling bij de paraplu's zit. Dan weet de eigenaar waar de daling zit, maar niet waarom: een droge maand, een winkel verderop die ze goedkoper verkoopt, of een schap dat naar achteren is verhuisd. En of de eigenaar er iets aan moet doen, is weer een andere vraag. Uitleg bij een model is zo'n uitsplitsing: hij laat zien waar een verschil in het model zit, niet waarom de wereld is zoals ze is, en ook niet of het model mag beslissen zoals het doet.
        """),
        draw.project_steps(["Meet de bias van je classifier", "Leg je classifier uit"], 1,
                           here_label="← vandaag", title="Waar je project staat: je audit"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_gap():
    guess_gap = L.guess(
        "gap",
        question="Je classifier krijgt de kolom sex niet. Welke van de kolommen die hij wel krijgt, draagt volgens jou het grootste deel van het verschil tussen vrouwen en mannen?",
        options={
            "relationship": "relationship, met Husband en Wife",
            "hours": "hours_per_week",
            "right": "marital_status",
            "occupation": "occupation",
        },
        reveal="**marital_status**, met afstand. In de logit, het getal vóór de sigmoid, komen de vrouwen in de test set gemiddeld op −3,063 en de mannen op −1,647, een verschil van 1,416, ongeveer 1,42. Daarvan draagt marital_status 1,01, hours_per_week 0,18 en occupation 0,09; relationship, met onder zijn waarden Husband en Wife, draagt maar 0,04. *Het idee* laat zien hoe je een verschil zo verdeelt.",
    )
    guess_gap
    return (guess_gap,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Vier vragen.** Stap één mat uitkomsten per groep. De uitleg uit deze module stelt de classifier achter die uitkomsten telkens een andere vraag:

        - Op welke kolommen leunt hij, over alle mensen? Permutation importance, uit [Welke features doen ertoe?](/course/open/rai-permutation-importance?lang=nl): schud één kolom en kijk hoeveel minder vaak de classifier goed zit.
        - Hoe beweegt zijn antwoord mee met één kolom? Partial dependence, uit [Hoe een voorspelling meebeweegt met een feature](/course/open/rai-pdp-ice?lang=nl): geef iedereen dezelfde waarde en neem de gemiddelde kans. Hier voor vrouwen en mannen apart.
        - Wat duwt het antwoord voor één persoon omhoog of omlaag? Shapley-waarden, uit [Shapley-waarden](/course/open/rai-shapley-values?lang=nl). Gemiddeld over een groep laten ze zien waar een verschil tussen groepen zit.
        - Wat zou iemand moeten veranderen? Een counterfactual, uit [Wat zou er moeten veranderen?](/course/open/rai-counterfactuals?lang=nl), met alleen een verandering in iets waar die persoon zelf iets aan kan doen: hier meer uren werk per week.

        **Shapley-waarden van een logistische regressie, exact.** In [Shapley-waarden](/course/open/rai-shapley-values?lang=nl) liep je elke coalitie van vier kolommen af. Je classifier heeft geen coalities nodig. Een logistische regressie telt op: zijn logit is de bias term plus elke kolom keer zijn weight: de drie kolommen met getallen, gedeeld door hun grootste waarde, en de 0/1-kolommen. In een som voegt een kolom aan elke coalitie hetzelfde toe: zijn weight keer hoe ver de persoon van het gemiddelde van de achtergrond ligt, hier de mensen om op te trainen. Dat is de Shapley-waarde van de kolom op de logit, exact. Voor een kolom met woorden, zoals `marital_status`, tel je zijn 0/1-kolommen op. De waarden van één persoon zijn samen de logit van die persoon min de gemiddelde logit van de mensen om op te trainen. Bij een logistische regressie is dat gemiddelde ook de logit van de gemiddelde persoon, een verzonnen rij met elke kolom op zijn gemiddelde: −2,105 voor jouw classifier, een kans van 0,109. Dat is niet de gemiddelde kans van die mensen; die is 0,241. Op de kans zelf, na de sigmoid, gaat deze kortere weg niet op.

        **Wat de uitleg laat zien over je classifier.** Schud `marital_status` in de test set, en de classifier zit bij 76,5% van de mensen goed in plaats van bij 83,4%: een permutation importance van 0,070 (niet afgerond 0,8344 − 0,7648 = 0,0697), de grootste van de acht kolommen, voor `education_num` met 0,043 en `occupation` met 0,021 (seed 0). Gemiddeld over de vrouwen in de test set zijn de Shapley-waarden samen −0,96; over de mannen +0,46. Van dat verschil van 1,42 draagt `marital_status` 1,01. Van de vrouwen in de test set is 15,2% `Married-civ-spouse`, tegen 60,6% van de mannen, en tussen `Never-married` en `Married-civ-spouse` ligt 2,46 in de logit. Als elke vrouw in de test set `Married-civ-spouse` was, zou hun gemiddelde kans op een ja 0,318 zijn in plaats van 0,126; voor de mannen 0,381 in plaats van 0,295. En van de 5039 vrouwen met een nee zou 19,8% een ja krijgen door meer uren per week te werken, hooguit 99, met al het andere gelijk; van de 8168 mannen met een nee 43,9%. De meesten van hen zouden 60 uur per week of meer nodig hebben: 78,6% van die vrouwen en 74,0% van die mannen. Of iemand echt zo veel kan werken, hoort dus ook in het rapport. Berk Ustun, Alexander Spangher en Yang Liu waarschuwden dat recourse niet voor iedereen gelijk hoeft te zijn: wat recourse kost, kan voor mannen en vrouwen verschillen, ook als een model geslacht niet gebruikt. Hier verschilt zelfs of er recourse is.
        """),
        mo.Html(C.course_gap_figure("nl")),
        mo.md(r"""
        **Wat de uitleg niet kan laten zien.** De uitleg legt de classifier uit, niet de wereld. Dat hij op burgerlijke staat leunt, zegt niets over waarom getrouwde mensen in de enquête van 1994 vaker meer verdienen, en ook niets over of een beslissing over mensen op burgerlijke staat mag leunen. Die vragen zijn voor de mensen die met de classifier beslissen, en ze horen als open vragen in het rapport. Uitleg kan ook geruststellend lijken waar dat niet hoort: permutation importance geeft `race` −0,001, dus als je die kolom schudt, zit de classifier bijna even vaak goed. Toch is de Shapley-waarde van `race` gemiddeld over de mensen met `Other` −0,64, over `Black` −0,11, en over `White` +0,03. Oefening 4 vraagt wat je daarvan moet denken.

        **Uitleg kan mooier gemaakt worden.** Dylan Slack en anderen lieten zien hoe iemand die een bias wil verbergen LIME en SHAP kan misleiden. Allebei leggen ze een model uit door het verzonnen inputs te sturen, verstoorde versies van echte, die vaak niet op echte data lijken. Een "scaffolding"-classifier vraagt eerst of een input er echt uitziet: echte inputs gaan naar het model met bias, verzonnen inputs naar een onschuldig model. Bij LIME nam een onschuldige kolom de bovenste plaats over van de gevoelige, voor 100% van de achtergehouden gevallen in twee datasets, en een kolom over de lening deed dat voor 91% in een derde; bij SHAP met één onschuldige kolom verloor de gevoelige kolom de bovenste plaats voor 84% tot 100%, afhankelijk van de data, en met twee onschuldige kolommen lukte de aanval minder vaak. SHAP misleiden had een prijs: de scaffold was het dan maar bij 75% tot 91% van de echte inputs eens met het model met bias. Hun conclusie: uitleg die op verstoorde inputs steunt, zoals LIME en SHAP, is niet betrouwbaar. De aanval vraagt iemand die erop uit is te misleiden; je eigen classifier is zo'n scaffold niet. Ulrich Aïvodji en anderen hebben een naam voor een model mooier voorstellen dan het is, fairwashing: de valse indruk wekken dat een model zich aan bepaalde ethische waarden houdt. Op UCI Adult had een random forest een demographic-parity-unfairness van 0,13 bij de mensen voor wie hij uitgelegd moest worden, en 0,18 op de test set; de eenvoudige rule lists die hun methode vond om hem na te doen, kwamen er op fairness beter uit: de beste deed het model voor 90,8% na (een fidelity van 0,908), met een unfairness van 0,058.

        **Laat een spoor na.** Margaret Mitchell en anderen noemen zelf een beperking van model cards: hoe bruikbaar en juist een model card is, hangt af van de integriteit van wie hem maakte. Dat geldt ook voor je audit, een first-party audit door iemand die het belang deelt van wie de classifier bouwde. Je kunt er wel voor zorgen dat een ander hem kan nagaan: Inioluwa Deborah Raji en anderen vragen dat een interne audit een spoor van documenten nalaat, uit elke fase. Schrijf dus op hoe elke uitleg gemaakt is: voor welke classifier, op welke mensen, met welke seeds en welke achtergrond.

        **Het auditrapport.** Een audit eindigt in een **auditrapport**: wat er gemeten en uitgelegd is, hoe, wat eruit kwam, en wat er niet gemeten is. Volgens Mitchell en anderen kan een model card bij de details van een model ook vermelden hoe het uitgelegd wordt, en moeten uitkomsten uitgesplitst zijn, per groep: leg dus ook per groep uit, zoals je hier per geslacht doet.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **auditrapport** | het document waarin een audit eindigt: wat er gemeten en uitgelegd is, hoe, met welke uitkomst, en wat er niet gemeten is |
        """),
        L.watch_out("Uitleg is een aanwijzing over een model, geen oordeel erover. Hij kan laten zien waar een verschil zit, en hij kan geruststellend lijken waar dat niet hoort. Geef bij elke uitleg hoe hij gemaakt is, en presenteer hem nooit als bewijs dat de classifier fair is."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Het tweede deel van je audit:

    1. Begin bij wat stap één mat: het verschil in het deel met een ja, en de classifier waarop het gemeten is.
    2. Over alle mensen: op welke kolommen leunt de classifier? Permutation importance op de test set, met een paar seeds. Dat is `importance`.
    3. Hoe beweegt zijn gemiddelde antwoord mee met één kolom, per groep? Partial dependence voor vrouwen en voor mannen. Dat is `dependence`.
    4. Voor één persoon of een groep: wat duwt één persoon, of een groep gemiddeld, omhoog of omlaag? Shapley-waarden; bij een logistische regressie exact op de logit, elke weight keer hoe ver de persoon van het gemiddelde van de achtergrond ligt.
    5. Wat zou iemand moeten veranderen, en kan die persoon dat? Een counterfactual met alleen een verandering in iets waar die persoon zelf iets aan kan doen, hier de uren per week; vergelijk hoeveel vrouwen en mannen er een hebben. Dat is `fewest_hours`.
    6. Schrijf op wat de uitleg laat zien, hoe elke uitleg gemaakt is, en wat hij niet kan laten zien.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 4 voor één persoon: de vrouw in rij 34982 van de tabel, in de test set: 40 jaar, een bachelordiploma, 50 uur per week in `Exec-managerial`, nooit getrouwd, en een nee van de classifier. De cel traint de classifier van stap één, zoals in je `model.py`. `_model.coef_[0]` bevat zijn weights, één per kolom, in de volgorde van de kolommen van `_X`. `_X.loc[34982]` is de rij met het label 34982, en `_X[_train].mean()` het gemiddelde van de mensen om op te trainen: aftrekken geeft hoe ver de persoon ervan ligt, kolom voor kolom, en vermenigvuldigen met de weights geeft de Shapley-waarde van elke kolom van `_X`. `_shares.index.str.startswith(_column)` is True voor de 0/1-kolommen waarvan de naam begint met de naam van de kolom, zoals `marital_status_Never-married`, en bij een kolom met getallen, zoals `age`, voor de kolom zelf. Voer de cel uit; dat duurt een seconde of twee.
    """)
    return


@app.cell
def example_shapley():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    # de classifier van stap één, zoals in work/projects/audit/model.py
    _people = pd.read_csv("data/adult.csv.gz")
    _train = _people["split"] == "train"
    _words = ["workclass", "marital_status", "occupation", "relationship", "race"]
    _columns = ["age", "education_num", "hours_per_week"] + _words
    _X = pd.get_dummies(_people[_columns], columns=_words, dtype=float)
    _X = _X / _X.max()
    _model = LogisticRegression(max_iter=1000)
    _model.fit(_X[_train], _people["income"][_train] == ">50K")

    # de Shapley-waarden op de logit van één persoon: elke weight keer hoe ver de persoon van het gemiddelde ligt
    _shares = _model.coef_[0] * (_X.loc[34982] - _X[_train].mean())
    for _column in _columns:
        print(_column, round(_shares[_shares.index.str.startswith(_column)].sum(), 3))
    print("sum", round(_shares.sum(), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De classifier geeft deze persoon een kans van 0,278, een logit van −0,954, tegen −2,105 voor de gemiddelde persoon; de acht waarden zijn samen het verschil, 1,151. `education_num` duwt omhoog met 0,863 en `occupation` met 0,759; `marital_status` duwt omlaag met 1,211, meer dan elke andere kolom. Met `Married-civ-spouse` en de rest gelijk zou de kans 0,818 zijn, een ja. Die veranderde rij komt in de data bijna niet voor, met `relationship` nog steeds `Not-in-family`: 23 van de 48.842 rijen zijn `Married-civ-spouse` en `Not-in-family`. Zo'n wat-als verandert één kolom en laat de rest staan. Oefening 3 vraagt in plaats daarvan wat deze persoon zou kunnen veranderen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_importance():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. importance: op welke kolommen leunt je classifier?

        `importance(scores_of, people, column, seed)` krijgt een functie `scores_of`, die voor elke rij van een tabel de kans op een ja van de classifier geeft, een tabel `people` met een kolom `rich`, 1 voor de mensen die meer dan 50.000 dollar per jaar verdienen en 0 voor de anderen, de naam `column` van een kolom om te schudden, en een `seed`. `shuffled` is een kopie van `people`, gemaakt met `.copy()`, waarvan de kolom `column` geschud is: `.sample(frac=1, random_state=seed)` zet de waarden van de kolom in een willekeurige volgorde, en `.to_numpy()` neemt ze zonder hun rijlabels, zodat ze in die nieuwe volgorde terechtkomen. Vervang de `...` door drie regels: `right`, het deel van `people` waarbij de classifier goed zit, een ja (een kans boven 0,5) als `rich` 1 is en een nee als het 0 is; `after`, hetzelfde voor `shuffled`; en `return right - after`. De check roept `importance(scores_of, PEOPLE, "marital_status", 0)` aan, met `PEOPLE` de test set en `scores_of` de classifier van de course, en daarna andere kolommen, seeds en tabellen. Klopt hij, dan bewaart de check hem in je project, in `work/projects/audit/importance.py`.
        """),
        L.task("Geef terug hoeveel minder vaak de classifier goed zit op `people` na het schudden."),
    ])
    return


@app.cell(hide_code=True)
def exercise_importance():
    box_importance = L.code_box("importance", lines=7)
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
            "Stap 2 van het recept. In stap één vergeleek `report` elke ja met `rich`. Hier vraag je voor elke persoon of de ja van de classifier, een kans boven 0,5, klopt met `rich == 1`, en neem je het deel dat klopt.",
            "`(scores_of(people) > 0.5) == (people[\"rich\"] == 1)` is True voor elke persoon bij wie de classifier goed zit; je wilt het deel True, zoals in stap één. Doe hetzelfde voor `shuffled`, en trek af.",
            "`after = ((scores_of(shuffled) > 0.5) == (shuffled[\"rich\"] == 1)).___()`",
        ],
        why="`right = ((scores_of(people) > 0.5) == (people[\"rich\"] == 1)).mean()`, `after = ((scores_of(shuffled) > 0.5) == (shuffled[\"rich\"] == 1)).mean()` en `return right - after`. Voor `marital_status` met seed 0 gaat het deel van 0,8344 naar 0,7648, een daling van 0,0697, die de check als 0,070 laat zien.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_dependence():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. dependence: hoe beweegt het antwoord mee met één kolom?

        `dependence(scores_of, people, column, value)` geeft de partial dependence bij één waarde: de gemiddelde kans op een ja als iedereen in `people` de waarde `value` heeft in de kolom `column`. `changed` is al een kopie van `people`. Vervang de `...` door twee regels: geef elke rij van `changed` de waarde `value` in de kolom `column`, en geef het gemiddelde van `scores_of(changed)` terug. De check roept `dependence(scores_of, WOMEN, "marital_status", "Married-civ-spouse")` aan, met `WOMEN` de vrouwen in de test set, en daarna de mannen, een andere waarde en een kolom met getallen. Hij bewaart je functie in `work/projects/audit/dependence.py`.
        """),
        L.task("Geef de gemiddelde kans op een ja terug met iedereen op `value` in de kolom `column`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_dependence():
    box_dependence = L.code_box("dependence", lines=5)
    box_dependence
    return (box_dependence,)


@app.cell(hide_code=True)
def check_dependence(box_dependence):
    result_dependence = L.check("dependence", box_dependence.value)
    result_dependence
    return (result_dependence,)


@app.cell(hide_code=True)
def hints_dependence():
    L.hints(
        "dependence",
        [
            "Stap 3 van het recept. In [Hoe een voorspelling meebeweegt met een feature](/course/open/rai-pdp-ice?lang=nl) gaf je iedereen dezelfde waarde en nam je het gemiddelde van de antwoorden van het model. Hier komen de antwoorden van `scores_of`.",
            "`changed[column] = value` zet `value` in elke rij van die kolom. Daarna geeft `scores_of(changed)` één kans per rij.",
            "`return scores_of(changed).___()`",
        ],
        why="`changed[column] = value` en `return scores_of(changed).mean()`. Als elke vrouw in de test set `Married-civ-spouse` was, zou hun gemiddelde kans 0,318 zijn in plaats van 0,126.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_hours():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. fewest_hours: wat zou één persoon kunnen veranderen?

        Een counterfactual met één verandering waar iemand zelf iets aan kan doen: de uren werk per week, met al het andere gelijk. `fewest_hours(scores_of, person, top)` krijgt een tabel `person` met één rij en geeft het kleinste aantal uren per week, van 1 tot `top`, waarbij de classifier die persoon een ja geeft, of `None` als geen aantal tot `top` dat doet. `changed` is een kopie van `person`, en de loop gaat met `hours` van 1 tot `top`. Vervang de `...` door drie regels: geef `changed` de uren van deze ronde in de kolom `hours_per_week`, vraag met een `if` of de kans van `changed` boven 0,5 ligt, en geef dan `hours` terug. `scores_of(changed)` geeft een array met één kans, en `[0]` pakt die. Het eerste aantal dat een ja geeft is het kleinste, dus de functie kan daar stoppen. De check roept `fewest_hours(scores_of, ONE, 99)` aan, met `ONE` de persoon uit het uitgewerkte voorbeeld, en daarna andere mensen en andere waarden van `top`. Hij bewaart je functie in `work/projects/audit/recourse.py`.
        """),
        L.task("Geef het kleinste aantal uren per week terug, van 1 tot `top`, dat `person` een ja geeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hours():
    box_hours = L.code_box("hours", lines=8)
    box_hours
    return (box_hours,)


@app.cell(hide_code=True)
def check_hours(box_hours):
    result_hours = L.check("hours", box_hours.value)
    result_hours
    return (result_hours,)


@app.cell(hide_code=True)
def hints_hours():
    L.hints(
        "hours",
        [
            "Stap 5 van het recept. In stap één liep `threshold_for` in een loop de thresholds af en hield hij de laatste die slaagde. Hier stop je bij het eerste aantal uren dat slaagt, met `return`.",
            "`changed[\"hours_per_week\"] = hours` geeft de kopie de uren van deze ronde. `scores_of(changed)` bevat één kans, op `[0]`; vergelijk die met 0,5, zoals overal in deze les.",
            "`if scores_of(changed)[0] ___ 0.5:`",
        ],
        why="`changed[\"hours_per_week\"] = hours`, dan `if scores_of(changed)[0] > 0.5:` en daaronder `return hours`. De persoon uit het uitgewerkte voorbeeld zou vanaf 82 uur per week een ja krijgen, 32 meer dan nu.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel legt je eigen classifier uit, die in `work/projects/audit/model.py`, met je eigen functies: je `report` uit stap één per geslacht, je `importance` voor elke kolom over drie seeds, je `dependence` voor vrouwen en mannen bij elke waarde van `marital_status`, en je `fewest_hours` voor 100 vrouwen en 100 mannen met een nee, willekeurig gekozen. De Shapley-waarden per geslacht rekent de course zelf uit, met de weights van je classifier, zoals het uitgewerkte voorbeeld doet. Dat duurt een paar seconden, en de cel draait opnieuw na elke oefening.
    """)
    return


@app.cell(hide_code=True)
def demo_project(result_dependence, result_hours, result_importance):
    _r = (result_importance, result_dependence, result_hours)   # zodat deze cel na elke check draait
    _audit = C.demo()
    if _audit is None:
        _out = mo.md("*Maak eerst oefening 1: dan staat je `importance` in je project, en gebruikt deze cel hem.*")
    elif isinstance(_audit, str):
        _out = mo.md(C.broken(_audit, L.lang))
    else:
        _parts = C.demo_parts(_audit, L.lang)
        _out = mo.vstack([mo.md(_p) if _kind == "md" else mo.Html(_p) for _kind, _p in _parts])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_race():
    mo.md(r"""
    ### Oefening 4. Doet race er niet toe?
    """)
    return


@app.cell(hide_code=True)
def choice_race_cell():
    choice_race = L.choice(
        "race",
        question="Je `importance` geeft `race` −0,001: schud de kolom, en de classifier zit bijna even vaak goed als ervoor. Een collega schrijft in het rapport: \"De classifier gebruikt race niet.\" Wat is er mis met die zin?",
        options={
            "nothing": "Niets: race schudden verandert niets wat telt",
            "right": "Dat getal telt goede antwoorden, niet wie zakt",
            "seed": "Met een andere seed krijgt race misschien meer",
            "small": "De test set heeft te weinig mensen om dat te zien",
        },
        correct="right",
        why={
            "nothing": "De classifier gebruikt race wel: hij heeft een weight per waarde, en gemiddeld over de mensen met `Other` duwt race de logit 0,64 omlaag, over `Black` 0,11. Als je de kolom schudt, verandert nauwelijks hoe vaak hij goed zit, maar dat is iets anders dan de kolom niet gebruiken.",
            "right": "Permutation importance vraagt hoeveel minder vaak de classifier goed zit met de kolom geschud. Race duwt sommige mensen omlaag, maar brengt weinig antwoorden over 0,5 heen, en 85,7% van de test set is `White`, dus schudden deelt vooral weer `White` uit. De Shapley-waarden laten de duw zien: gemiddeld −0,64 voor `Other`, −0,11 voor `Black`, +0,03 voor `White`. Onbelangrijk voor hoe vaak hij goed zit en niet gebruikt zijn twee verschillende beweringen.",
            "seed": "Met seeds 0 tot en met 4 krijgt race van −0,001 tot 0,000: altijd ongeveer 0. Meer seeds veranderen dat niet, want het getal beantwoordt een andere vraag: hoe vaak de classifier goed zit, niet wie hij omlaag duwt.",
            "small": "De test set heeft 16.281 mensen, en de Shapley-waarden laten zien dat race de logit duidelijk laat bewegen, gemiddeld −0,64 voor de mensen met `Other`. Het aantal mensen is het probleem niet: permutation importance beantwoordt een andere vraag, hoe vaak de classifier goed zit.",
        },
    )
    choice_race
    return (choice_race,)


@app.cell(hide_code=True)
def text_exercise_report():
    mo.md(r"""
    ### Oefening 5. De uitleg in je auditrapport
    """)
    return


@app.cell(hide_code=True)
def explain_report_cell():
    explain_report = L.explain(
        "report",
        question="Schrijf het deel van je auditrapport over de uitleg, voor iemand die deze les niet deed: wat laat de uitleg zien over het verschil tussen vrouwen en mannen dat stap één mat, wat kan hij niet laten zien, en hoe is hij gemaakt? Gebruik de getallen van je eigen project, of die van de course.",
        model_answer="Stap één mat een deel met een ja van 0,070 bij de vrouwen in de test set en 0,248 bij de mannen, een impact ratio van 0,28, terwijl de classifier geen geslacht krijgt. De uitleg laat zien waar dat verschil in de classifier zit. Als je marital_status schudt, zit hij bij 0,765 van de mensen goed in plaats van bij 0,834, de grootste permutation importance (seed 0, test set). Gemiddeld per geslacht leggen de Shapley-waarden op de logit, tegen de mensen om op te trainen, 1,01 van het verschil van 1,42 bij marital_status: 15,2% van de vrouwen is Married-civ-spouse tegen 60,6% van de mannen. Met iedereen Married-civ-spouse zou de gemiddelde kans op een ja 0,318 zijn voor de vrouwen en 0,381 voor de mannen, in plaats van 0,126 en 0,295. Van de mensen met een nee kunnen 19,8% van de vrouwen en 43,9% van de mannen een ja krijgen door meer uren per week te werken, hooguit 99, de meesten 60 uur of meer. De uitleg laat niet zien waarom burgerlijke staat in de data van 1994 samengaat met inkomen, en ook niet of een beslissing erop mag leunen; en race, met een permutation importance van ongeveer 0, duwt de logit toch omlaag, gemiddeld met 0,64 voor de mensen met Other. Dit alles geldt voor deze classifier op deze test set; de seeds en de achtergrond staan erbij, zodat een ander het kan nagaan.",
        points=[
            "je zegt met getallen waar het verschil volgens de uitleg zit: welke kolom er het meeste van draagt, volgens permutation importance en volgens de Shapley-waarden per geslacht",
            "je zegt wat de uitleg niet kan laten zien: waarom de data zo zijn, of de classifier zo mag beslissen, en dat een kolom met een importance van ongeveer 0, zoals race, mensen toch omlaag kan duwen",
            "je zegt hoe de uitleg gemaakt is, voor welke classifier, op welke mensen en met welke seeds of achtergrond, zodat een ander hem kan nagaan",
        ],
    )
    explain_report
    return (explain_report,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De uitleg uit deze module, [Welke features doen ertoe?](/course/open/rai-permutation-importance?lang=nl), [Hoe een voorspelling meebeweegt met een feature](/course/open/rai-pdp-ice?lang=nl), [Shapley-waarden](/course/open/rai-shapley-values?lang=nl) en [Wat zou er moeten veranderen?](/course/open/rai-counterfactuals?lang=nl), stelde elk één vraag aan de classifier uit [Project: meet de bias van je eigen classifier](/course/open/rai-project-audit-bias?lang=nl). [Tools voor uitleg](/course/open/rai-xai-tools?lang=nl) liet zien hoe SHAP en DiCE zulk werk op grote schaal doen; bij een logistische regressie hadden de Shapley-waarden op de logit helemaal geen tool nodig. Je project `audit` bevat nu de functies van beide stappen, met `importance.py`, `dependence.py` en `recourse.py` als nieuwe van vandaag. De volgende les, *Kan een model zichzelf uitleggen?*, vraagt of je de uitleg van een language model bij zijn eigen antwoord kunt vertrouwen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_race, explain_report, result_dependence, result_hours, result_importance):
    L.wrap_up(result_importance, result_dependence, result_hours, choice_race, explain_report)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
