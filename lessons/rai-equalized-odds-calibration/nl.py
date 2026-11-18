# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-equalized-odds-calibration. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-equalized-odds-calibration", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl) en [Evenveel ja per groep](/course/open/rai-demographic-parity?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_recall():
    quick_recall = L.choice(
        "quick-recall",
        question="Van 50 mensen die echt meer dan 50.000 dollar verdienen, geeft een model er 40 een ja. Wat is de recall voor hen?",
        options={
            "miss": "0,2, het deel dat het mist",
            "right": "0,8, want 40 / 50",
            "count": "40 mensen",
        },
        correct="right",
        why={
            "miss": "Recall telt de mensen die het model vindt, niet die het mist: 40 / 50 = 0,8. Zie *Het idee* in [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
            "right": "Van de 50 die meer verdienen, vindt het model er 40: een recall van 0,8.",
            "count": "Recall is een deel, geen aantal: 40 van de 50, dus 0,8. Zie *Het idee* in [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
        },
    )
    quick_recall
    return (quick_recall,)


@app.cell(hide_code=True)
def quick_check_parity():
    quick_parity = L.choice(
        "quick-parity",
        question="Met een threshold per groep kregen vrouwen en mannen allebei in 0,19 van de gevallen een ja. Wat bleef ongelijk?",
        options={
            "rates": "Het deel van elke groep met een ja",
            "right": "De precision: 0,43 tegen 0,73",
            "nothing": "Niets: het model was toen in orde",
        },
        correct="right",
        why={
            "rates": "Dat was gelijk, 0,19 voor allebei: dat deed de threshold per groep. Zie *Probeer het* in [Evenveel ja per groep](/course/open/rai-demographic-parity?lang=nl).",
            "right": "Van de vrouwen met een ja verdiende 0,43 echt meer, van de mannen 0,73. Evenveel ja per groep maakt niet dat een ja hetzelfde betekent.",
            "nothing": "Evenveel ja per groep is één criterium. Een ja betekende nog steeds iets anders voor vrouwen dan voor mannen: een precision van 0,43 tegen 0,73. Zie *Probeer het* in [Evenveel ja per groep](/course/open/rai-demographic-parity?lang=nl).",
        },
    )
    quick_parity
    return (quick_parity,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Evenveel ja per groep](/course/open/rai-demographic-parity?lang=nl) gaf een threshold per groep vrouwen en mannen even vaak een ja, maar keek niet wie de ja verdiende. Deze les stelt twee andere vragen. Zijn de vergissingen van een model gelijk over de groepen verdeeld? En betekent de score in elke groep hetzelfde? Elke vraag is een criterium van group fairness, en op de tabel UCI Adult trekken ze elk een andere kant op.

    **Een voorbeeld uit het dagelijks leven.** Een weerdienst zegt *70 procent kans op regen*. Als het daarna op ongeveer 7 van de 10 zulke dagen regent, in de stad en aan de kust, betekent de voorspelling op beide plekken hetzelfde. Een andere vraag: hoe vaak waarschuwde de dienst op dagen die droog bleven toch voor regen, en hoe vaak bleef een waarschuwing uit op dagen dat het regende? Kan de voorspelling op beide plekken hetzelfde betekenen én beide vergissingen op beide plekken even vaak maken, als het aan de kust vaker regent? Alleen als de dienst zich nooit vergist. Deze les laat zien waarom.
    """)
    return


@app.cell(hide_code=True)
def predict_missed():
    guess_missed = L.guess(
        "missed",
        question="Met één threshold van 0,5 is het model zonder `sex` ongeveer even precies voor vrouwen en mannen: van wie een ja krijgt, verdient 0,67 en 0,69 echt meer dan 50.000 dollar. Van de mannen die echt meer verdienen, krijgt 0,57 een ja. Hoeveel van zulke vrouwen krijgen een ja, denk je?",
        options={
            "same": "Ongeveer evenveel: 0,57",
            "right": "Minder: zo'n 0,44",
            "more": "Meer: zo'n 0,70",
        },
        reveal="**Minder: zo'n 0,44.** Een ja betekent ongeveer hetzelfde voor vrouwen en mannen, maar het model vindt minder van de vrouwen die meer verdienen. In de data verdienen minder vrouwen dan mannen meer, 0,11 tegen 0,30, en het model geeft de vrouwen die dat wel doen gemiddeld een lagere kans dan zulke mannen: 0,45 tegen 0,54. Meer van hen blijven onder de threshold. De demo hieronder laat het zien.",
    )
    guess_missed
    return (guess_missed,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Gelijke fouten.** Een model dat ja of nee zegt, kan zich op twee manieren vergissen: het kan iemand missen die echt positief is, en het kan een ja geven aan iemand die dat niet is. Tel ze allebei per groep. De recall van een groep, het deel van de positieven dat een ja krijgt, heet ook de **true positive rate**. De **false positive rate** is het deel van de negatieven dat toch een ja krijgt: $FP / (FP + TN)$. Moritz Hardt, Eric Price en Nathan Srebro (2016) vragen dat ze allebei in elke groep gelijk zijn, en noemen dat **equalized odds**. Barocas, Hardt en Narayanan noemen het separation: het vraagt om gelijke foutpercentages, met één vraag erachter: wie draait er op voor een verkeerde voorspelling? Anders dan demographic parity laat het een model toe dat altijd gelijk heeft, maar het straft modellen die alleen voor de meerderheid goed werken. Hardt en anderen noemen ook een zwakkere versie, die volgens hen nog steeds interessant is: **equal opportunity**, waarbij alleen de true positive rate gelijk hoeft te zijn. Hun voorbeeld is een lening: wie de lening terugbetaalt, moet evenveel kans hebben om haar te krijgen.

    **Gelijke betekenis.** Het model rekent ook voor iedereen een kans uit. Het is gekalibreerd als die kansen net zo vaak uitkomen als ze zeggen: van alle mensen die het ongeveer 0,7 geeft, is ongeveer 70 procent echt positief. Barocas, Hardt en Narayanan zeggen het zo: van alle mensen met een score r is een deel r positief. **Calibration within groups** vraagt dat dat in elke groep geldt: dan betekent een score hetzelfde, in welke groep je ook zit. Bij een ja of een nee is de eenvoudigste controle dezelfde precision in elke groep, wat Alexandra Chouldechova (2017) predictive parity noemt; die is zwakker, want hij zegt niets over de mensen met een nee. Jon Kleinberg, Sendhil Mullainathan en Manish Raghavan (2016) zeggen waarom het ertoe doet: zonder kalibratie heeft wie beslist er belang bij om twee mensen met dezelfde score uit verschillende groepen anders te behandelen. Het zegt iets over alle mensen met dezelfde score, niet over één van hen: kalibratie vertelt niets over de uitkomst van een bepaalde persoon. Barocas, Hardt en Narayanan noemen het sufficiency, en zien dat het vaak vanzelf komt, bij benadering, als je een model op de gewone manier traint, op één voorwaarde: het model moet kunnen zien bij welke groep iemand hoort. Het model zonder `sex` kan dat: in [Een kolom weglaten is niet genoeg](/course/open/rai-unawareness-proxies?lang=nl) raadde een model het geslacht uit de andere kolommen goed voor 0,84 van de mensen.

    **Waarom niet allebei.** Neem 1000 mensen uit een groep, de **base rate**, het deel dat echt positief is, en de precision en recall van een model in die groep. Dan volgt de false positive rate. Met de getallen van de zwarte verdachten in [COMPAS en een sollicitatietool](/course/open/rai-case-compas-amazon?lang=nl), waar een markering gemiddeld of hoog risico betekent:

    1. Van elke 1000 zwarte verdachten werden er 514 binnen twee jaar opnieuw opgepakt: een base rate van 0,514.
    2. De score markeerde 0,72 van hen, de recall: 370 terechte markeringen.
    3. De precision was 0,63: de 370 terechte markeringen zijn 0,63 van alle markeringen, dus waren er 370 / 0,63 = 587 markeringen, en daarvan waren er 587 − 370 = 217 onterecht.
    4. Van de 1000 werden er 486 niet opnieuw opgepakt, dus de false positive rate is 217 / 486 = 0,45.

    Dezelfde stappen met de getallen van de witte verdachten, een base rate van 0,394, een recall van 0,52 en een precision van 0,59, geven 0,23. Geef nu twee groepen dezelfde precision en dezelfde recall: de base rate verandert stap 1 en stap 4 nog steeds, dus verschillen hun false positive rates. Kleinberg en anderen bewezen het in het algemeen: calibration within groups en gelijke foutpercentages kunnen alleen samen als het model perfect is of als de groepen dezelfde base rate hebben, hoe de scores ook tot stand komen, door een model of door mensen. Chouldechova bewees een versie voor één threshold. Dat maakt fairness niet onmogelijk. Het betekent dat deze criteria niet allemaal tegelijk kunnen, dus moet je kiezen.

    **COMPAS liet het zien.** In 2016 vonden journalisten van ProPublica dat de score COMPAS zwarte verdachten bijna twee keer zo vaak onterecht markeerde als witte verdachten: de false positive rates van 0,45 en 0,23 hierboven. Alles bij elkaar, schreven ze, vergiste de score zich bij zwarte en witte verdachten ongeveer even vaak, maar op heel verschillende manieren. De maker, Northpointe, antwoordde met gelijke betekenis. Volgens Barocas, Hardt en Narayanan benadrukte het bedrijf in zijn weerwoord kalibratie per groep, terwijl ProPublica zonder het te zeggen gelijke false positive rates als criterium nam. In een reactie op ProPublica vonden Anthony Flores en anderen (2016) dat een COMPAS-score ongeveer dezelfde kans op een nieuw misdrijf betekent, wat de afkomst ook is. Chouldechova liet zien waarom beide bevindingen tegelijk klopten: in de data werd 51 procent van de zwarte verdachten opnieuw opgepakt en 39 procent van de witte, en COMPAS is ongeveer gekalibreerd en heeft predictive parity bij een threshold van 4 of hoger, maar geen gelijke foutpercentages. Met die base rates kon de score niet allebei hebben.

    **Probeer het.** De cel hieronder traint het model zonder de kolom `sex` uit [Evenveel ja per groep](/course/open/rai-demographic-parity?lang=nl) opnieuw. Met één threshold van 0,5 laat de cel per geslacht de recall, de false positive rate en de precision zien. Daarna controleert ze gelijke betekenis: per band van kansen, van 0 tot 0,2 tot en met 0,8 tot 1, het deel van de vrouwen en van de mannen dat echt meer verdient. Ten slotte krijgen de vrouwen een lagere threshold, zodat hun recall die van de mannen wordt: equal opportunity. `*` tussen twee maskers houdt de rijen waarin allebei `True` zijn. Voer de cel uit; het duurt een paar seconden.
    """)
    return


@app.cell
def demo_errors():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    # 48.842 mensen uit een Amerikaans onderzoek van 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _train = _people["split"] == "train"
    _test = _people["split"] == "test"
    _tested = _people[_test]
    _rich = _tested["income"] == ">50K"
    _poor = _tested["income"] == "<=50K"
    _women = _tested["sex"] == "Female"
    _men = _tested["sex"] == "Male"
    print("echt meer: vrouwen", round(_rich[_women].mean(), 2), "mannen", round(_rich[_men].mean(), 2))

    # het model zonder de kolom sex, zoals in de vorige lessen
    _words = ["workclass", "marital_status", "occupation", "relationship", "race"]
    _columns = ["age", "education_num", "hours_per_week"] + _words
    _X = pd.get_dummies(_people[_columns], columns=_words, dtype=float)
    _X = _X / _X.max()
    _model = LogisticRegression(max_iter=1000)
    _model.fit(_X[_train], _people["income"][_train] == ">50K")
    _score = _model.predict_proba(_X[_test])[:, 1]   # voor elke persoon de kans op een ja


    def _show(name, yes):
        # per geslacht: de recall, de false positive rate en de precision van de ja's
        print(name)
        for sex, group in {"vrouwen": _women, "mannen": _men}.items():
            recall = yes[group * _rich].mean()
            false_positive_rate = yes[group * _poor].mean()
            precision = _rich[group * yes].mean()
            print("  ", sex, "| recall", round(recall, 2),
                  "| false positive rate", round(false_positive_rate, 2), "| precision", round(precision, 2))


    # 1. één threshold voor iedereen: een ja waar de kans boven 0.5 ligt
    _one = _score > 0.5
    _show("één threshold van 0.5", _one)

    # 2. gelijke betekenis: per band van kansen het deel van de vrouwen en van de mannen dat meer verdient
    print("per band van kansen, het deel dat echt meer verdient")
    for _low, _high in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1)]:
        _band = (_score >= _low) * (_score < _high)
        print("  ", _low, "tot", _high, "| vrouwen", round(_rich[_band * _women].mean(), 2),
              "| mannen", round(_rich[_band * _men].mean(), 2))

    # 3. gelijke recall: een lagere threshold voor de vrouwen, zodat van de vrouwen die meer verdienen
    # een even groot deel een ja krijgt als van de mannen die meer verdienen
    _recall_men = _one[_men * _rich].mean()
    _threshold_women = np.quantile(_score[_women * _rich], 1 - _recall_men)
    print("de threshold van de vrouwen voor gelijke recall:", round(_threshold_women, 2))
    # elke vrouw krijgt de threshold van de vrouwen, elke man 0.5
    _threshold = _threshold_women * _women + 0.5 * _men
    _show("gelijke recall", _score > _threshold)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.picture_meaning({"title": "Betekent een kans hetzelfde voor vrouwen en mannen?",
                                   "xlabel": "gemiddelde kans in de band", "ylabel": "verdient echt meer",
                                   "line": "kans = deel", "key": "open: vrouwen, gevuld: mannen; één punt per band van kansen",
                                   "decimal": ",",
                                   "alt": "Vijf punten voor vrouwen en vijf voor mannen, één per band van kansen. Horizontaal de gemiddelde kans van het model in de band, verticaal het deel dat echt meer verdient. Vrouwen: 0,05 en 0,03, 0,28 en 0,25, 0,49 en 0,45, 0,70 en 0,69, 0,86 en 0,88. Mannen: 0,06 en 0,06, 0,29 en 0,32, 0,49 en 0,50, 0,70 en 0,69, 0,86 en 0,84. Alle punten liggen dicht bij de gestippelde lijn, waar de kans gelijk is aan het deel."})),
        mo.Html(C.picture_errors({"title": "De fouten van het model per geslacht",
                                  "rows": ["Eén threshold van 0,5", "Gelijke recall"],
                                  "xlabels": ["recall", "false positive rate", "precision"],
                                  "key": "open: vrouwen, gevuld: mannen", "decimal": ",",
                                  "alt": "Drie panelen met elk twee rijen. De recall van vrouwen en mannen: één threshold 0,44 en 0,57, gelijke recall 0,57 en 0,57. De false positive rate: één threshold 0,03 en 0,11, gelijke recall 0,05 en 0,11. De precision: één threshold 0,67 en 0,69, gelijke recall 0,59 en 0,69."})),
        mo.md(r"""
        Met één threshold betekenen de kansen van het model ongeveer hetzelfde voor vrouwen en mannen: de punten van allebei liggen dicht bij de gestippelde lijn, en van de vrouwen en de mannen met een ja verdient 0,67 en 0,69 echt meer. Bijna gelijke betekenis dus. Maar de fouten verschillen. Van de vrouwen die meer verdienen, krijgt 0,44 een ja, van de mannen 0,57: het model mist meer van de vrouwen. En van de mensen die niet meer verdienen, krijgt 0,03 van de vrouwen toch een ja, en 0,11 van de mannen. Wie welke kosten draagt, hangt af van wat de ja oplevert.

        Met een lagere threshold voor de vrouwen, 0,37, waarvoor je bij de beslissing weer de kolom `sex` nodig hebt, wordt hun recall die van de mannen, 0,57: equal opportunity. Hun false positive rate stijgt naar 0,05, nog steeds niet die van de mannen, 0,11, en hun precision daalt naar 0,59: een ja betekent nu minder voor een vrouw dan voor een man. Het model heeft even vaak gelijk als eerst, in 0,83 van de gevallen. Ook gelijke false positive rates lukten Hardt en anderen op kredietscores niet met één threshold per groep: daar was toeval voor nodig, zodat sommige mensen vlak bij de threshold door het opgooien van een munt een ja zouden krijgen.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **true positive rate** | een andere naam voor de recall: van de mensen die echt positief zijn, het deel dat een ja krijgt |
        | **false positive rate** | van de mensen die echt negatief zijn, het deel dat toch een ja krijgt: $FP / (FP + TN)$ |
        | **base rate** | het deel van een groep dat echt positief is, zoals 0,11 van de vrouwen en 0,30 van de mannen in UCI Adult |
        | **equalized odds** | elke groep heeft dezelfde true positive rate en dezelfde false positive rate: de fouten zijn gelijk verdeeld; ook separation genoemd |
        | **equal opportunity** | alleen de true positive rate is in elke groep gelijk |
        | **calibration within groups** | in elke groep is van de mensen met een kans van ongeveer 0,7 ongeveer 70 procent positief: een score betekent in elke groep hetzelfde; bij een ja of nee in elk geval dezelfde precision in elke groep |
        """),
        L.watch_out("Zeg welke klasse positief is. Hier is een ja iets goeds, dus geeft een false positive iemand per vergissing een voordeel. Bij COMPAS was de markering de slechte uitkomst, dus nam een false positive iemand iets af. Barocas, Hardt en Narayanan noemen die omwisseling een bekende bron van verwarring over termen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om de fouten en de betekenis van een model of een regel per groep te controleren:

    1. Zeg wat als positief telt, en of een ja helpt of schaadt.
    2. Neem per groep de true positive rate, het deel van de positieven met een ja, en de false positive rate, het deel van de negatieven met een ja. Allebei gelijk: equalized odds. Alleen de eerste: equal opportunity.
    3. Neem per groep het deel dat positief is per band van scores. Gelijk in elke groep: calibration within groups. Vergelijk bij een ja of nee in elk geval de precision: predictive parity.
    4. Vergelijk de base rates. Verschillen ze en maakt het model fouten, dan kan het niet aan beide criteria voldoen: kies het criterium dat bij de beslissing past, en zeg waarom.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De verzonnen regel voor werving uit [Evenveel ja per groep](/course/open/rai-demographic-parity?lang=nl): een ja voor iedereen met minstens een bachelor, een `education_num` van 13 of meer. `_column` is de kolom `race`. De cel neemt de mensen die meer dan 50.000 dollar verdienen, en rekent per groep uit welk deel van hen de regel een ja geeft: de true positive rate uit stap 2. Voer de cel uit.
    """)
    return


@app.cell
def example_recall(pd):
    # 48.842 mensen uit een Amerikaans onderzoek van 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _column = "race"

    _rich = _people[_people["income"] == ">50K"]
    _tpr = _rich[_rich["education_num"] >= 13].groupby(_column).size() / _rich.groupby(_column).size()
    print(_tpr.round(2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Van de mensen in de groep Asian-Pac-Islander die meer verdienen, geeft de regel 0,67 een ja; in de groep Amer-Indian-Eskimo 0,33. Dat is ver van equal opportunity, al verdienen in de tweede groep maar 55 mensen meer, dus rust dat deel op weinig mensen. Oefening 1 voegt de false positive rate toe, per geslacht.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_errors():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De fouten van de regel per geslacht

        De regel uit het uitgewerkte voorbeeld, nu voor `column`, de kolom `sex`. Vervang de `...` door vier regels: `rich` en `tpr` zoals in het uitgewerkte voorbeeld, met de namen zonder hun liggend streepje, en daarna op dezelfde manier `poor`, de rijen waarin `income` gelijk is aan `"<=50K"`, en `fpr`, het deel daarvan dat de regel een ja geeft. De laatste regel is `tpr, fpr`, zodat de check ze allebei ziet. De check probeert ook een andere kolom.
        """),
        L.task("Reken `tpr` en `fpr` van de regel uit per waarde van `sex`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_errors():
    box_errors = L.code_box("errors", lines=12)
    box_errors
    return (box_errors,)


@app.cell(hide_code=True)
def check_errors(box_errors):
    result_errors = L.check("errors", box_errors.value)
    result_errors
    return (result_errors,)


@app.cell(hide_code=True)
def hints_errors():
    L.hints(
        "errors",
        [
            "De true positive rate kijkt naar de mensen die meer verdienen, de false positive rate naar de mensen die dat niet doen. Deel bij allebei de mensen met een ja per groep door alle mensen van die soort per groep.",
            "`poor = people[people[\"income\"] == \"<=50K\"]`. De regel kiest de rijen met `poor[\"education_num\"] >= 13`.",
            "`fpr = poor[poor[\"education_num\"] >= 13].groupby(column).size() / ___`.",
        ],
        why="`rich = people[people[\"income\"] == \">50K\"]`, `tpr = rich[rich[\"education_num\"] >= 13].groupby(column).size() / rich.groupby(column).size()`, `poor = people[people[\"income\"] == \"<=50K\"]` en `fpr = poor[poor[\"education_num\"] >= 13].groupby(column).size() / poor.groupby(column).size()`. Dat geeft een recall van 0,53 voor vrouwen en 0,49 voor mannen, en een false positive rate van 0,18 en 0,16.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_clash():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Dezelfde precision, dezelfde recall

        In de test set van UCI Adult verdient 0,109 van de vrouwen en 0,300 van de mannen meer dan 50.000 dollar. Stel dat een model voor allebei dezelfde precision heeft, 0,69, en dezelfde recall, 0,57. Welke false positive rate moet elke groep dan hebben? `p` bevat de base rate van elke groep; de loop gaat langs de sleutels, met in `group` de naam van één groep. Vervang de `...` in de loop door de stappen van *Waarom niet allebei*, per 1000 mensen van de groep: `positives`, de mensen die meer verdienen; `true_yes`, het deel `recall` daarvan; `false_yes`, alle ja's (`true_yes / precision`) min de terechte; en `fpr[group]`, de onterechte ja's gedeeld door de mensen die niet meer verdienen. De check probeert ook een andere precision en recall.
        """),
        L.task("Reken `fpr[group]` uit voor elke groep in `p`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_clash():
    box_clash = L.code_box("clash", lines=14)
    box_clash
    return (box_clash,)


@app.cell(hide_code=True)
def check_clash(box_clash):
    result_clash = L.check("clash", box_clash.value)
    result_clash
    return (result_clash,)


@app.cell(hide_code=True)
def hints_clash():
    L.hints(
        "clash",
        [
            "Volg de vier stappen van *Waarom niet allebei* voor 1000 mensen uit één groep. Alleen de base rate verschilt van groep tot groep.",
            "`positives = 1000 * p[group]` en `true_yes = recall * positives`. De mensen die niet meer verdienen, zijn `1000 - positives`.",
            "`false_yes = true_yes / precision - ___`.",
        ],
        why="In de loop: `positives = 1000 * p[group]`, `true_yes = recall * positives`, `false_yes = true_yes / precision - true_yes` en `fpr[group] = false_yes / (1000 - positives)`. Dat geeft 0,03 voor de vrouwen en 0,11 voor de mannen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ### Oefening 3. Wat een bewering vergelijkt

    Zes beweringen over een model of een score. Deel elke bewering in bij wat ze tussen de groepen vergelijkt.
    """)
    return


@app.cell(hide_code=True)
def sort_which_cell():
    sort_which = L.sort("which")
    sort_which
    return (sort_which,)


@app.cell(hide_code=True)
def check_which(sort_which):
    result_which = L.check("which", sort_which)
    result_which
    return (result_which,)


@app.cell(hide_code=True)
def hints_which():
    L.hints(
        "which",
        [
            "Kijk waar elke bewering begint: bij iedereen, bij de mensen die de uitkomst wel of niet hebben, of bij de mensen met een ja of een score.",
            "Bij iedereen: hoe vaak elke groep ja krijgt. Bij de mensen met of zonder de uitkomst: de fouten. Bij de mensen met een ja of een score: wat de score betekent.",
            "De fouten: ProPublica, en \\_\\_\\_. Wat een score betekent: Northpointe, en \\_\\_\\_.",
        ],
        why="**Hoe vaak elke groep ja krijgt:** een threshold per groep, en gesprekken. **De fouten:** ProPublica, en gemist. **Wat een score betekent:** ongeveer 0,7, en Northpointe.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Evenveel ja per groep](/course/open/rai-demographic-parity?lang=nl) telde je de ja's per groep. Hier telde je de vergissingen per groep, voor equalized odds en equal opportunity, en vroeg je of een score in elke groep hetzelfde betekent, voor calibration within groups. Op UCI Adult kwam het model dicht bij gelijke betekenis met ongelijke fouten, en jouw regel op opleiding dicht bij gelijke fouten met ongelijke betekenis. Je tweede oefening liet zien waarom: met verschillende base rates en een model dat fouten maakt, gaat het ene ten koste van het andere, zoals bij COMPAS. Welk criterium past, hangt af van de beslissing en van wie de kosten van een vergissing draagt. De volgende les, *Testen op discriminatie*, gaat over hoe je uitzoekt of beslissingen groepen verschillend behandelen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_clash, result_errors, result_which):
    L.wrap_up(result_errors, result_clash, result_which)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
