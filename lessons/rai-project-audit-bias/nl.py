# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-project-audit-bias. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-project-audit-bias", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Fairness-tools en benchmarks](/course/open/rai-fairness-benchmarks?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_metricframe():
    quick_metricframe = L.choice(
        "quick-metricframe",
        question="`MetricFrame` van Fairlearn met `selection_rate` geeft in `by_group`: Female 0,8, Male 0,4. Wat is die 0,8?",
        options={
            "share": "Van alle ja's, het deel voor vrouwen",
            "right": "Het deel van de vrouwen dat een ja krijgt",
            "accuracy": "Het deel van de vrouwen waarbij het model goed zit",
        },
        correct="right",
        why={
            "share": "Dan zou je door alle ja's delen. `selection_rate` deelt door de mensen van de groep: van de vrouwen krijgt 0,8 een ja. Zie oefening 1 in [Fairness-tools en benchmarks](/course/open/rai-fairness-benchmarks?lang=nl).",
            "right": "`selection_rate` is het deel met een ja, per groep uitgerekend: van de vrouwen krijgt 0,8 een ja. De tabel die je in deze les maakt begint ermee.",
            "accuracy": "Hoe vaak het model goed zit is de accuracy, een andere maat. `selection_rate` telt alleen de ja's: van de vrouwen krijgt 0,8 een ja. Zie *Het idee* in [Fairness-tools en benchmarks](/course/open/rai-fairness-benchmarks?lang=nl).",
        },
    )
    quick_metricframe
    return (quick_metricframe,)


@app.cell(hide_code=True)
def quick_check_split():
    quick_split = L.choice(
        "quick-split",
        question="Op de hele tabel UCI Adult bracht reweighing het deel met een ja van vrouwen gedeeld door dat van mannen van 0,28 naar 0,57. Wat lieten tien steekproeven van 1000 mensen zien?",
        options={
            "same": "Ongeveer dezelfde toename van 0,29 in elke steekproef",
            "right": "De toename verschilde sterk per steekproef",
            "none": "Reweighing verlaagde de verhouding in de meeste",
        },
        correct="right",
        why={
            "same": "De toename liep van ongeveer 0,07 tot ongeveer 0,51: met 300 mensen om te testen verschuift één vrouw meer of minder met een ja de verhouding met ongeveer 0,05. Zie de demo in [Fairness-tools en benchmarks](/course/open/rai-fairness-benchmarks?lang=nl).",
            "right": "Van ongeveer 0,07 tot ongeveer 0,51. Eén splitsing van een kleine tabel zegt weinig, en ook je audit meet op één splitsing: schrijf op op hoeveel mensen elk getal rust.",
            "none": "Reweighing bracht de verhouding in alle tien de steekproeven omhoog; hoeveel verschilde, van ongeveer 0,07 tot ongeveer 0,51. Zie de demo in [Fairness-tools en benchmarks](/course/open/rai-fairness-benchmarks?lang=nl).",
        },
    )
    quick_split
    return (quick_split,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In de vorige lessen mat je telkens één ding: een deel met een ja, een foutpercentage, een kalibratie, een threshold per groep. Een audit zet ze voor één model bij elkaar, in een vaste volgorde. In dit project doe je een audit op je eigen classifier, die van de vorige lessen op UCI Adult, zoals iemand uit het team dat hem bouwde dat zou doen: eerst wat je gaat meten en voor welke groepen, dan de getallen per groep, dan een eerste aanpassing, opnieuw gemeten, en als laatste wat de getallen wel en niet kunnen zeggen. De functies die je schrijft blijven in je project. Verderop in de course legt *Project: leg je classifier uit* dezelfde classifier uit en gebruikt ze opnieuw.

    **Een voorbeeld uit het dagelijks leven.** Een inspecteur komt de keuken van een restaurant controleren met een vaste lijst: de temperatuur van de koelkast, of rauw en bereid voedsel apart blijven, of het personeel de handen wast. De lijst is voor elk restaurant gelijk en ligt vast voor het bezoek, dus wat er gecontroleerd wordt, kiest de inspecteur niet pas na het zien van de keuken. Aan het eind komt een rapport: wat in orde was, wat niet, en wat voor het volgende bezoek moet veranderen. En een schoon rapport betekent dat de keuken op die dag door die lijst kwam, niet dat er nooit iemand ziek zal worden.
    """)
    return


@app.cell(hide_code=True)
def predict_white():
    guess_white = L.guess(
        "white",
        question="De classifier van de vorige lessen geeft ongeveer 20 op de 100 witte mensen in de test set een ja, en ongeveer 7 op de 100 zwarte mensen. Markeert de 80%-regel ook witte mensen, denk je?",
        options={
            "no": "Nee: witte mensen krijgen veel vaker een ja dan zwarte mensen",
            "right": "Ja, want één groep krijgt nog vaker een ja",
            "small": "Alleen als ze minder dan de helft zijn",
        },
        reveal="**Ja.** De mensen met `Asian-Pac-Islander` in de tabel krijgen ongeveer 26 keer op de 100 een ja. Preciezer: 20,26 tegen 25,83, en 20,26 gedeeld door 25,83 is 0,78, net onder 0,8. De regel vergelijkt elke groep met de groep die het vaakst een ja krijgt, welke groep dat ook is: hier een groep van maar 480 mensen. De demo hieronder laat alle vijf de groepen zien.",
    )
    guess_white
    return (guess_white,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Wat een audit is.** Het woord wordt op twee manieren gebruikt. Van buitenaf is een audit volgens Danaë Metaxa en collega's een methode om een algoritme herhaald en stelselmatig inputs te geven en te kijken wat eruit komt, om zo iets te leren over hoe het van binnen werkt, terwijl je daar niet in kunt kijken. Van binnenuit nemen Inioluwa Deborah Raji en collega's een definitie over uit audits van software: een onafhankelijke beoordeling of software en de manier waarop die gemaakt wordt voldoen aan de wetten, normen, richtlijnen, plannen en eisen die ervoor gelden. In deze course is een **algoritme-audit** een controle van een algoritmisch systeem aan criteria zoals fairness, accuracy of rechtmatigheid, door te kijken naar zijn data, zijn code of wat het doet.

    **Wie de audit doet.** Sasha Costanza-Chock en collega's onderscheiden drie soorten: een first-party audit doet het team zelf, een second-party audit een opdrachtnemer, en een third-party audit iemand zonder contract met het bedrijf. Jij bouwde de classifier en hebt zijn data, dus jouw audit is een first-party audit. Je ziet wat een buitenstaander niet ziet, zoals de data om op te trainen en de code. Maar, waarschuwen Raji en collega's, wie intern een audit doet, deelt noodzakelijk een belang met wat er gecontroleerd wordt.

    **Wat erin hoort.** Raji en collega's verdelen een audit binnen een bedrijf in vijf fasen: afbakenen, in kaart brengen, stukken verzamelen zoals datasheets, testen, en terugkijken. Elke fase levert documenten op, die samen het auditrapport vormen. Afbakenen komt eerst: waar is het systeem voor, en wie kan het schaden? De getallen komen later, en ze mogen volgens hen geen doel op zich worden. Voor de getallen raden Margaret Mitchell en collega's, die model cards voorstelden om modellen te beschrijven, aan om uitkomsten uit te splitsen naar de gekozen factoren, zoals geslacht en afkomst. Een rapport hoort alle foutpercentages te geven en te zeggen welke voorrang kreeg en waarom, en de thresholds van de beslissing te noemen en waarom ze gekozen zijn.

    **Wat het wel en niet kan laten zien.** Een audit meet uitkomsten per groep: wie een ja krijgt, en wie welke fouten draagt. Metaxa en collega's schrijven dat de uitkomsten van een audit alleen te zien zijn voor het systeem als geheel; waarom de uitkomsten verschillen, laat hij uit zichzelf niet zien. Een audit kan een probleem vinden, maar niet bewijzen dat er geen is: een schone uitkomst geldt voor de groepen, de criteria, de data en de thresholds die gemeten zijn. En hoe dicht bij gelijk dicht genoeg is, moet de audit zelf beslissen. Raji en collega's laten die vraag open: hoe dicht bij gelijk moet een classifier zitten om aanvaardbaar te zijn? Voor het deel met een ja gaf [De 80%-regel](/course/open/rai-80-percent-rule?lang=nl) één antwoord, uit Amerikaanse regels voor werving. Er is nog geen vaste standaard: Costanza-Chock en collega's vroegen 43 auditors om een link naar hoe ze een audit doen, en maar zeven gaven een link naar enige documentatie, en vier naar uitkomsten.

    **Je classifier.** Toen je deze les opende, zette de course een bestand in je project: `work/projects/audit/model.py`. Daarin staat de classifier van de vorige lessen: logistische regressie op leeftijd, opleiding, werkuren, soort werk, burgerlijke staat, beroep, gezinsrol en afkomst, zonder geslacht. Zijn functie `scores(people)` traint hem op de mensen om op te trainen en geeft elke persoon de kans op een ja. Lezen wat erin gaat hoort bij de audit: de classifier krijgt geen geslacht, maar wel `relationship`, met de waarden `Husband` en `Wife`, een proxy voor geslacht, en hij krijgt afkomst zelf. Het is jouw bestand. Verander gerust zijn lijsten `NUMBERS` en `WORDS`: aan het eind van de les doet je project de audit op de classifier die in dat bestand staat.

    **Probeer het.** De cel hieronder traint dezelfde classifier, geeft een ja boven 0,5, en maakt de eerste tabel van een audit: per afkomst in de test set hoeveel mensen, hoe vaak een ja, en dat gedeeld door het hoogste deel met een ja, de impact ratio uit [De 80%-regel](/course/open/rai-80-percent-rule?lang=nl). De vijf waarden van `race` zijn de eigen woorden van de tabel uit 1994. Sommige, zoals `Amer-Indian-Eskimo`, worden nu niet meer gebruikt; de les houdt ze, zodat je ze in de data terugvindt. Voer de cel uit; dat duurt een seconde of twee.
    """)
    return


@app.cell
def demo_race():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression

    # 48.842 mensen uit een Amerikaanse enquête van 1994 (UCI Adult)
    _people = pd.read_csv("data/adult.csv.gz")
    _train = _people["split"] == "train"

    # de classifier van de vorige lessen, zonder de kolom sex, zoals in work/projects/audit/model.py
    _words = ["workclass", "marital_status", "occupation", "relationship", "race"]
    _columns = ["age", "education_num", "hours_per_week"] + _words
    _X = pd.get_dummies(_people[_columns], columns=_words, dtype=float)
    _X = _X / _X.max()
    _model = LogisticRegression(max_iter=1000)
    _model.fit(_X[_train], _people["income"][_train] == ">50K")
    _people["yes"] = _model.predict_proba(_X)[:, 1] > 0.5

    # de eerste tabel van de audit: per afkomst in de test set
    _test = _people[_people["split"] == "test"]
    _table = pd.DataFrame()
    _table["people"] = _test.groupby("race").size()
    _table["yes rate"] = _test.groupby("race")["yes"].mean()
    _table["impact ratio"] = _table["yes rate"] / _table["yes rate"].max()
    print(_table.round(3))
    return (pd,)


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.picture_race({"title": "Deel met een ja per afkomst, op de test set",
                                "xlabel": "deel met een ja", "line": "0,8 × hoogste", "decimal": ",",
                                "key": "Een staaf per afkomst, met tussen haakjes het aantal mensen en aan het eind van de staaf de impact ratio. Een staaf die voor de stippellijn ophoudt zit onder de 80%-regel.",
                                "alt": "Deel met een ja per afkomst: Asian-Pac-Islander (480 mensen) 0,26, impact ratio 1; White (13.946) 0,20, 0,78; Black (1561) 0,07, 0,27; Other (135) 0,06, 0,23; Amer-Indian-Eskimo (159) 0,04, 0,17. De stippellijn bij 0,8 keer het hoogste ligt bij 0,21."})),
        mo.md(r"""
        Vier van de vijf groepen krijgen minder dan 80 procent zo vaak een ja als de groep die het vaakst een ja krijgt, `Asian-Pac-Islander`, en daaronder is `White`, met 0,78. `Black` krijgt 0,27 van dat deel, en `Amer-Indian-Eskimo` 0,17. Twee dingen horen in het rapport. Ten eerste legt één groep de lat, hier een groep van 480 mensen: een paar ja's meer of minder in die groep verschuiven elke verhouding. Ten tweede tellen de groepen van 135 tot 13.946 mensen. Een deel met een ja over 135 mensen kan met andere mensen makkelijk een paar honderdsten anders uitvallen, zoals [Fairness-tools en benchmarks](/course/open/rai-fairness-benchmarks?lang=nl) liet zien voor steekproeven van 1000. Model cards vragen daarom om een marge van onzekerheid; geef op zijn minst op hoeveel mensen elk getal rust.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **algoritme-audit** | een controle van een algoritmisch systeem aan criteria zoals fairness, accuracy of rechtmatigheid, door te kijken naar zijn data, zijn code of wat het doet |
        | **first-, second- en third-party audit** | een audit door het team dat het systeem bouwde, door een opdrachtnemer die het inhuurde, of door iemand zonder contract ermee |
        | **uitgesplitst** | per groep gegeven, zoals per geslacht of per afkomst, in plaats van één getal voor iedereen |
        """),
        L.watch_out("Jouw audit is een first-party audit: je controleert je eigen werk, en je wilt dat het slaagt. Beslis wat je gaat meten, en welke getallen een probleem zijn, voordat je naar de uitkomsten kijkt, en rapporteer elk getal, ook de getallen die slecht uitvallen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Het eerste deel van je audit:

    1. Baken af: wat beslist de classifier, en wie kan een onterecht ja of een onterecht nee schaden? Schrijf de groepen en de criteria op voordat je meet: hier geslacht en afkomst; het deel met een ja met de 80%-regel, de recall, de false positive rate, en calibration within groups.
    2. Lees wat erin gaat: welke kolommen, en welke daarvan een protected attribute zijn of een proxy ervoor.
    3. Meet op de test set, per groep: hoeveel mensen, het deel met een ja, de impact ratio, de recall en de false positive rate. Dat is `report`.
    4. Vergelijk per band van scores per groep de gemiddelde score met het deel dat meer verdient. Dat is `calibration`.
    5. Probeer een eerste aanpassing, zoals een threshold per groep met dezelfde recall, gekozen op de mensen om op te trainen. Dat is `threshold_for`. Meet daarna alles opnieuw, voor elke groep, niet alleen voor de groep die je aanpaste.
    6. Schrijf op wat de getallen laten zien, op hoeveel mensen ze rusten, en wat de audit niet heeft gemeten.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Acht verzonnen mensen in twee groepen, a en b, met voor elk of ze meer verdienen, `rich`, 1 of 0, en het antwoord van de classifier, `yes`. `_rich` bevat de rijen van de mensen die meer verdienen, en `_poor` de rijen van de anderen. `_table` begint leeg. Zijn eerste kolom is de recall per groep: van de mensen die meer verdienen het deel met een ja, dat `groupby` en `.mean()` van `yes` in één keer geven. Zijn tweede kolom is de false positive rate: hetzelfde voor `_poor`.
    """)
    return


@app.cell
def example_errors(pd):
    # acht verzonnen mensen: hun groep, rich (1 als ze meer verdienen) en het antwoord van de classifier
    _people = pd.DataFrame({"group": ["a", "a", "a", "a", "b", "b", "b", "b"],
                            "rich": [1, 1, 0, 0, 1, 1, 1, 0],
                            "yes": [True, False, True, False, True, True, False, False]})
    _rich = _people[_people["rich"] == 1]
    _poor = _people[_people["rich"] == 0]

    _table = pd.DataFrame()
    _table["recall"] = _rich.groupby("group")["yes"].mean()
    _table["false positive rate"] = _poor.groupby("group")["yes"].mean()
    print(_table)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In groep a krijgt een van de twee mensen die meer verdienen een ja, een recall van 0,5, en een van de twee anderen krijgt er ook een, een false positive rate van 0,5. In groep b krijgen twee van de drie die meer verdienen een ja, 0,67, en de ene persoon die dat niet doet krijgt een nee, 0. `groupby` geeft elke groep een eigen rij, op naam gesorteerd, en elke nieuwe kolom van `_table` komt per groep op de goede plek. Oefening 1 maakt hier een functie in je project van, samen met het deel met een ja en de impact ratio uit de demo.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_report():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. report: de tabel van je audit

        `report(people, column)` krijgt een tabel zoals de test set van de demo en de naam van de kolom om op te groeperen. De tabel heeft een kolom `rich`, 1 voor de mensen die meer verdienen en 0 voor de anderen, en een kolom `yes` met het antwoord van de classifier, True of False. De functie heeft al `rich`, `poor` en `table`, waarvan de eerste kolom `people` de mensen per groep telt. Vervang de `...` door vier regels, elk een kolom van `table`: `yes rate`, het deel van de groep met een ja; `impact ratio`, het deel met een ja gedeeld door het hoogste deel met een ja, zoals in de demo; en `recall` en `false positive rate`, zoals in het uitgewerkte voorbeeld. Schrijf `column` overal waar de naam van de kolom hoort. De check roept `report(PEOPLE, "sex")` aan, met `PEOPLE` de test set met de classifier van de demo, daarna `report(PEOPLE, "race")` en een kleine tabel van zichzelf. Als hij klopt, bewaart de check hem in je project, in `work/projects/audit/report.py`.
        """),
        L.task("Zet de kolommen `yes rate`, `impact ratio`, `recall` en `false positive rate` in `table`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_report():
    box_report = L.code_box("report", lines=12)
    box_report
    return (box_report,)


@app.cell(hide_code=True)
def check_report(box_report):
    result_report = L.check("report", box_report.value)
    result_report
    return (result_report,)


@app.cell(hide_code=True)
def hints_report():
    L.hints(
        "report",
        [
            "De demo maakte `yes rate` en `impact ratio` voor de kolom `race`, en het uitgewerkte voorbeeld maakte `recall` en `false positive rate` voor `group`. Hier krijgt elke `groupby` de naam `column`.",
            "`table[\"yes rate\"] = people.groupby(column)[\"yes\"].mean()`. Voor de recall groepeer je `rich` in plaats van `people`, en voor de false positive rate `poor`.",
            "`table[\"impact ratio\"] = table[\"yes rate\"] / ___`.",
        ],
        why="`table[\"yes rate\"] = people.groupby(column)[\"yes\"].mean()`, `table[\"impact ratio\"] = table[\"yes rate\"] / table[\"yes rate\"].max()`, `table[\"recall\"] = rich.groupby(column)[\"yes\"].mean()` en `table[\"false positive rate\"] = poor.groupby(column)[\"yes\"].mean()`. Voor geslacht krijgen de vrouwen een deel met een ja van 0,07 en een impact ratio van 0,28.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_calibration():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. calibration: betekent een score voor elke groep hetzelfde?

        Een score voldoet aan calibration within groups als hij in elke groep ongeveer zo vaak uitkomt als hij zegt, zoals in [Gelijke fouten of gelijke betekenis](/course/open/rai-equalized-odds-calibration?lang=nl). `calibration(people, column, low, high)` kijkt naar één band van scores: de mensen met een score van `low` of meer en onder `high`. `band` bevat al de rijen met een score van `low` of meer. Vervang de eerste `...` door een regel die de rijen van `band` houdt waarvan de score onder `high` ligt. Vervang de tweede door twee kolommen van `table`: `mean score`, het gemiddelde van `score` per groep in de band, en `share rich`, het gemiddelde van `rich` per groep in de band, dat het deel is dat meer verdient. De check roept `calibration(PEOPLE, "sex", 0.5, 0.6)` aan, daarna een andere kolom en band, en een kleine tabel met scores precies op de randen van een band. Hij bewaart je functie in `work/projects/audit/calibration.py`.
        """),
        L.task("Houd de mensen met een score onder `high`, en zet `mean score` en `share rich` per groep erbij."),
    ])
    return


@app.cell(hide_code=True)
def exercise_calibration():
    box_calibration = L.code_box("calibration", lines=11)
    box_calibration
    return (box_calibration,)


@app.cell(hide_code=True)
def check_calibration(box_calibration):
    result_calibration = L.check("calibration", box_calibration.value)
    result_calibration
    return (result_calibration,)


@app.cell(hide_code=True)
def hints_calibration():
    L.hints(
        "calibration",
        [
            "Stap 4 van het recept. Een band heeft twee randen: `band` heeft de onderste al, en de volgende regel neemt de rijen van `band` met een score onder `high`, in de vorm `band = band[...]`.",
            "Houd de rijen met `band[\"score\"] < high`. Daarna geeft `band.groupby(column)[...].mean()`, zoals `report` het voor `people` deed, het gemiddelde van één kolom per groep: van de scores voor `\"mean score\"`, en van `rich` voor `\"share rich\"`.",
            "`table[\"share rich\"] = band.groupby(column)[___].mean()`.",
        ],
        why="`band = band[band[\"score\"] < high]`, `table[\"mean score\"] = band.groupby(column)[\"score\"].mean()` en `table[\"share rich\"] = band.groupby(column)[\"rich\"].mean()`. In de band van 0,5 tot 0,6 is de gemiddelde score van de vrouwen 0,55 en verdient 0,48 van hen meer; voor de mannen 0,55 en 0,54.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_threshold():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. threshold_for: een eerste aanpassing

        Je project gaat een threshold per geslacht zoeken met dezelfde recall, zoals in [De beslissing aanpassen: een threshold per groep](/course/open/rai-fair-thresholds?lang=nl). `threshold_for(people, recall)` krijgt de mensen van één groep en een recall om te halen. De functie moet de hoogste van de thresholds 0,01, 0,02, tot 0,99 geven waarbij minstens het deel `recall` van de mensen die meer verdienen een ja krijgt, dat wil zeggen een score boven de threshold heeft. `rich` bevat die mensen, `best` begint bij 0, en de loop loopt met `t` van 1 tot 99, zodat de threshold `t / 100` is. Vervang de `...` door een `if` die vraagt of het deel van `rich` met een score boven `t / 100` minstens `recall` is, en daaronder een regel die `t / 100` in `best` bewaart. Een hogere threshold die ook slaagt vervangt hem later. De check roept `threshold_for(WOMEN, 0.55)` aan, met `WOMEN` de vrouwen om op te trainen, daarna de mannen, en een kleine tabel waarin de recall precies gehaald wordt. Hij bewaart je functie in `work/projects/audit/fix.py`.
        """),
        L.task("Zoek de hoogste threshold die de recall van de groep op `recall` of meer houdt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_threshold():
    box_threshold = L.code_box("threshold", lines=8)
    box_threshold
    return (box_threshold,)


@app.cell(hide_code=True)
def check_threshold(box_threshold):
    result_threshold = L.check("threshold", box_threshold.value)
    result_threshold
    return (result_threshold,)


@app.cell(hide_code=True)
def hints_threshold():
    L.hints(
        "threshold",
        [
            "De recall bij een threshold is het deel van de mensen die meer verdienen met een score erboven. Van een Series van True en False is `.mean()` het deel True.",
            "`(rich[\"score\"] > t / 100).mean()` is de recall bij de threshold `t / 100`. Vergelijk die in een `if` met `recall`, en bewaar de threshold in `best`.",
            "`if (rich[\"score\"] > t / 100).mean() ___ recall:`",
        ],
        why="`if (rich[\"score\"] > t / 100).mean() >= recall:` en daaronder `best = t / 100`. De recall stijgt nooit als de threshold stijgt, dus de laatste threshold die slaagt is de hoogste: 0,38 voor de vrouwen en 0,52 voor de mannen.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel doet je audit op je eigen classifier: `scores` uit je `model.py` op de hele tabel, daarna je `report` per geslacht en per afkomst op de test set, je `calibration` per geslacht in tien banden van 0,1, en je `threshold_for` voor de vrouwen en de mannen om op te trainen, bij een recall van 0,55. Met die thresholds beslist hij opnieuw over de test set en roept hij je `report` nog een keer aan. Dat duurt een paar seconden, en hij draait opnieuw na elke oefening.
    """)
    return


@app.cell(hide_code=True)
def demo_project(result_calibration, result_report, result_threshold):
    _r = (result_report, result_calibration, result_threshold)   # zodat deze cel na elke check draait
    _audit = C.demo()
    if _audit is None:
        _out = mo.md("*Maak eerst oefening 1: dan staat je `report` in je project, en gebruikt deze cel hem.*")
    elif isinstance(_audit, str):
        _out = mo.md(C.broken(_audit, L.lang))
    else:
        _parts = C.audit_parts(_audit, L.lang)
        _out = mo.vstack([mo.md(_p) if _kind == "md" else mo.Html(_p) for _kind, _p in _parts])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_claim():
    mo.md(r"""
    ### Oefening 4. Eén zin in het rapport
    """)
    return


@app.cell(hide_code=True)
def choice_claim_cell():
    choice_claim = L.choice(
        "claim",
        question="Een collega schrijft in je auditrapport: \"De audit laat zien dat de classifier met een threshold per geslacht fair is.\" Wat is er mis met die zin?",
        options={
            "nothing": "Niets: vrouwen en mannen hebben nu dezelfde recall",
            "right": "Een audit kan een probleem vinden, niet bewijzen dat er geen is",
            "third": "Alleen een onafhankelijke auditor mag een classifier fair noemen",
            "accuracy": "De zin moet zeggen hoe vaak de classifier het goed heeft",
        },
        correct="right",
        why={
            "nothing": "Gelijke recall is één criterium, op één test set, bij één paar thresholds. De impact ratio van de vrouwen ligt nog ver onder 0,8, en de groepen naar afkomst veranderden nauwelijks. En wat de audit niet heeft gemeten, kan de audit niet garanderen.",
            "right": "De audit mat deze groepen, met deze criteria, op deze test set en met deze thresholds. De audit vond verschillen waar gekeken werd; een schone uitkomst zou niets zeggen over waar niet gekeken werd. Daarom zegt het rapport wat er wel en niet gemeten is.",
            "third": "Het maakt uit wie de audit doet: die van jou is een first-party audit, en je deelt het belang van wie bouwde. Maar ook een third-party auditor kan niet bewijzen dat een classifier fair is. Het probleem zit in de bewering, niet in wie de bewering doet.",
            "accuracy": "Het rapport hoort te zeggen hoe vaak de classifier het goed heeft, ook per groep. Maar dat maakt de zin niet goed: vaak goed zitten zegt niets over wie de fouten draagt, en geen getal bewijst dat er geen probleem is.",
        },
    )
    choice_claim
    return (choice_claim,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [De 80%-regel](/course/open/rai-80-percent-rule?lang=nl) gaf je de impact ratio, [Gelijke fouten of gelijke betekenis](/course/open/rai-equalized-odds-calibration?lang=nl) de foutpercentages en calibration within groups, en [De beslissing aanpassen: een threshold per groep](/course/open/rai-fair-thresholds?lang=nl) de eerste aanpassing. Hier werden ze één audit van één classifier, in de volgorde die een audit nodig heeft: beslissen wat je meet, per groep meten, aanpassen, opnieuw meten, en opschrijven wat de getallen niet laten zien. [Fairness-tools en benchmarks](/course/open/rai-fairness-benchmarks?lang=nl) liet zien waarom het aantal mensen achter elk getal in het rapport hoort. Je project `audit` heeft nu `model.py`, `report.py`, `calibration.py` en `fix.py`. Het komt terug in *Project: leg je classifier uit*, waar je dezelfde classifier uitlegt en opschrijft wat de uitleg laat zien over de bias die je hier mat. De volgende les, *Bias in language models*, zoekt bias waar geen ja of nee te tellen is: in wat een language model schrijft.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_claim, result_calibration, result_report, result_threshold):
    L.wrap_up(result_report, result_calibration, result_threshold, choice_claim)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
