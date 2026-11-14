# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-project-table-trees. Route: course/route/lessen/klassiek-ml.yaml. Checks, de folds, het speelblok en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-project-table-trees", "nl")
    C = L.checks   # de schone verkeerstabel, de vijf folds, wat elk model mist, de plaatjes en de demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een forecast testen op het verleden](/course/open/ts-backtesting?lang=nl) en [Project: welk model is het best? Een eerlijke vergelijking](/course/open/ml-project-fair-comparison?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_origin():
    quick_origin = L.choice(
        "quick-origin",
        question="Je backtest een forecast van het aantal fietsers per dag met een rolling origin. Bij de origin maandag 2 juli 2012 voorspelt die de week erna. Van welke dagen mag de forecast leren?",
        options={
            "week": "Van elke dag behalve de week die voorspeld wordt",
            "right": "Alleen van de dagen vóór 2 juli 2012",
            "all": "Van elke dag: er wordt toch op de week erna getest",
        },
        correct="right",
        why={
            "week": "De dagen na die week waren op 2 juli ook nog niet bekend. Een forecast mag alleen gebruiken wat bij de origin bekend was: de dagen ervoor. Zie [Een forecast testen op het verleden](/course/open/ts-backtesting?lang=nl).",
            "all": "Dan leert de forecast precies de week waarop getest wordt, en zegt de score niets over een echte forecast. Alleen de dagen vóór de origin. Zie [Een forecast testen op het verleden](/course/open/ts-backtesting?lang=nl).",
            "right": "Alleen het verleden. Vandaag liggen de origins op 1 januari van elk jaar, en leert een model van elk uur daarvoor.",
        },
    )
    quick_origin
    return (quick_origin,)


@app.cell(hide_code=True)
def quick_check_split():
    quick_split = L.choice(
        "quick-split",
        question="Op de pinguïns haalde logistische regressie een accuracy van 0,91 op één willekeurige splitsing, en k-NN 0,89 op een andere willekeurige splitsing. Welk model is beter?",
        options={
            "logistic": "Logistische regressie: de accuracy is 0,02 hoger",
            "equal": "Ze zijn even goed: 0,02 is te klein om ertoe te doen",
            "right": "Dat weet je nog niet: vergelijk ze op dezelfde folds",
        },
        correct="right",
        why={
            "logistic": "Elk model is op een eigen splitsing gescoord, en een splitsing met makkelijkere pinguïns laat een model beter lijken. Op dezelfde folds lagen de beste vier modellen ongeveer één pinguïn per fold uit elkaar. Zie [Project: welk model is het best? Een eerlijke vergelijking](/course/open/ml-project-fair-comparison?lang=nl).",
            "equal": "Misschien, maar dat kunnen deze twee getallen niet zeggen: ze komen van verschillende pinguïns. Dezelfde folds, fold na fold, kunnen het wel. Zie [Project: welk model is het best? Een eerlijke vergelijking](/course/open/ml-project-fair-comparison?lang=nl).",
            "right": "Dezelfde folds voor elk model, en de verschillen fold na fold. Vandaag doe je dat voor modellen van het verkeer, met folds in de tijd.",
        },
    )
    quick_split
    return (quick_split,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je schone verkeerstabel krijgt modellen. In [De standaard voor tabellen: gradient-boosted trees](/course/open/ml-gbdt-default?lang=nl) versloeg histogram gradient boosting een random forest met 14,5 voertuigen per uur. Dat was één splitsing, op 1 januari 2017: één testperiode. Zou boosting ook in een ander jaar winnen? En verslaat een van beide modellen een eenvoudige forecast die de week kent: het verkeer van hetzelfde uur een week eerder? Een eerlijke test beantwoordt allebei. Vandaag krijgt je project zijn drie onderdelen: een baseline, folds in de tijd, en één functie die elk model op dezelfde uren scoort, fold na fold. Aan het eind schrijf je op wat wint, en met hoeveel.

        **Een voorbeeld uit het dagelijks leven.** Je wilt weten of een nieuwe route naar je werk sneller is. Je probeert de nieuwe route op een vrijdag in augustus, en die scheelt 6 minuten; maar op een vrijdag in augustus is het sowieso rustig op de weg. Een eerlijke test neemt de oude en de nieuwe route op dezelfde soort dagen, week na week, en schrijft voor elke week op welke sneller was, en met hoeveel minuten. Eén vrijdag kan je dat niet vertellen; tien weken wel.
        """),
        draw.project_steps(["Je tabel, verkend en schoongemaakt", "Bomen op je tabel, eerlijk getest"], 1,
                           here_label="← vandaag", title="Waar je project staat: bomen op je tabel"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_years():
    guess_years = L.guess(
        "years",
        question="Herhaal de splitsing van *De standaard voor tabellen* één keer voor elk jaar van 2014 tot en met 2018: leer van de jaren ervoor, test op het jaar zelf. In hoeveel van de vijf jaren verslaat boosting het forest?",
        options={
            "few": "In één of twee",
            "most": "In drie of vier",
            "right": "In alle vijf",
        },
        reveal="**In alle vijf**, maar met heel verschillende marges: met 3,7 voertuigen per uur in 2015, met 25,2 in 2018. Eén splitsing had je één van die getallen laten zien.",
    )
    guess_years
    return (guess_years,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **De vraag.** Hoeveel verkeer komt er in een uur, gegeven de kalender en het weer van dat uur? Een model leert van de uren vóór een moment, de origin, en wordt getest op de uren erna. Het krijgt het weer van elk uur zoals het gemeten is. Een echte forecast een week vooruit zou alleen een weersverwachting hebben, en die is minder precies; de scores van deze les zijn dus iets beter dan een echte forecast zou halen.

        **Folds in de tijd.** In [Een forecast testen op het verleden](/course/open/ts-backtesting?lang=nl) schoof de origin steeds een week op. Hier schuift die steeds een jaar op: een origin op 1 januari van elk jaar van 2014 tot en met 2018, een model dat leert van elk uur daarvoor, en een test op de uren van dat jaar. Dat zijn vijf folds, en de uren om van te leren groeien van 9.397 naar 34.031. `X.index.year` geeft het jaar van elk uur, dus `X.index.year < 2016` is de mask van de uren vóór 2016. Het jaar 2015 heeft pas rijen vanaf 11 juni, na het gat dat je in de eerste stap van het project vond.

        **Een baseline voor uren.** De seasonal naive forecast zegt dat elk uur vooruit lijkt op hetzelfde uur één seizoen eerder. Het seizoen van het verkeer is een week: 168 uur. In de schone tabel zelf is 168 rijen terug niet altijd een week terug, want 11.987 uren hebben geen rij: de 11.976 uit [Tijd als index: datums en resampling in pandas](/course/open/ts-time-index?lang=nl), en de 11 onmogelijke uren die je `clean` weglaat. Op `y.asfreq("h")` heeft elk uur een rij, dus geeft `shift(168)` hetzelfde uur een week eerder, met NaN waar dat uur geen rij heeft: bij 3.066 van de 40.564 uren. `before[y.index]` neemt daarna weer de waarden bij de uren van de tabel. De week ervoor kent het verkeer van een week geleden, en de modellen kennen alleen de jaren vóór hun origin: een model dat de week ervoor toch verslaat, is de moeite waard.

        **Dezelfde uren voor elk model.** Een model kun je op elk testuur scoren; de week ervoor alleen waar het uur een week eerder een rij heeft. Voor een eerlijke vergelijking scoor je elk model op dezelfde uren: de testuren waar de week ervoor bekend is, `test & base.notna()`. In 2016 zijn dat er 7.144 van de 7.837 testuren.

        **Fold na fold.** Net als bij de pinguïns verbergt het gemiddelde van vijf folds wat er in elk ervan gebeurt. Schrijf voor elk paar modellen op in hoeveel folds het ene het andere verslaat en met hoeveel, en kijk goed naar een fold die tegen de andere ingaat.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | `.year` | het jaar van elk moment in een datetime-index, zoals `.hour` het uur geeft | `X.index.year < 2016` |
        | `&` | voor twee masks van dezelfde lengte: True waar ze allebei True zijn; een vergelijking ernaast zet je tussen haakjes | `test & base.notna()`, `test & (X.index.hour == 8)` |
        | `s[y.index]` | de waarden van de Series `s` bij de momenten in `y.index` | `before[y.index]` |
        """),
        L.watch_out("Willekeurige folds, zoals bij de pinguïns, zien er hier beter uit dan ze zijn: uren van dezelfde weken komen aan beide kanten terecht, en een model leert van uren na de uren waarop het getest wordt. Op dezelfde 29.283 uren van 2014 tot en met 2018 zit boosting er met vijf willekeurige folds 251,0 voertuigen per uur naast, en met de vijf jaarfolds 284,8. De jaarfolds komen dichter bij wat een model kan in een jaar dat het nooit gezien heeft."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Schuif de origin door de jaren. Links: elke maand van de tabel, met de uren die een rij hebben; gevuld, de maanden waar de modellen van leren; open, het jaar waarop ze getest worden; grijs, de maanden daarna. Rechts: wat de week ervoor, het forest en boosting missen op dezelfde uren van dat jaar.
    """)
    return


@app.cell(hide_code=True)
def explore_origin_cell():
    def _frame(year):
        year = int(year)
        _said = C.said(year, L.lang)
        _alt = ("Links: de uren met een rij in elke maand van de tabel, de maanden om van te leren gevuld, het jaar om op te "
                "testen open, de latere maanden grijs. Rechts: de MAE van de week ervoor, het forest en boosting op dezelfde "
                "uren. " + _said.replace("**", ""))
        return C.picture(year, C.play_words(L.lang) | {"alt": _alt}, L.lang), _said

    explore_origin = L.explore("origin", "testjaar", _frame, start=2017, values=C.YEARS, role="hyper",
                               task="Zoek het jaar waarin het forest meer mist dan de week ervoor. Met hoeveel verslaat boosting de week ervoor in dat jaar?")
    explore_origin
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Boosting verslaat het forest in alle vijf de jaren, met 3,7 tot 25,2 voertuigen per uur, en de week ervoor in vier ervan met 38,8 tot 93,5. In 2016 verslaat het de week ervoor met maar 1,5 voertuigen per uur, een gelijkspel, terwijl het forest 22,3 meer mist dan de week ervoor. Van juli tot en met november 2016 was het verkeer van maandag tot en met vrijdag, van 6.00 tot en met 19.00 uur, 387 tot 457 voertuigen per uur lager dan in dezelfde maanden van 2015 en 2017. De modellen leerden alleen van de jaren ervoor, dus bleven ze het oude verkeer verwachten: in juli 2016 schatte boosting gemiddeld 502,2 voertuigen per uur te veel. De week ervoor volgt de verandering een week later. Eén splitsing op 1 januari 2017 had je dat jaar nooit laten zien.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om modellen van een tabel in de tijd te vergelijken:

    1. Kies de folds in de tijd: een origin op 1 januari van elk jaar, leer van elk uur daarvoor, test op de uren van dat jaar.
    2. Maak een baseline die alleen gebruikt wat bekend was: hetzelfde uur één seizoen eerder, op het grid van elk uur, `y.asfreq("h").shift(...)`.
    3. Scoor elk model op dezelfde uren: de testuren waar de baseline bekend is.
    4. Fit in elke fold elk model op de uren om van te leren, en scoor het op de uren van stap 3.
    5. Schrijf per fold op wat wint, en met hoeveel, niet alleen het gemiddelde; en zoek waarom een fold tegen de andere ingaat.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel maakt de schone tabel en de features van [De standaard voor tabellen: gradient-boosted trees](/course/open/ml-gbdt-default?lang=nl), legt één origin op 1 januari 2017, net als daar, en vergelijkt boosting met een baseline: hetzelfde uur een *dag* eerder. Ze scoort allebei op de testuren die een rij een dag eerder hebben. Ze print de uren om van te leren, de testuren en de uren waarop allebei gescoord worden; de twee MAE's; en wat de dag ervoor mist op elke dag van de week, maandag eerst. `threadpool_limits(1)` laat scikit-learn op één core rekenen, zoals in *De standaard voor tabellen*. Voer de cel uit.
    """)
    return


@app.cell
def example_compare():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits

    # de schone tabel van het project, met de tijd als index
    _df = pd.read_csv("data/metro-interstate-traffic.csv.gz")
    _df = _df[_df["temp"] > 0]
    _df = _df[_df["rain_1h"] <= 100]
    _df = _df.drop_duplicates(subset="date_time")
    _df["date_time"] = pd.to_datetime(_df["date_time"])
    _df["temp"] = _df["temp"] - 273.15
    _df = _df.set_index("date_time")

    # de features van De standaard voor tabellen, en het label
    _X = pd.DataFrame(index=_df.index)
    _X["hour"] = _df.index.hour
    _X["weekday"] = _df.index.dayofweek
    _X["month"] = _df.index.month
    _X["temp"] = _df["temp"]
    _X["rain"] = _df["rain_1h"]
    _X["snow"] = _df["snow_1h"]
    _X["clouds"] = _df["clouds_all"]
    _X["weather"] = _df["weather_main"].astype("category")
    _y = _df["traffic_volume"]

    # 1. één origin: leer van de uren vóór 2017, test op de uren vanaf dan
    _train = _X.index < "2017-01-01"
    _test = _X.index >= "2017-01-01"

    # 2. een baseline: hetzelfde uur een dag eerder, op het grid van elk uur
    _every = _y.asfreq("h")
    _day = _every.shift(24)[_y.index]

    # 3. dezelfde uren voor allebei: de testuren met een rij een dag eerder
    _known = _test & _day.notna()
    print(_train.sum(), _test.sum(), _known.sum())

    # 4. fit op de uren om van te leren, scoor allebei op dezelfde uren
    with threadpool_limits(1):
        _boost = HistGradientBoostingRegressor(random_state=0).fit(_X[_train], _y[_train])
    print(round(np.mean(abs(_y[_known] - _day[_known])), 1), round(np.mean(abs(_y[_known] - _boost.predict(_X[_known]))), 1))

    # 5. de dag ervoor, dag na dag van de week: maandag is 0
    _by_day = []
    for _d in range(7):
        _on = _known & (_X.index.dayofweek == _d)
        _by_day.append(float(round(np.mean(abs(_y[_on] - _day[_on])), 1)))
    print(_by_day)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    25.318 uren om van te leren, 15.246 om op te testen, en 15.183 daarvan hebben een rij een dag eerder. Daarop zit de dag ervoor er 558,6 voertuigen per uur naast, boosting 260,3. Per dag van de week mist de dag ervoor 1.155,2 op maandagen en 1.058,3 op zaterdagen: een maandag lijkt niet op de zondag ervoor, en een zaterdag niet op de vrijdag. Van dinsdag tot en met vrijdag mist de dag ervoor 256,6 tot 375,9. Het seizoen van het verkeer is de week. Een baseline die makkelijk te verslaan is, zegt weinig; daarom is de baseline van je project hetzelfde uur een week eerder.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_week():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De week ervoor

        `week_before(y)` moet voor elk uur van de Series `y` het verkeer van hetzelfde uur een week eerder geven, en NaN waar dat uur geen rij heeft. Vervang de twee `...`: `every` met een rij voor elk uur van het eerste tot het laatste, en `before` met elke waarde een week later gezet. De laatste regel neemt weer de waarden bij de uren van `y`. De check roept `week_before(traffic)` aan, met het verkeer van de schone tabel en de tijd als index, en probeert drie verzonnen Series van uren. Als je functie klopt, bewaart de check die in je project, in `work/projects/tabular/baseline.py`.
        """),
        L.task("Geef elk uur een rij, en zet dan elke waarde een week later."),
    ])
    return


@app.cell(hide_code=True)
def exercise_week():
    box_week = L.code_box("week-before", lines=5)
    box_week
    return (box_week,)


@app.cell(hide_code=True)
def check_week(box_week):
    result_week = L.check("week-before", box_week.value)
    result_week
    return (result_week,)


@app.cell(hide_code=True)
def hints_week():
    L.hints(
        "week-before",
        [
            "Stap 2 van het recept; stap 2 van het uitgewerkte voorbeeld deed het voor een dag. Wat verandert er voor een week?",
            "`asfreq` geeft elk uur een rij, zoals in *Tijd als index*; op dat grid telt `shift` uren, en een week heeft 7 dagen van 24 uur.",
            "`every = y.asfreq(___)` en `before = every.shift(___)`.",
        ],
        why="Op het grid van elk uur is een rij een uur, dus als je de waarden 168 rijen later zet, staat naast elk uur het verkeer van hetzelfde uur een week eerder, en NaN waar dat uur geen rij had. `before[y.index]` houdt alleen de uren die in de tabel staan.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_folds():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Folds in de tijd

        `year_folds(X, years)` moet een lijst geven met één fold voor elk jaar in `years`: een paar masks, `train` voor de uren vóór 1 januari van dat jaar en `test` voor de uren van het jaar. Vervang de twee `...`. De check roept `year_folds(X, [2014, 2015, 2016, 2017, 2018])` aan, met de features van de schone tabel, en probeert een kleine verzonnen tabel met andere jaren, waaronder een jaar zonder uren ervoor en een jaar helemaal zonder uren. Ze bewaart je functie in `work/projects/tabular/folds.py`.
        """),
        L.task("Leer van elk uur vóór het jaar; test op de uren van het jaar."),
    ])
    return


@app.cell(hide_code=True)
def exercise_folds():
    box_folds = L.code_box("folds", lines=7)
    box_folds
    return (box_folds,)


@app.cell(hide_code=True)
def check_folds(box_folds):
    result_folds = L.check("folds", box_folds.value)
    result_folds
    return (result_folds,)


@app.cell(hide_code=True)
def hints_folds():
    L.hints(
        "folds",
        [
            "Stap 1 van het recept. Stap 1 van het uitgewerkte voorbeeld maakte één origin door `_X.index` met een moment te vergelijken: wat vergelijk je om de uren vóór een jaar te krijgen, en de uren van dat jaar?",
            "`X.index.year` geeft het jaar van elk uur; vergelijk dat met `year`.",
            "`train = X.index.year ___ year` en `test = X.index.year ___ year`.",
        ],
        why="Elke fold leert alleen van wat bij de origin bekend was, elk uur vóór 1 januari, en wordt getest op het jaar dat volgt: vijf origins, vijf tests, en nooit een test op een uur waar het model van leerde.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_scores():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Elk model op dezelfde uren

        `fold_scores(model, X, y, years)` haalt je twee functies uit je project: `week_before` voor de baseline en `year_folds` voor de folds. In elke fold moet die het model fitten op de uren om van te leren, en de MAE geven op de testuren die een rij een week eerder hebben, de uren waarop de week ervoor gescoord wordt. Vervang de twee `...`. De check roept `fold_scores(boosting, X, traffic, [2014, 2015, 2016, 2017, 2018])` aan, met als `boosting` histogram gradient boosting met hoogstens 30 bomen, zodat de check een paar seconden duurt; ze probeert ook twee andere modellen, één ervan op een kleine verzonnen tabel. Waar een module van jou ontbreekt of kapot is, springt de versie van de course in, zoals de melding bij *Waar je project staat* zegt. Als je functie klopt, bewaart de check die in `work/projects/tabular/scores.py`.
        """),
        L.task("In elke fold: fit op de uren vóór het jaar, en scoor op de testuren waar de week ervoor bekend is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_scores():
    box_scores = L.code_box("scores", lines=15)
    box_scores
    return (box_scores,)


@app.cell(hide_code=True)
def check_scores(box_scores):
    result_scores = L.check("scores", box_scores.value)
    result_scores
    return (result_scores,)


@app.cell(hide_code=True)
def hints_scores():
    L.hints(
        "scores",
        [
            "Stap 3 en 4 van het recept, zoals in stap 3 en 4 van het uitgewerkte voorbeeld, nu binnen de loop. Van welke uren leert het model, en op welke uren wordt het gescoord?",
            "Twee masks, één voor de testuren en één voor de uren waar `base` geen NaN is, gecombineerd met `&`. En welk mask van de fold bevat de uren om van te leren?",
            "`known = test & ___` en `model.fit(X[___], y[___])`.",
        ],
        why="Het model van elke fold ziet alleen de uren vóór zijn jaar, en elk model, ook de week ervoor, wordt op precies dezelfde uren gescoord. Een verschil tussen twee scores is dan een verschil tussen de modellen, niet tussen de uren waarop ze gescoord zijn.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel neemt je project zoals het is: je `clean` op de hele tabel, de features van *De standaard voor tabellen*, en je `fold_scores` voor boosting met de standaardinstellingen en voor een random forest van 100 bomen, met one-hot kolommen voor het woord, in de vijf folds; je `week_before` en `year_folds` geven de week ervoor op dezelfde uren. Er leren tien modellen, en de forests duren even. Druk op de knop.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    demo_button = mo.ui.run_button(label="Vergelijk op je project")
    demo_button
    return (demo_button,)


@app.cell(hide_code=True)
def demo_cell(demo_button, result_scores):
    _r = result_scores   # zodat deze cel opnieuw draait na elke check van oefening 3
    if not demo_button.value:
        _out = mo.md("*Druk op de knop om de modellen met je project te vergelijken.*")
    else:
        _result = C.demo()
        if _result is None:
            _out = mo.md("*Maak eerst oefening 3 af: dan staat je `fold_scores` in je project, en gebruikt deze cel die.*")
        elif isinstance(_result, str):
            _out = mo.md("*Je project stopt met een fout, of geeft niet voor elke fold een getal. Heb je een van zijn bestanden in `work/projects/tabular/` veranderd? Verwijder het, en check de oefening ervan opnieuw: de check bewaart je code dan opnieuw.*")
        else:
            _who = "" if not _result["stand_ins"] else "De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de melding bij *Waar je project staat* zegt. "
            _same = all(round(a, 1) == b for _k, _i in (("week", 0), ("forest", 1), ("boost", 2))
                        for a, b in zip(_result[_k], [C.MAE[_y][_i] for _y in C.YEARS]))
            _out = mo.vstack([
                mo.Html(C.report_chart(_result, L.lang)),
                mo.md(C.report(_result, L.lang)),
                mo.md(_who + ("Dit zijn de getallen van de slider: je project maakt de eerlijke vergelijking nu zelf." if _same else "")),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_report():
    mo.md(r"""
    ### Oefening 4. Wat je opschrijft
    """)
    return


@app.cell(hide_code=True)
def choice_report_cell():
    choice_report = L.choice(
        "report",
        question="Je verslag voor 2016 zegt: de week ervoor zit er 344,2 voertuigen per uur naast, het forest 366,5, boosting 342,7. Maand na maand mist boosting minder dan de week ervoor in 6 van de 12 maanden. Wat schrijf je op voor 2016?",
        options={
            "won": "Boosting won 2016, met 1,5 voertuigen per uur, en het forest werd laatste",
            "right": "Gelijkspel met de week ervoor, en het forest verloor: van juli tot en met november was het verkeer lager",
            "drop": "Laat 2016 weg uit het verslag: het verkeer was dat jaar ongewoon, dus het zegt niets",
        },
        correct="right",
        why={
            "won": "1,5 voertuigen per uur, op uren waarop allebei er ongeveer 340 naast zitten, is een verschil dat niemand zou merken, en maand na maand gaat het twee kanten op, 6 maanden elk. Een verslag dat *gewonnen* zegt, belooft meer dan de getallen laten zien: schrijf *gelijkspel*, met de getallen.",
            "drop": "Dan verbergt het verslag het ene jaar waarin een model dat van het verleden leerde verrast werd, en dat kan weer gebeuren. Een fold die slecht gaat, is een bevinding: schrijf die op, met wat je weet over waarom.",
            "right": "De getallen, en wat ze betekenen: in 2016 was boosting niet beter dan de vorige week opzoeken, omdat het verkeer dat jaar veranderde. In de vier andere jaren versloeg boosting allebei.",
        },
    )
    choice_report
    return (choice_report,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De folds zijn de rolling origin van [Een forecast testen op het verleden](/course/open/ts-backtesting?lang=nl), met een jaar als stap en een model dat bij elke origin leert; dezelfde uren voor elk model, een baseline en de verschillen fold na fold komen uit [Project: welk model is het best? Een eerlijke vergelijking](/course/open/ml-project-fair-comparison?lang=nl). Het model dat je als eerste probeert, uit [De standaard voor tabellen: gradient-boosted trees](/course/open/ml-gbdt-default?lang=nl), verslaat het forest in elk jaar, maar met 3,7 tot 25,2 voertuigen per uur, en één jaar laat zien dat zelfs een goed model verrast kan worden door een verandering. Je project `tabular` heeft nu zes modules: `impossible.py`, `hours.py` en `clean.py` uit [Project: je tabel, verkend en schoongemaakt](/course/open/ml-project-table-explore?lang=nl), en `baseline.py`, `folds.py` en `scores.py`. Met deze les heb je de basis van klassieke machine learning af.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_report, result_folds, result_scores, result_week):
    L.wrap_up(result_week, result_folds, result_scores, choice_report)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
