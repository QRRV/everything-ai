# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-gbdt-default. Route: course/route/lessen/klassiek-ml.yaml. Checks en plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-gbdt-default", "nl")
    C = L.checks   # de schone tabel met verkeer, de test-MAE van elk aantal bins op de slider, en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Boom na boom op wat overblijft: gradient boosting](/course/open/ml-gradient-boosting?lang=nl) en één over [Vraag voorspellen met bomen](/course/open/ts-forecast-demand?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_trees():
    quick_trees = L.choice(
        "quick-trees",
        question="Op de pinguïns was de validatie-RMSE van gradient boosting het kleinst bij 42 bomen, 415 g, en steeg hij naar 575 g bij 1.000 bomen. Hoe kies je het aantal bomen?",
        options={
            "right": "Waar de fout op de validation set het kleinst is",
            "more": "Zoveel als de computer aankan: elke boom leert meer",
            "train": "Waar de fout op de training set het kleinst is",
        },
        correct="right",
        why={
            "more": "Meer bomen blijven de trainingsfout verlagen, maar na een tijd leren ze de eigenaardigheden van losse trainingsvoorbeelden: op de pinguïns steeg de validatie-RMSE van 415 g naar 575 g. Zie stap 5 van het recept in [Boom na boom op wat overblijft](/course/open/ml-gradient-boosting?lang=nl).",
            "train": "De trainingsfout daalt met elke boom, tot 158 g bij 1.000 bomen, dus die zou altijd de meeste kiezen. Zie stap 5 van het recept in [Boom na boom op wat overblijft](/course/open/ml-gradient-boosting?lang=nl).",
            "right": "Vandaag doet het model dat zelf, terwijl het leert.",
        },
    )
    quick_trees
    return (quick_trees,)


@app.cell(hide_code=True)
def quick_check_calendar():
    quick_calendar = L.choice(
        "quick-calendar",
        question="Een tabel met verkeer, één rij per uur. Welke van deze kolommen is een kalenderfeature?",
        options={
            "right": "`weekday`: de dag van de week van elk uur",
            "lag": "`lag_1`: het verkeer van het uur ervoor",
            "temp": "`temp`: de temperatuur van elk uur",
        },
        correct="right",
        why={
            "lag": "Dat is een lag feature: de reeks zelf, één stap eerder. Een kalenderfeature is wat een datum over zichzelf zegt. Zie *Kalenderfeatures* in *Het idee* in [Vraag voorspellen met bomen](/course/open/ts-forecast-demand?lang=nl).",
            "temp": "De temperatuur wordt in dat uur gemeten; aan de datum alleen zie je die niet. Een kalenderfeature is wat een datum over zichzelf zegt. Zie *Kalenderfeatures* in *Het idee* in [Vraag voorspellen met bomen](/course/open/ts-forecast-demand?lang=nl).",
            "right": "Vandaag zijn het uur, de weekdag en de maand weer features, en maak je zelf nog een kalenderfeature: de feestdagen.",
        },
    )
    quick_calendar
    return (quick_calendar,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Veel data van bedrijven en instellingen is een tabel: bestellingen, klanten, machines, patiënten, één rij per ding, getallen en woorden door elkaar, sommige cellen leeg. Op zulke tabellen probeer je meestal eerst één soort model: gradient-boosted trees. In een studie uit 2022 op 45 tabellen van gemiddelde grootte deden bomen, met gradient boosting erbij, het beter dan de neural networks die geprobeerd werden. Van de 29 winnende inzendingen die de wedstrijdsite Kaggle in 2015 op zijn blog beschreef, gebruikten er 17 XGBoost, een library voor gradient boosting; LightGBM en CatBoost zijn er nog twee. De versie van scikit-learn, `HistGradientBoostingRegressor`, is gebaseerd op LightGBM, en je kwam hem tegen in [Boom na boom op wat overblijft](/course/open/ml-gradient-boosting?lang=nl). Vandaag zie je waarom hij de standaard is: wat hij doet met getallen, met woorden, met lege cellen en met het aantal bomen, zodat zijn standaardinstellingen het al goed doen.

    **Een voorbeeld uit het dagelijks leven.** Een café wil weten wanneer het druk is. Het verdeelt zijn bonnetjes over 24 stapels, één per uur, niet op de precieze minuut: 24 stapels laten alles zien wat ertoe doet. Met 6 stapels van 4 uur belanden de rustige vroege ochtend en het begin van de ochtendspits op één stapel, en kan het café ze niet meer uit elkaar houden.
    """)
    return


@app.cell(hide_code=True)
def predict_boost_cell():
    guess_boost = L.guess(
        "boost-guess",
        question="Verkeer op een snelweg, uur na uur: leer van de uren vóór 2017, test op de 15.246 uren van 2017 en 2018. Een random forest van 100 bomen zit er gemiddeld 274,8 voertuigen per uur naast. Gradient boosting met zijn standaardinstellingen leert van dezelfde tabel. Hoeveel zit het ernaast, en hoe lang duurt het?",
        options={
            "slow": "Ongeveer 275, en het duurt langer",
            "right": "Ongeveer 260, in een twintigste van de tijd",
            "half": "Ongeveer 140, de helft, in dezelfde tijd",
        },
        reveal="**260,3, in ongeveer een twintigste van de tijd.** In een test van de course, op één core, deed het forest er 6,7 seconden over en gradient boosting 0,3. Een lineaire regressie zit er 570,4 naast, en altijd het gemiddelde van de trainingsuren zeggen 1.738,3. Gradient boosting ligt hier niet ver voor op het forest. Wat het de standaard maakt, is dat het dit met zijn standaardinstellingen haalt, en snel.",
    )
    guess_boost
    return (guess_boost,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **De tabel.** Het verkeer op Interstate 94 uit [Project: je tabel, verkend en schoongemaakt](/course/open/ml-project-table-explore?lang=nl), schoongemaakt zoals daar: geen uren op 0 K, niet het uur met 9.831,3 mm regen, en één rij per uur. Dan blijven er 40.564 uren over, van oktober 2012 tot september 2018. Het label is `traffic_volume`, de voertuigen die in dat uur geteld zijn. De features zijn de kalenderfeatures uit [Vraag voorspellen met bomen](/course/open/ts-forecast-demand?lang=nl), het uur, de weekdag en de maand; het weer in getallen, de temperatuur in °C, regen en sneeuw in mm en de bewolking in %; en het weer in één woord, zoals `Clear`, `Rain` of `Snow`, 11 woorden in totaal. De modellen leren van de 25.318 uren vóór 2017 en worden getest op de 15.246 uren van 2017 en 2018: op latere uren, zoals in [Een forecast testen op het verleden](/course/open/ts-backtesting?lang=nl), omdat het model gebruikt gaat worden op uren die later komen.

        **Bins.** Voordat hij één boom laat groeien, verdeelt `HistGradientBoostingRegressor` de waarden van elke feature over hoogstens `max_bins` **bins**, standaard 255: elke bin een stuk van het bereik, met ongeveer evenveel rijen in elk. Een feature met hoogstens zoveel verschillende waarden krijgt een bin per waarde: `hour` heeft er 24. Een boom vraagt dan alleen naar de grenzen tussen bins. Om de beste vraag over een feature te vinden, telt hij de residuen van de rijen in elke bin op, een histogram, en probeert hij alleen de grenzen: veel minder werk dan elke waarde van elke rij proberen. Dat is het *histogram* in **histogram gradient boosting**. Waarden in één bin zien er voor elke boom hetzelfde uit.

        **Lege cellen.** Een ontbrekende waarde krijgt een eigen bin. Bij elke vraag probeert de boom de rijen van die bin aan beide kanten, en stuurt ze naar waar ze het meest helpen. Zo handelt hij ontbrekende waarden zelf af, zoals [Boom na boom op wat overblijft](/course/open/ml-gradient-boosting?lang=nl) zei. De schone tabel met verkeer heeft geen lege cellen; je eigen tabellen wel.

        **Woorden.** Een lineair model heeft getallen nodig. Voor het weer in één woord krijgt elk woord een kolom die True is in de uren met dat woord en False in de andere: de one-hot-vectoren uit [One-hot-vectoren voor klassen](/course/open/ml-one-hot?lang=nl) als kolommen: `pd.get_dummies` maakt ze. Een lineair model heeft hetzelfde nodig voor het uur, de weekdag en de maand, omdat het verkeer niet in een rechte lijn met het uur stijgt: de 8 features worden 58 kolommen. Een boom heeft zulke kolommen voor het uur niet nodig: hij kan vragen of het uur hoogstens 6 is, en daarna of het hoogstens 9 is. En voor woorden heeft `HistGradientBoostingRegressor` ook geen one-hot kolommen nodig. Een kolom van het pandas-type `category`, gemaakt met `astype("category")`, neemt hij zoals hij is, en bij een vraag kan hij elke groep woorden naar één kant sturen, zoals `Rain`, `Snow` en `Thunderstorm` tegen de rest. Dat is **native categorical support**. Op deze tabel maakt het woord weinig uit: met het weer in één woord als type `category` is de test-MAE 260,3, met 11 one-hot kolommen 259,1, en helemaal zonder het woord 259,8. Het weer in getallen en de kalender zeggen al bijna alles.

        **Early stopping.** In [Boom na boom op wat overblijft](/course/open/ml-gradient-boosting?lang=nl) koos je het aantal bomen op de validation set: de fout daar daalt en stijgt dan weer. Het model kan dat zelf doen terwijl het leert: na elke boom meet het de fout op een validation set, en het stopt zodra 10 bomen op rij die niet lieten dalen. Dat is **early stopping**. `early_stopping=True` zet het aan, en `fit(X, y, X_val=..., y_val=...)` geeft het de validation set; `n_iter_` zegt daarna hoeveel bomen het model hield. De fout waar het naar kijkt, is de gekwadrateerde fout waarop het leert: de RMSE op de validation set.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **histogram gradient boosting** | gradient boosting die eerst elke feature over bins verdeelt en alleen naar de grenzen ertussen vraagt; snel op grote tabellen | `HistGradientBoostingRegressor()` |
        | **bin** | een stuk van het bereik van een feature; elke boom behandelt de waarden in één bin hetzelfde | `max_bins=255` |
        | **native categorical support** | het model neemt een kolom met woorden zoals hij is: een vraag kan elke groep woorden naar één kant sturen | `df["weather_main"].astype("category")` |
        | **early stopping** | stoppen met bomen toevoegen zodra de fout op een validation set een aantal bomen op rij niet is gedaald | `early_stopping=True` |
        | `X_val`, `y_val` | de validation set waarop `fit` stopt | `model.fit(X_train, y_train, X_val=X_val, y_val=y_val)` |
        | `n_iter_` | hoeveel bomen het model hield | `model.n_iter_` |
        | `pd.get_dummies` | de tabel met een kolom van True en False voor elke waarde van de genoemde kolommen | `pd.get_dummies(X, columns=["weather"])` |
        | `notna()` | True waar een waarde er is: het omgekeerde van `isna()` | `df["holiday"].notna()` |
        | `index.date` | de datum van elk moment van een datetime-index, zonder de tijd | `df.index.date` |
        """),
        L.watch_out("Met zijn standaard, `early_stopping=\"auto\"`, stopt het model alleen vroeg als het van meer dan 10.000 rijen leert, en dan op een willekeurig gekozen tiende van die rijen, waarvan het niet meer leert. Het uitgewerkte voorbeeld heeft 25.318 trainingsuren, dus daar gebeurt het: het model houdt toch alle 100 bomen die het standaard laat groeien, en leert van negen tiende van de uren. Met `early_stopping=False` leert het van alle uren en zit het er 258,1 naast in plaats van 260,3. Op een tabel in de tijd ligt een willekeurig tiende bovendien tussen de uren waarvan het leert. Kies zelf: `early_stopping=True` met een validation set na de trainingsuren, of `False`."),
        mo.md(r"""
        De slider zet `max_bins`. De grijze staven zijn het gemiddelde verkeer van elk uur van de dag in de trainingsuren. De <span class="c-param">oranje</span> lijnen zijn wat een boom die alleen naar `hour` vraagt uit elkaar kan houden: één waarde per bin, het gemiddelde van zijn uren. De gestippelde lijnen zijn de grenzen tussen bins. Onder het plaatje staat de test-MAE van het model met zoveel bins, vooraf gemeten door de course. Beweeg de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_bins_cell():
    def _frame(bins):
        said = C.bins_said(bins, L.lang)
        alt = ("Een staafdiagram van het gemiddelde verkeer van elk uur van de dag in de trainingsuren, van minder dan 400 "
               "voertuigen 's nachts tot ongeveer 5.600 om 16:00, met een horizontale lijn voor elke bin van het uur op het "
               "gemiddelde van zijn uren. " + said)
        return (C.bins_picture(bins, {"title": "Verkeer per uur van de dag", "hour": "uur van de dag",
                                      "vehicles": "voertuigen per uur", "alt": alt}), said)

    explore_bins = L.explore("max-bins", "max_bins", _frame, start=255, values=C.BINS, role="hyper",
                             task="Ga van 255 bins omlaag naar 12. Vanaf hoeveel bins heeft elk uur een eigen bin, en wat doet de MAE daaronder?")
    explore_bins
    return


@app.cell(hide_code=True)
def text_after_bins():
    mo.md(r"""
    Met 12 bins krijgt `hour` 12 bins, de meeste van twee uur. Uur 4, met gemiddeld 684 voertuigen, deelt een bin met uur 5, met 2.057: het begin van de ochtendspits. Geen boom kan ze uit elkaar houden, en de test-MAE is 395,6. Dat komt door het uur: met alleen `hour` in die 12 groepen geknipt en 255 bins voor de andere features is hij 397,3, ongeveer hetzelfde. Met 16 bins is de MAE 334,4, met 20 bins 304,1. Vanaf 24 bins heeft elk uur een eigen bin, en blijft de MAE tussen 258,9 en 260,6. De standaard van 255 is hier ruim genoeg.

    ### Recept

    Zo begin je aan een tabel:

    1. Maak één rij per ding, met het label en features die bekend zijn als je voorspelt.
    2. Geef kolommen met woorden het type `category` met `astype("category")`, en laat lege cellen leeg.
    3. Zet een test set apart, op volgorde van tijd als de rijen in de tijd liggen, en een validation set om op te stoppen.
    4. Fit `HistGradientBoostingRegressor`, of `HistGradientBoostingClassifier` voor klassen, met zijn standaardinstellingen, `early_stopping=True` en de validation set.
    5. Vergelijk met een baseline, zoals het gemiddelde, en met een eenvoudig model: de bomen moeten ze allebei duidelijk verslaan.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel maakt de schone tabel en de features, knipt op 1 januari 2017, en vergelijkt vier modellen op de testuren: altijd het gemiddelde van de trainingsuren, als baseline; een lineaire regressie op 58 kolommen, met one-hot kolommen voor de kalender en het woord; een random forest van 100 bomen, met one-hot kolommen voor het woord, omdat het forest van scikit-learn geen woorden neemt; en histogram gradient boosting met zijn standaardinstellingen, op de tabel zoals hij is. `threadpool_limits(1)` laat scikit-learn op één core rekenen, zoals in [Vraag voorspellen met bomen](/course/open/ts-forecast-demand?lang=nl). Het forest doet er een paar seconden over: in een test van de course, op één core, 6,7 seconden, tegen 0,3 voor gradient boosting. De cel print het aantal uren, trainingsuren en testuren, het aantal kolommen van de lineaire regressie, en de vier MAE's. Voer de cel uit.
    """)
    return


@app.cell
def example_compare():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from threadpoolctl import threadpool_limits

    # 1. de schone tabel van het project: één rij per uur, de tijd als index
    _df = pd.read_csv("data/metro-interstate-traffic.csv.gz")
    _df = _df[_df["temp"] > 0]
    _df = _df[_df["rain_1h"] < 100]
    _df = _df.drop_duplicates(subset="date_time")
    _df["date_time"] = pd.to_datetime(_df["date_time"])
    _df = _df.set_index("date_time")

    # de features: de kalender, het weer in getallen, het weer in één woord als type category
    _X = pd.DataFrame(index=_df.index)
    _X["hour"] = _df.index.hour
    _X["weekday"] = _df.index.dayofweek
    _X["month"] = _df.index.month
    _X["temp"] = _df["temp"] - 273.15
    _X["rain"] = _df["rain_1h"]
    _X["snow"] = _df["snow_1h"]
    _X["clouds"] = _df["clouds_all"]
    _X["weather"] = _df["weather_main"].astype("category")   # 2. woorden als type category
    _y = _df["traffic_volume"]

    # 3. leer van de uren vóór 2017, test op 2017 en 2018
    _train = _X.index < "2017-01-01"
    _test = _X.index >= "2017-01-01"
    print(len(_X), _train.sum(), _test.sum())

    # 5. een baseline: altijd het gemiddelde van de trainingsuren
    _mean = np.mean(abs(_y[_test] - _y[_train].mean()))

    # een lineaire regressie heeft getallen nodig: one-hot kolommen voor de kalender en het woord
    _D = pd.get_dummies(_X, columns=["hour", "weekday", "month", "weather"])
    print(_D.shape[1])
    _linear = LinearRegression().fit(_D[_train], _y[_train])
    _lin = np.mean(abs(_y[_test] - _linear.predict(_D[_test])))

    with threadpool_limits(1):
        # het forest van scikit-learn neemt geen woorden: one-hot kolommen voor het woord
        _F = pd.get_dummies(_X, columns=["weather"])
        _forest = RandomForestRegressor(random_state=0).fit(_F[_train], _y[_train])
        _rf = np.mean(abs(_y[_test] - _forest.predict(_F[_test])))
        # 4. gradient boosting neemt de tabel zoals hij is
        _boost = HistGradientBoostingRegressor(random_state=0).fit(_X[_train], _y[_train])
        _gb = np.mean(abs(_y[_test] - _boost.predict(_X[_test])))
    print(round(_mean, 1), round(_lin, 1), round(_rf, 1), round(_gb, 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De tabel heeft 40.564 uren: 25.318 om van te leren en 15.246 om op te testen. De lineaire regressie krijgt 58 kolommen: de 4 getallen van het weer, 24 voor het uur, 7 voor de weekdag, 12 voor de maand en 11 voor het woord. Altijd het gemiddelde zit er 1.738,3 voertuigen per uur naast, de lineaire regressie 570,4, het forest 274,8 en gradient boosting 260,3. Zelfs met een kolom voor elk uur telt de lineaire regressie één getal voor het uur en één voor de weekdag op, dus het verschil tussen een maandag en een zondag komt op elk uur hetzelfde uit. In de trainingsuren is dat niet zo: om 7:00 heeft een maandag gemiddeld 4.655 voertuigen meer dan een zondag, om 2:00 heeft hij er 343 minder. Een boom vraagt naar allebei, de een na de ander.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_whole_day():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een feestdag voor de hele dag

        De kolom `holiday` heeft de naam van een feestdag, zoals `Independence Day`, maar alleen in het eerste uur van die dag, zoals je in het project zag; elk ander uur heeft NaN. Een model dat die kolom krijgt, ziet een feestdag alleen om middernacht. `whole_day(df)` moet een lijst geven met een markering voor elke rij van `df`: 1 als de dag van de rij een feestdag is, dus als een van de rijen van die dag een naam heeft, anders 0. `days = df.index.date` geeft de datum van elke rij. Vervang de twee `...`. De check roept `whole_day(traffic)` aan, met `traffic` de schone tabel met zijn kolom `holiday`, en probeert vier verzonnen tabellen, een daarvan met ontbrekende uren.
        """),
        L.task("Laat `whole_day` elk uur markeren van een dag die in een van zijn uren de naam van een feestdag heeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_whole_day():
    box_whole_day = L.code_box("whole-day", lines=9)
    box_whole_day
    return (box_whole_day,)


@app.cell(hide_code=True)
def check_whole_day(box_whole_day):
    result_whole_day = L.check("whole-day", box_whole_day.value)
    result_whole_day
    return (result_whole_day,)


@app.cell(hide_code=True)
def hints_whole_day():
    L.hints(
        "whole-day",
        [
            "Twee stappen: eerst de datums die in een van hun uren een naam hebben, dan voor elke rij of zijn datum daar een van is.",
            "De uren met een naam zijn waar `holiday` niet ontbreekt: het omgekeerde van `isna()`. Een array met een mask tussen de blokhaken houdt de waarden waar de mask True is, zoals in [Delen van een array kiezen](/course/open/np-indexing?lang=nl), en `set` houdt elke datum één keer. `in` vraagt of een waarde in een set zit.",
            "`holidays = set(days[df[\"holiday\"].___()])` en `if day ___ holidays:`.",
        ],
        why="De 53 namen worden 1.203 gemarkeerde uren. Met die kolom daalt de test-MAE van 260,3 naar 235,8 voertuigen per uur, en op de 18 feestdagen van 2017 en 2018 van 913,5 naar 507,3: het model ziet een feestdag nu in elk uur ervan, niet alleen om middernacht.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_stop_on():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Stoppen op 2016

        `stop_on(X_train, y_train, X_val, y_val)` moet `HistGradientBoostingRegressor` fitten met hoogstens 300 bomen en early stopping op de validation set, en teruggeven hoeveel bomen het hield. Vervang de twee `...`. De check roept `stop_on(X_train, y_train, X_val, y_val)` aan: de features uit het uitgewerkte voorbeeld met de kolom `holiday` uit oefening 1, de 17.481 uren vóór 2016 om van te leren en de 7.837 uren van 2016 om op te stoppen. Hij probeert ook twee verzonnen tabellen van 2.000 en 5.000 rijen.
        """),
        L.task("Laat `stop_on` vroeg stoppen op de validation set en het aantal bomen geven dat het hield."),
    ])
    return


@app.cell(hide_code=True)
def exercise_stop_on():
    box_stop_on = L.code_box("stop-on", lines=7)
    box_stop_on
    return (box_stop_on,)


@app.cell(hide_code=True)
def check_stop_on(box_stop_on):
    result_stop_on = L.check("stop-on", box_stop_on.value)
    result_stop_on
    return (result_stop_on,)


@app.cell(hide_code=True)
def hints_stop_on():
    L.hints(
        "stop-on",
        [
            "Twee dingen: zet early stopping aan in het model, en geef `fit` de validation set naast de rijen waarvan het leert.",
            "`early_stopping` neemt `True`, `False` of `\"auto\"`, en in scikit-learn 1.9 zet `\"auto\"` het pas aan boven 10.000 rijen, ook als `X_val` gegeven is. `fit` neemt de validation set als twee keyword arguments na de features en het label.",
            "`early_stopping=___` en `model.fit(X_train, y_train, X_val=___, y_val=___)`.",
        ],
        why="Het model stopt bij 50 bomen: de fout op 2016 was het laagst na 40, en de 10 bomen daarna brachten hem niet lager. Op 2017 en 2018 zitten de 50 bomen er 245,7 voertuigen per uur naast, en alle 300 bomen 243,8: ongeveer even goed, met een zesde van de bomen. Allebei liggen boven de 235,8 uit oefening 1, omdat deze modellen alleen leerden van de uren vóór 2016.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_first():
    mo.md(r"""
    ### Oefening 3. Het eerste model op een nieuwe tabel
    """)
    return


@app.cell(hide_code=True)
def choice_first_cell():
    choice_first = L.choice(
        "first-model",
        question="Een nieuwe tabel: 50.000 leden van een sportschool, 30 kolommen met getallen en woorden, sommige cellen leeg, en de vraag wie volgende maand opzegt. Welk model probeer je eerst?",
        options={
            "right": "`HistGradientBoostingClassifier` met zijn standaardinstellingen, de woorden als `category`, early stopping op een validation set",
            "linear": "Een logistische regressie, na one-hot kolommen voor elk woord en een waarde in elke lege cel",
            "tune": "Gradient boosting, maar pas na een lange zoektocht door al zijn instellingen",
        },
        correct="right",
        why={
            "linear": "Dat kan werken, en het is een goed model om mee te vergelijken. Maar het heeft dat werk eerst nodig, en het telt alleen effecten op: op het verkeer zat een lineaire regressie op 58 kolommen er 570,4 voertuigen per uur naast, gradient boosting op de tabel zoals hij was 260,3.",
            "tune": "De standaardinstellingen doen het al goed: op het verkeer versloeg gradient boosting het forest zonder dat er één instelling veranderde. Zoek de instellingen pas als je een eerste model hebt om te verslaan; *Instellingen zoeken: grid search en random search* laat zien hoe.",
            "right": "Het eerste model om te proberen op een tabel; `HistGradientBoostingClassifier` is hetzelfde model voor klassen. Vergelijk het met een baseline, zoals altijd de klasse die het vaakst voorkomt, voordat je zijn score vertrouwt.",
        },
    )
    choice_first
    return (choice_first,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De bomen zijn die uit [Boom na boom op wat overblijft](/course/open/ml-gradient-boosting?lang=nl); early stopping kiest hun aantal op een validation set, zoals jij daar met de hand deed, en in *Early stopping* doe je hetzelfde voor een neural network, epoch na epoch. De kalenderfeatures komen uit [Vraag voorspellen met bomen](/course/open/ts-forecast-demand?lang=nl), en de schone tabel uit [Project: je tabel, verkend en schoongemaakt](/course/open/ml-project-table-explore?lang=nl). In de volgende les, *Wat wedstrijden leren: het leaderboard*, zie je wat er gebeurt als je steeds op hetzelfde deel van de data blijft kiezen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_first, result_stop_on, result_whole_day):
    L.wrap_up(result_whole_day, result_stop_on, choice_first)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
