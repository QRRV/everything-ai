# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ts-forecast-demand. Route: course/route/lessen/klassiek-ml.yaml. Checks en afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ts-forecast-demand", "nl")
    C = L.checks   # de fietsers van elk uur, de boom een week vooruit gebacktest voor elke stap van de slider, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Een forecast testen op het verleden](/course/open/ts-backtesting?lang=nl) en één over [Boom na boom op wat overblijft: gradient boosting](/course/open/ml-gradient-boosting?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_origin():
    quick_origin = L.choice(
        "quick-origin",
        question="Een backtest legt zijn origin op maandag 4 juni 2012 en voorspelt de week erna. Op welke dagen mag de forecast trainen?",
        options={
            "right": "Op elke dag voor 4 juni",
            "around": "Op elke dag van 2011 en 2012 behalve die week",
            "week": "Alleen op de week voor 4 juni",
        },
        correct="right",
        why={
            "around": "De dagen na die week zijn ook de toekomst: een forecast die op 4 juni gemaakt wordt, kan ze niet kennen. Zie *Alleen het verleden* in *Het idee* in [Een forecast testen op het verleden](/course/open/ts-backtesting?lang=nl).",
            "week": "De seasonal naive forecast gebruikt alleen de laatste week, maar een forecast mag trainen op alles wat voor de origin kwam. Zie *Alleen het verleden* in *Het idee* in [Een forecast testen op het verleden](/course/open/ts-backtesting?lang=nl).",
            "right": "Alleen wat bij de origin bekend was. Vandaag geldt dat voor elke kolom van een tabel: die mag alleen bevatten wat bij de origin bekend was.",
        },
    )
    quick_origin
    return (quick_origin,)


@app.cell(hide_code=True)
def quick_check_missing():
    quick_missing = L.choice(
        "quick-missing",
        question="Sommige rijen van een tabel hebben NaN in een feature. Wat doet `HistGradientBoostingRegressor` daarmee?",
        options={
            "right": "Hij gaat zelf met de ontbrekende waarden om",
            "error": "Hij stopt met een fout, dus laat die rijen eerst vallen",
            "zero": "Hij telt ze als 0",
        },
        correct="right",
        why={
            "error": "Dit model niet: [Boom na boom op wat overblijft](/course/open/ml-gradient-boosting?lang=nl) zegt dat het zelf met ontbrekende waarden omgaat.",
            "zero": "Een 0 zou een waarde zijn als elke andere, en een verkeerde. [Boom na boom op wat overblijft](/course/open/ml-gradient-boosting?lang=nl) zegt dat dit model zelf met ontbrekende waarden omgaat.",
            "right": "Vandaag doet dat ertoe: de eerste twee weken hebben geen waarde van twee weken eerder, en 165 uren hebben helemaal geen rij. Een ontbrekend label is een ander verhaal, zoals je zult zien.",
        },
    )
    quick_missing
    return (quick_missing,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Hoeveel fietsen, maaltijden, megawatts of verpleegkundigen morgen en volgende week: forecasts van de vraag plannen voorraad, personeel en stroom. De vraag hangt van meer af dan van zijn eigen verleden: van het uur, de dag van de week, een feestdag, het weer. De forecasts van de vorige les herhalen alleen het verleden. Een model als gradient boosting kan van dat alles tegelijk leren, zodra het verleden in de kolommen van een tabel staat. Maar een tabel maakt het ook makkelijk om vals te spelen zonder het te merken: één kolom met een waarde van na de origin, en de backtest ziet er veel beter uit dan de forecast ooit zal zijn.

    **Een voorbeeld uit het dagelijks leven.** Een bakkerij plant op woensdag het brood voor zaterdag. Ze weet wat ze vorige zaterdag en de zaterdag daarvoor verkocht, dat het zaterdag markt is, en dat maandag een feestdag is. Wat ze donderdag en vrijdag verkoopt, hoe handig dat ook zou zijn, weet ze nog niet.
    """)
    return


@app.cell(hide_code=True)
def predict_tree_cell():
    guess_tree = L.guess(
        "tree-guess",
        question="Voorspel een week vooruit, uur voor uur, vanaf elke maandag van 2 januari tot 24 december 2012. De seasonal naive forecast, hetzelfde uur een week eerder, zit er gemiddeld 46,8 fietsers per uur naast. Een boom leert van de fietsers één en twee weken eerder, het uur, de weekdag, de maand en of een dag een werkdag is. Hoeveel zit hij ernaast?",
        options={
            "worse": "Meer dan 46,8",
            "right": "Ongeveer 42: een tiende minder",
            "half": "Ongeveer 23: de helft",
        },
        reveal="**42,0, een tiende minder.** Maar het gewone gemiddelde van hetzelfde uur één en twee weken eerder, zonder enig model, haalt 42,1. Over het hele jaar voegt de boom bijna niets toe aan die twee weken. Hij verdient zijn plek op de dagen die een eenvoudige regel niet ziet aankomen, zoals feestdagen, en met kolommen die een eenvoudige regel niet kan gebruiken, zoals het weer.",
    )
    guess_tree
    return (guess_tree,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Van een reeks naar een tabel.** Een boom leert van een tabel: rijen met features en een label. Voor een forecast wordt elk uur een rij, de fietsers zijn het label, en de features zijn wat er vooraf over dat uur bekend is.

        **Lag features.** De fietsers van dezelfde reeks `k` uur eerder vormen een **lag feature**, `lag_k`. Op het rooster van uren van `asfreq("h")` schuift `y.shift(k)` elke waarde `k` rijen naar achteren, dus krijgt elke rij het uur `k` ervoor, zoals in [Tijd als index](/course/open/ts-time-index?lang=nl). De autocorrelatie uit [Lijkt vandaag op gisteren?](/course/open/ts-autocorrelation?lang=nl) zegt welke lags een kolom waard zijn: 0,886 bij lag 168, een week.

        **Alleen lags die de horizon toestaat.** Een forecast vanaf maandag 0:00 voor de week erna, `h` = 168 uur, moet `lag_k` voor elk uur van die week invullen met wat bij de origin bekend is. Het laatste uur, zondag 23:00, ligt 167 uur na de origin; `k` uur daarvoor ligt alleen voor de origin als `k` minstens 168 is. Een lag moet dus minstens de horizon zijn: `k >= h`. Een kortere lag, zoals `lag_24`, is voor maandag bekend, maar vanaf dinsdag bevat hij uren van de week zelf, de uren die de forecast moet voorspellen.

        **Kalenderfeatures.** Wat een datum over zichzelf zegt, is zo ver vooruit bekend als je wilt: het uur, `index.hour`, van 0 tot 23; de weekdag, `index.dayofweek`, 0 voor maandag zoals in [Trend, seizoen en de rest](/course/open/ts-trend-season?lang=nl); de maand, `index.month`, van 1 tot 12; en of een dag een werkdag is, wat de tabel van de course als `workingday` heeft: 0 in het weekend en op feestdagen. Feestdagen zijn jaren vooruit bekend. Dit zijn **kalenderfeatures**.

        **Een boom, getest zoals eerder.** `HistGradientBoostingRegressor` uit [Boom na boom op wat overblijft](/course/open/ml-gradient-boosting?lang=nl) leert van de tabel: bij elke origin van de rijen ervoor waarvan de fietsers bekend zijn. Een rij zonder fietsers heeft geen label om van te leren, dus die gaat eruit: `dropna(subset=["registered"])` laat alleen de rijen met NaN in die kolom vallen. Met een ontbrekende lag gaat het model zelf om. De backtest is die uit [Een forecast testen op het verleden](/course/open/ts-backtesting?lang=nl): 52 origins, elke maandag van 2 januari tot 24 december 2012, elk een week vooruit, gescoord met het gemiddelde van hun MAE's.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **lag feature** | een kolom met de waarde van de reeks `k` stappen eerder | `y.shift(k)` |
        | **kalenderfeature** | een kolom met wat een datum over zichzelf zegt, vooraf bekend | `y.index.dayofweek` |
        | `hour`, `month` | het uur, 0 tot 23, en de maand, 1 tot 12, van elk moment in een datetime index | `y.index.hour`, `y.index.month` |
        | `dropna(subset=[...])` | de tabel zonder de rijen met NaN in de genoemde kolommen; NaN elders blijft staan | `t.dropna(subset=["registered"])` |
        | `threadpool_limits(1)` | binnen zijn `with`-blok rekent scikit-learn op één kern | `with threadpool_limits(1):` |
        """),
        L.watch_out("Een tabel kan vooruitkijken zonder enige fout. Zet `lag_1` en `lag_24` in de tabel van een forecast een week vooruit, en de backtest zegt 24,4 fietsers per uur, ongeveer de helft van de 46,8 van de seasonal naive forecast. Dat getal klopt niet: vanaf het tweede uur van de week is `lag_1` een uur van de week zelf, en vanaf dinsdag `lag_24` ook. Een forecast die maandag 0:00 gemaakt wordt, heeft geen van beide. Vraag van elke kolom: was hij bij de origin bekend, voor elk uur van de horizon?"),
        mo.md(r"""
        De slider voegt stap voor stap kolommen toe. Stap 1: alleen de kalender. Stap 2: alleen `lag_168`. Stap 3: `lag_168` en `lag_336`. Stap 4: beide lags en de kalender. Stap 5: ook het werkelijke weer, temperatuur, soort weer en luchtvochtigheid, wat een forecast van een week eerder niet kan weten: het laat zien wat een perfecte weersverwachting waard zou zijn. Links de week vanaf maandag 4 juni 2012: de fietsers als grijze lijn, de seasonal naive forecast gestreept, de boom van de stap dik. Rechts de MAE van elke stap over de 52 weken van 2012, die de course vooraf heeft gemeten met de backtest van het uitgewerkte voorbeeld, en die van de seasonal naive forecast als gestreepte lijn. Schuif met de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_step_cell():
    def _frame(step):
        said = C.said(step, L.lang)
        alt = ("Links een lijngrafiek van de fietsers van elk uur van de week vanaf maandag 4 juni 2012, met de seasonal naive "
               "forecast en de forecast van de boom op de kolommen van deze stap. Rechts vijf staven: de MAE van elke stap over "
               "de 52 weken van 2012, met de seasonal naive forecast als gestreepte lijn. " + said)
        return (C.picture(step, {"left": "Week van 4 juni 2012", "right": "MAE over 2012", "ylabel": "geregistreerde fietsers",
                                 "step": "stap", "seasonal": "seasonal naive", "lang": "nl", "alt": alt}), said)

    explore_step = L.explore("feature-step", "stap", _frame, start=1, values=[1, 2, 3, 4, 5], role="hyper",
                             task="Ga van stap 1 naar stap 5. Welke kolommen verlagen de MAE, en welke nauwelijks?")
    explore_step
    return


@app.cell(hide_code=True)
def text_after_steps():
    mo.md(r"""
    Met alleen de kalender zit de boom er 70,2 fietsers per uur naast, en te laag: gemiddeld 58,7 fietsers onder de werkelijkheid. Hij weet wanneer mensen fietsen, niet hoeveel er nu fietsen: minstens de helft van zijn trainingsuren komt uit 2011, toen een uur gemiddeld 115 fietsers had, tegen 192 in 2012, en geen van zijn kolommen zegt welk jaar het is. `lag_168` brengt het niveau, maar alleen daarmee doet de boom het slechter dan de seasonal naive forecast, die precies die kolom is: 49,5 tegen 46,8. Twee weken doen het beter, 42,1. De kalender voegt over het hele jaar bijna niets toe, 42,0, maar op de 11 feestdagen van 2012 zit de boom er 69,8 fietsers per uur naast, tegen 97,5 met alleen de lags. Dat doet `workingday`, dat zegt dat de dag een feestdag is, iets wat de lags niet kunnen weten: met het uur, de weekdag en de maand maar zonder `workingday` zit de boom er op de feestdagen nog 93,9 naast. Het werkelijke weer brengt de MAE naar 34,2: ongeveer wat een perfecte weersverwachting hier toevoegt.

    ### Recept

    Zo voorspel je met een boom:

    1. Zet de reeks op een rooster van tijd met `asfreq`, zodat `k` rijen `k` stappen zijn.
    2. Kies de horizon `h`, en maak lag features met lags van minstens `h`.
    3. Voeg kalenderfeatures uit de index toe, en wat er verder bij de origin bekend is, zoals feestdagen.
    4. Train bij elke origin op de rijen ervoor waarvan het label bekend is, en voorspel de `h` rijen erna.
    5. Vergelijk met de seasonal naive forecast en met een gewoon gemiddelde van de lags: de boom moet ze allebei verslaan.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel bouwt de tabel van het recept, met `lag_168` en `lag_336`, het uur, de weekdag, de maand en `workingday`, en backtest de boom een week vooruit vanaf elke maandag van 2 januari tot 24 december 2012. Rij 8784 is maandag 2 januari 2012, 0:00: de 366 dagen ervoor maal 24 uur. Naast de boom scoort de cel de seasonal naive forecast, die de kolom `lag_168` zelf is, en het gemiddelde van de twee lags, dat geen model nodig heeft. `threadpool_limits(1)`, uit threadpoolctl, dat met scikit-learn meekomt, laat scikit-learn binnen zijn `with`-blok op één kern rekenen. Op een drukke computer kosten kernen die op elkaar wachten meer dan ze opleveren: in een test van de course, terwijl een ander programma één kern bezighield, deden de 52 modellen er 67 seconden over op vier kernen en 5 op één, met dezelfde forecasts. De cel print het aantal uren in de tabel en hoeveel daarvan geen fietsers hebben, en daarna de MAE van de seasonal naive forecast, van het gemiddelde van twee weken en van de boom. Voer de cel uit.
    """)
    return


@app.cell
def example_forecast():
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits

    _rows = pd.read_csv("data/bike-sharing-hour.csv.gz")
    _when = pd.to_datetime(_rows["dteday"]) + pd.to_timedelta(_rows["hr"], unit="h")
    _hours = _rows.set_index(_when).asfreq("h")   # 1. elk uur een rij, met NaN waar de tabel er geen had

    # 2. en 3. het label, twee lags van minstens een week, en de kalender
    _table = pd.DataFrame(index=_hours.index)
    _table["registered"] = _hours["registered"]
    _table["lag_168"] = _hours["registered"].shift(168)
    _table["lag_336"] = _hours["registered"].shift(336)
    _table["hour"] = _hours.index.hour
    _table["weekday"] = _hours.index.dayofweek
    _table["month"] = _hours.index.month
    _table["workingday"] = _hours["workingday"]
    _columns = ["lag_168", "lag_336", "hour", "weekday", "month", "workingday"]
    print(len(_table), _table["registered"].isna().sum())

    # 4. elke maandag van 2012: train op de uren ervoor die fietsers hebben, voorspel de 168 erna
    _h = 168
    _tree = []
    _seasonal = []
    _two = []
    with threadpool_limits(1):   # scikit-learn op één kern: zie de tekst boven de cel
        for _origin in range(8784, len(_table) - _h + 1, _h):
            _train = _table.iloc[:_origin].dropna(subset=["registered"])
            _test = _table.iloc[_origin:_origin + _h]
            _model = HistGradientBoostingRegressor(random_state=0)
            _model.fit(_train[_columns], _train["registered"])
            _y = _test["registered"]
            _tree.append(np.nanmean(abs(_y - _model.predict(_test[_columns]))))
            # 5. de seasonal naive forecast is de kolom lag_168; het gemiddelde van de twee lags heeft geen model nodig
            _seasonal.append(np.nanmean(abs(_y - _test["lag_168"])))
            _two.append(np.nanmean(abs(_y - (_test["lag_168"] + _test["lag_336"]) / 2)))
    print(round(np.mean(_seasonal), 1), round(np.mean(_two), 1), round(np.mean(_tree), 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De tabel heeft 17.544 uren, waarvan 165 zonder fietsers. Een week vooruit over 2012 zit de seasonal naive forecast er 46,8 fietsers per uur naast, het gemiddelde van twee weken 42,1, en de boom 42,0; de boom verslaat de seasonal naive forecast in 34 van de 52 weken. Over het hele jaar doen de twee lags het werk. De eigen winst van de boom zit waar hij kolommen combineert die een vuistregel niet kan combineren, zoals de feestdagen van `workingday`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lag_features():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Lag features

        `lag_features(y, lags, h)` moet een tabel geven met de rijen van `y` en een kolom `lag_k` voor elke lag `k` in `lags` die een forecast `h` stappen vooruit mag gebruiken, en geen andere. De eerste regels maken de lege tabel en gaan langs de lags; vervang de `...` door wat er in de loop moet. De check roept `lag_features(riders, [1, 24, 168, 336], 168)` aan, met `riders` de geregistreerde fietsers van elk uur op het rooster van uren, en probeert vier korte Series met andere lags en horizons.
        """),
        L.task("Laat `lag_features` een kolom maken voor elke lag die de horizon toestaat, en geen andere."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lag_features():
    box_lag_features = L.code_box("lag-features", lines=9)
    box_lag_features
    return (box_lag_features,)


@app.cell(hide_code=True)
def check_lag_features(box_lag_features):
    result_lag_features = L.check("lag-features", box_lag_features.value)
    result_lag_features
    return (result_lag_features,)


@app.cell(hide_code=True)
def hints_lag_features():
    L.hints(
        "lag-features",
        [
            "Twee dingen per lag: staat de horizon hem toe, en zo ja, een nieuwe kolom. *Alleen lags die de horizon toestaat* in *Het idee* zegt welke lags mogen blijven.",
            "Een lag `k` mag blijven als hij minstens `h` is. Zijn kolom is `y` met elke waarde `k` rijen naar achteren, met als naam `lag_` gevolgd door het getal `k`; een f-string maakt die naam.",
            "`if k >= ___:` en daarbinnen `table[f\"lag_{k}\"] = y.shift(___)`.",
        ],
        why="Een week vooruit is `h` 168: `lag_1` en `lag_24` gaan eruit, `lag_168` en `lag_336` blijven. Voor zondag 23:00, 167 uur na de origin, is `lag_168` de zondag 23:00 een week eerder: het laatste uur voor de origin.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_tree_forecast():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een forecast met een boom

        `tree_forecast(table, columns, origin, h)` moet een boom trainen op de rijen voor `origin` waarvan `registered` bekend is, met `columns` als features en `registered` als label, en de `h` rijen vanaf `origin` voorspellen. De regels voor `test`, het model en `predict` staan er al; vervang de drie keer `...`. De check roept `tree_forecast(table, columns, 12480, 168)` aan: de tabel van het uitgewerkte voorbeeld, zijn zes kolommen, en de origin maandag 4 juni 2012, 0:00. Hij probeert ook drie gevallen op twee verzonnen tabellen, waarvan één met gaten.
        """),
        L.task("Laat `tree_forecast` alleen leren van de rijen voor de origin die fietsers hebben."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tree_forecast():
    box_tree_forecast = L.code_box("tree-forecast", lines=9)
    box_tree_forecast
    return (box_tree_forecast,)


@app.cell(hide_code=True)
def check_tree_forecast(box_tree_forecast):
    result_tree_forecast = L.check("tree-forecast", box_tree_forecast.value)
    result_tree_forecast
    return (result_tree_forecast,)


@app.cell(hide_code=True)
def hints_tree_forecast():
    L.hints(
        "tree-forecast",
        [
            "Het model mag leren van wat bij de origin bekend was: de rijen ervoor. En het kan alleen leren van een rij waarvan het het label heeft.",
            "`train` zijn de rijen van `table` tot `origin`, zoals in de backtest van de vorige les, zonder de rijen waar `registered` NaN is; `dropna` met `subset` laat alleen die vallen. `fit` krijgt eerst de features, dan het label.",
            "`train = table.iloc[:___].dropna(subset=[___])` en `model.fit(train[___], train[___])`.",
        ],
        why="Bij origin 12.480 bevat `train` de 12.355 uren voor maandag 4 juni 2012 die fietsers hebben: 125 van de 12.480 hebben er geen. Het model leert van hun zes kolommen en voorspelt de 168 uren van de week uit dezelfde zes kolommen van `test`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_known():
    mo.md(r"""
    ### Oefening 3. Wat een forecast mag weten
    """)
    return


@app.cell(hide_code=True)
def choice_known_cell():
    choice_known = L.choice(
        "known-columns",
        question="Vanaf maandag 0:00 voorspelt een boom elk uur van de week erna. Welke van deze kolommen mag zijn tabel bevatten?",
        options={
            "right": "`workingday`: of elke dag een werkdag is",
            "lag24": "`lag_24`: de fietsers 24 uur eerder",
            "temp": "`temp`: de werkelijke temperatuur van elk uur",
        },
        correct="right",
        why={
            "lag24": "Voor maandag is `lag_24` de zondag ervoor, en die is bekend; vanaf dinsdag is het een uur van de week zelf, dat de forecast moet voorspellen. Zet hem erbij, en de backtest zegt 38,7 in plaats van 42,0: beter, en niet waar. Zie *Alleen lags die de horizon toestaat* in *Het idee*.",
            "temp": "De werkelijke temperatuur ken je pas achteraf. Met het werkelijke weer zegt de backtest 34,2: ongeveer wat een perfecte weersverwachting hier oplevert. Een echte forecast kan alleen een weersverwachting gebruiken die bij de origin gemaakt is. Zie stap 5 van de slider.",
            "right": "Weekenden en feestdagen zijn lang vooraf bekend: `workingday` is een kalenderfeature. Op de 11 feestdagen van 2012 zit de boom met het uur, de weekdag en de maand er nog 93,9 fietsers per uur naast; met `workingday` erbij 69,8.",
        },
    )
    choice_known
    return (choice_known,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Lag features zetten de autocorrelatie uit [Lijkt vandaag op gisteren?](/course/open/ts-autocorrelation?lang=nl) aan het werk: de lags met de sterkste correlatie zijn de kolommen die het waard zijn. De tabel, het label en de boom zijn die uit [Boom na boom op wat overblijft](/course/open/ml-gradient-boosting?lang=nl); de backtest en de seasonal naive forecast zijn die uit [Een forecast testen op het verleden](/course/open/ts-backtesting?lang=nl), en lags van minstens de horizon houden hem eerlijk. Hiermee eindigt de module over tijdreeksen. De volgende module gaat over aanbevelen, met een tabel die bijna leeg is: welke klant wil welk product.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_known, result_lag_features, result_tree_forecast):
    L.wrap_up(result_lag_features, result_tree_forecast, choice_known)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
