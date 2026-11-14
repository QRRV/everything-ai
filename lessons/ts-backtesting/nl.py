# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ts-backtesting. Route: course/route/lessen/klassiek-ml.yaml. Checks en plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ts-backtesting", "nl")
    C = L.checks   # de fietsers van elke dag, drie forecasts gebacktest over 2012 voor elke horizon, en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Lijkt vandaag op gisteren? Autocorrelatie](/course/open/ts-autocorrelation?lang=nl) en één over [Hoe ver ernaast? MAE, RMSE en R²](/course/open/ml-regression-metrics?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_lag():
    quick_lag = L.choice(
        "quick-lag",
        question="De fietsen die geregistreerde fietsers elk uur huren, hebben een autocorrelatie van 0,886 bij lag 168. Wat zegt dat?",
        options={
            "right": "Elk uur beweegt sterk samen met hetzelfde uur een week eerder",
            "share": "88,6% van de uren heeft evenveel fietsers als hetzelfde uur een week eerder",
            "rise": "Het aantal fietsers groeit elke week met 88,6%",
        },
        correct="right",
        why={
            "share": "r is geen aandeel gelijke uren: het zegt hoe goed een stijgende rechte lijn past bij de paren van een uur en het uur 168 eerder. Zie *Autocorrelatie* in *Het idee* in [Autocorrelatie](/course/open/ts-autocorrelation?lang=nl).",
            "rise": "Een correlatie zegt niets over groei: een reeks die elke week verdubbelt en een die gelijk blijft, kunnen allebei op een lijn met hun verleden liggen. Zie *De correlatiecoëfficiënt* in *Het idee* in [Autocorrelatie](/course/open/ts-autocorrelation?lang=nl).",
            "right": "Na een drukke maandag 8:00 komt een drukke maandag 8:00. Vandaag wordt dat een forecast: hetzelfde uur, of dezelfde dag, een week eerder.",
        },
    )
    quick_lag
    return (quick_lag,)


@app.cell(hide_code=True)
def quick_check_mae():
    quick_mae = L.choice(
        "quick-mae",
        question="Een forecast voor drie dagen zit 100 te laag, 300 te hoog en 200 te laag. Wat is de MAE ervan?",
        options={
            "right": "200",
            "zero": "0",
            "largest": "300",
        },
        correct="right",
        why={
            "zero": "Met hun tekens heffen de missers elkaar op: 100 − 300 + 200 = 0. De MAE neemt ze zonder teken. Zie *Het idee* in [Hoe ver ernaast?](/course/open/ml-regression-metrics?lang=nl)",
            "largest": "300 is de grootste misser. De MAE is hun gemiddelde grootte: (100 + 300 + 200) / 3. Zie *Het idee* in [Hoe ver ernaast?](/course/open/ml-regression-metrics?lang=nl)",
            "right": "(100 + 300 + 200) / 3 = 200: gemiddeld 200 ernaast, in de eenheid van wat voorspeld wordt. Vandaag beoordeelt ze forecasts op dagen die ze niet gezien hebben.",
        },
    )
    quick_mae
    return (quick_mae,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een forecast is maar zoveel waard als zijn missers op dagen die hij niet gezien had. Fit een model op heel 2012 en meet het op 2012, en het lijkt beter dan het ooit zal zijn, want het heeft de antwoorden gezien. De eerlijke test maakt elke forecast zoals hij toen gemaakt zou zijn: met alleen het verleden, en legt hem daarna naast wat er gebeurde. En de score betekent pas iets naast de score van een forecast die geen enkele moeite kost.

    **Een voorbeeld uit het dagelijks leven.** Om een weervoorspeller te beoordelen, neem je wat die elke avond over de volgende dag zei, en vergelijk je dat met het weer dat kwam. Een voorspelling die de dag erna is geschreven, telt niet. En als "morgen wordt zoals vandaag" het even goed doet, voegt de voorspeller niets toe.
    """)
    return


@app.cell(hide_code=True)
def predict_yesterday_or_last_week():
    guess_day = L.guess(
        "day-ahead-guess",
        question="Voorspel de geregistreerde fietsers van elke dag van 2012 één dag vooruit, op twee manieren: de dag ervoor, of dezelfde dag van de week een week ervoor. Welke zit er minder naast, gemeten met de MAE?",
        options={
            "right": "De dag ervoor",
            "week": "Dezelfde dag van de week een week ervoor",
            "same": "Ze zitten er ongeveer even ver naast",
        },
        reveal="**De dag ervoor: een MAE van 804, tegen 860.** Een dag lijkt meer op de dag ervoor dan op dezelfde dag een week eerder: de autocorrelatie bij lag 1 is 0,832, bij lag 7 0,764. Maar voorspel een hele week vooruit, en de dag ervoor moet voor alle zeven dagen dienen, zaterdag en zondag ook: dan groeit de MAE ervan naar 1.340, terwijl de week ervoor op 859 blijft.",
    )
    guess_day
    return (guess_day,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Alleen het verleden.** Een forecast mag alleen gebruiken wat bekend was toen hij gemaakt werd. Het moment waarop hij gemaakt wordt, is de **origin**; de dagen ervoor zijn de trainingsdagen, de `h` dagen erna de testdagen. `h` is hoe ver de forecast vooruit kijkt.

        **Backtesting.** Zet de origin op maandag 2 januari 2012, train op de dagen ervoor, voorspel de week erna, en meet de MAE op die week. Schuif de origin dan een week op en doe het opnieuw: 52 origins, elke maandag van 2 januari tot en met 24 december 2012. Het gemiddelde van de 52 MAE's is de score van de forecast. Een forecast zo op het verleden testen is **backtesting**, en de origin steeds opschuiven is een **rolling origin**. In Python telt `range(first, len(y) - h + 1, h)` vanaf `first` in stappen van `h`, het derde getal, tot de laatste origin waarvan de `h` testdagen nog passen.

        **Twee forecasts die geen moeite kosten.** De **naive forecast** zegt dat elke dag vooruit op de laatste dag lijkt. De **seasonal naive forecast** zegt dat elke dag vooruit lijkt op dezelfde dag van de laatste periode: met een periode van 7 dagen is elke maandag vooruit de laatste maandag. Met `last` de laatste periode van de trainingsdagen is dag `i` vooruit gelijk aan `last[i % period]`: `%` geeft de rest van `i` gedeeld door `period`, dus na de laatste dag van de periode begint het weer bij de eerste. Een forecast die ze allebei niet verslaat, is de moeite niet waard. Naast deze twee test deze les de forecast uit [Trend, seizoen en de rest](/course/open/ts-trend-season?lang=nl): de laatste trend plus het seizoen.

        **MASE.** Of een MAE van 859 fietsers goed of slecht is, hangt af van de reeks. De **MASE**, de mean absolute scaled error, deelt de MAE door de MAE die de naive forecast één dag vooruit op de trainingsdagen haalt: hier 510,6 fietsers. Onder 1 zit een forecast er minder naast dan "morgen is zoals vandaag" op de trainingsdagen deed; en reeksen van elke grootte zijn te vergelijken.

        **sMAPE.** De **sMAPE**, de symmetric mean absolute percentage error, neemt elke misser als deel van de echte waarde en de forecast samen: 2 × |echt − forecast| / (|echt| + |forecast|), en geeft daarvan het gemiddelde in procenten, van 0 tot 200%.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **origin** | het moment waarop een forecast gemaakt wordt: hij mag alleen gebruiken wat ervoor kwam | `origin` |
        | **backtesting** | een forecast op het verleden testen, zoals hij toen gemaakt zou zijn | |
        | **rolling origin** | de origin stap voor stap opschuiven, en vanaf elke opnieuw voorspellen | `range(first, len(y) - h + 1, h)` |
        | **naive forecast** | elke stap vooruit is de laatste waarde | `np.full(h, train[len(train) - 1])` |
        | **seasonal naive forecast** | elke stap vooruit is dezelfde stap van de laatste periode | `last[i % period]` |
        | `%` | de rest van een deling: `9 % 7` is 2 | `i % period` |
        | **MASE** | de MAE gedeeld door de MAE van de naive forecast één stap vooruit op de training data | |
        | **sMAPE** | het gemiddelde van elke misser als deel van de echte waarde en de forecast samen, in procenten | |
        """),
        L.watch_out("Een percentage ontspoort vlak bij 0. Op maandag 29 oktober 2012, orkaan Sandy, huurden geregistreerde fietsers 20 fietsen, en de seasonal naive forecast zei 6.228: die dag alleen telt in de sMAPE voor 198,7%, dicht bij het meeste wat een dag kan tellen, 200%. Op een dag met bijna niets te tellen, zit elke forecast er als aandeel bijna helemaal naast. De MAE telt die dag als 6.208 fietsers ernaast, net als elke andere misser van die grootte."),
        mo.md(r"""
        Links, vanaf maandag 4 juni 2012: de fietsers van de vier weken ervoor, een verticale gestreepte lijn bij de origin, de fietsers van de dagen erna, en de drie forecasts voor de komende `h` dagen: de open grijze stippen zijn de naive forecast, de gestreepte lijn de seasonal naive, de dikke lijn trend plus seizoen. Rechts de MAE van elk over alle origins van 2012 voor die horizon. Schuif met de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_horizon_cell():
    def _frame(h):
        said = C.said(h, L.lang)
        alt = ("Links een lijngrafiek van de fietsers van de vier weken voor maandag 4 juni 2012 en van het gekozen aantal dagen erna, "
               "met de naive, de seasonal naive en de forecast van trend plus seizoen voor die dagen. Rechts drie staven: de MAE van "
               "elke forecast over alle origins van 2012. " + said)
        return (C.picture(h, {"left": "Vanaf maandag 4 juni 2012", "right": "MAE over 2012", "ylabel": "geregistreerde fietsers",
                              "dates": ["7 mei", "21 mei", "4 jun", "18 jun", "2 jul"],
                              "methods": ["naive", "seasonal", "trend+seizoen"], "alt": alt}), said)

    explore_horizon = L.explore("horizon-days", "horizon (dagen)", _frame, start=7, values=C.HORIZONS, role="hyper",
                                task="Ga van een horizon van 1 dag naar 28. Waar wint de naive forecast, en hoe groeien de missers?")
    explore_horizon
    return


@app.cell(hide_code=True)
def text_after_horizon():
    mo.md(r"""
    Eén dag vooruit verslaat de naive forecast de seasonal naive, 804 tegen 860, zoals de gok liet zien. Vanaf 2 dagen wint de seasonal naive forecast, en verandert zijn MAE tot 14 dagen nauwelijks, van 859 naar 868; de naive forecast zit er elke dag verder naast: 1.340 een week vooruit, 1.510 vier weken vooruit. Trend plus seizoen is bij elke horizon het best, van 669 een dag vooruit tot 814 vier weken vooruit. Hoe verder vooruit, hoe meer een forecast ernaast zit: noem bij elke score de horizon.

    ### Recept

    Zo test je een forecast op het verleden:

    1. Kies de horizon die je nodig hebt en de eerste origin: hier een week, vanaf maandag 2 januari 2012.
    2. Train bij elke origin alleen op wat ervoor kwam, en voorspel de `h` stappen erna.
    3. Meet de fout op die stappen, en schuif de origin `h` op.
    4. Neem het gemiddelde over alle origins, en doe hetzelfde voor de naive en de seasonal naive forecast.
    5. Geef de MAE met de MASE, om reeksen te vergelijken; let op met de sMAPE waar waarden dicht bij 0 komen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel backtest de naive en de seasonal naive forecast een week vooruit over 2012, met de stappen van het recept. `_backtest` krijgt een forecast als argument: een functie die de trainingsdagen krijgt en de volgende 7 geeft. Ze geeft een lijst van twee getallen terug, de gemiddelde MAE en de sMAPE. De cel print de schaal van de MASE, en dan de MAE, de MASE en de sMAPE van beide forecasts. Voer de cel uit.
    """)
    return


@app.cell
def example_backtest():
    import numpy as np
    import pandas as pd

    _y = pd.read_csv("data/bike-sharing-day.csv.gz")["registered"].to_numpy()

    # 1. een week vooruit, vanaf maandag 2 januari 2012, plek 366
    _h = 7
    _first = 366


    def _naive(train):
        return np.full(_h, train[len(train) - 1])


    def _seasonal(train):
        _last = train[len(train) - 7:]
        _out = []
        for _i in range(_h):
            _out.append(_last[_i % 7])
        return np.array(_out)


    # 2. tot en met 4. elke origin: train op de dagen ervoor, test op de volgende 7, neem de gemiddelden
    def _backtest(forecast):
        _maes = []
        _smapes = []
        for _origin in range(_first, len(_y) - _h + 1, _h):
            _test = _y[_origin:_origin + _h]
            _guess = forecast(_y[:_origin])
            _maes.append(np.mean(abs(_test - _guess)))
            _smapes.append(np.mean(2 * abs(_test - _guess) / (abs(_test) + abs(_guess))))
        return [np.mean(_maes), 100 * np.mean(_smapes)]


    # 5. de schaal van de MASE: de naive forecast een dag vooruit, op de trainingsdagen
    _train = _y[:_first]
    _scale = np.mean(abs(_train[1:] - _train[:len(_train) - 1]))
    print(round(_scale, 1))
    for _forecast in [_naive, _seasonal]:
        _score = _backtest(_forecast)
        print(round(_score[0], 1), round(_score[0] / _scale, 2), round(_score[1], 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De naive forecast een dag vooruit zat op de trainingsdagen gemiddeld 510,6 fietsers ernaast. Een week vooruit over 2012 heeft de naive forecast een MAE van 1.340,0, een MASE van 2,62 en een sMAPE van 35,4%; de seasonal naive forecast 858,5, 1,68 en 22,6%. Trend plus seizoen haalt volgens de checks van deze les 739,1, 1,45 en 19,0%. Geen van de drie komt onder een MASE van 1: een week vooruit voorspellen is moeilijker dan morgen uit vandaag. De seasonal naive forecast verslaat de naive in 42 van de 52 weken.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_seasonal_naive():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De seasonal naive forecast

        `seasonal_naive(train, h, period)` moet de `h` stappen na `train` voorspellen: stap `i` vooruit is de waarde op dezelfde plek in de laatste periode. De eerste regel neemt die laatste periode, `last`, en de loop gaat langs de stappen vooruit. Vervang de `...`. De check roept `seasonal_naive(train, 10, 7)` aan, met `train` de geregistreerde fietsers tot en met zondag 1 januari 2012, zodat de forecast over een periode en een beetje loopt, en probeert er nog vier: een horizon korter dan de periode, een periode van 2 en van 3, en een periode van 1.
        """),
        L.task("Laat `seasonal_naive` de laatste periode herhalen voor elke stap vooruit."),
    ])
    return


@app.cell(hide_code=True)
def exercise_seasonal_naive():
    box_seasonal_naive = L.code_box("seasonal-naive", lines=8)
    box_seasonal_naive
    return (box_seasonal_naive,)


@app.cell(hide_code=True)
def check_seasonal_naive(box_seasonal_naive):
    result_seasonal_naive = L.check("seasonal-naive", box_seasonal_naive.value)
    result_seasonal_naive
    return (result_seasonal_naive,)


@app.cell(hide_code=True)
def hints_seasonal_naive():
    L.hints(
        "seasonal-naive",
        [
            "`last` bevat `period` waarden, één voor elke plek in de periode. Stap `i` vooruit neemt er een van; na de laatste begint het weer bij de eerste.",
            "Welke plek in de periode is stap `i`? Voor `period` 7: de stappen 0 tot en met 6 zijn de plekken 0 tot en met 6, stap 7 is weer plek 0, stap 8 plek 1. *Twee forecasts die geen moeite kosten* in *Het idee* heeft het teken dat dit doet.",
            "`out.append(last[i ___ period])`.",
        ],
        why="Met `period` 7 en `h` 10 nemen de stappen 0 tot en met 9 de plekken 0, 1, 2, 3, 4, 5, 6, 0, 1, 2 van `last`: 9 % 7 = 2.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_backtest():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een backtest met een rolling origin

        `backtest(y, first, h)` moet de MAE van de naive forecast voor elke origin geven, als lijst: vanaf `first`, in stappen van `h`. Bij elke origin moet `train` zijn wat ervoor kwam, en `test` de `h` stappen vanaf de origin. Vervang de twee `...`. De check roept `backtest(riders, 366, 7)` aan, de geregistreerde fietsers van alle 731 dagen, vanaf maandag 2 januari 2012 een week tegelijk, en probeert er nog vier korte arrays met andere origins en horizons.
        """),
        L.task("Laat `backtest` alleen op het verleden trainen, en testen op de `h` stappen na de origin."),
    ])
    return


@app.cell(hide_code=True)
def exercise_backtest():
    box_backtest = L.code_box("backtest", lines=10)
    box_backtest
    return (box_backtest,)


@app.cell(hide_code=True)
def check_backtest(box_backtest):
    result_backtest = L.check("backtest", box_backtest.value)
    result_backtest
    return (result_backtest,)


@app.cell(hide_code=True)
def hints_backtest():
    L.hints(
        "backtest",
        [
            "Bij elke origin mag de forecast weten wat er voor de origin kwam, en niets vanaf de origin. De testdagen zijn de dagen die hij voorspelt.",
            "`train` is een slice van `y` die vlak voor `origin` stopt; `test` begint bij `origin` en bevat `h` waarden. Een slice `y[a:b]` stopt vlak voor `b`.",
            "`train = y[:___]` en `test = y[origin:___]`.",
        ],
        why="Met `y` = [1, 2, ..., 10], `first` 4 en `h` 3 zijn de origins 4 en 7. Bij 4 is `train` gelijk aan [1, 2, 3, 4] en `test` aan [5, 6, 7]; de forecast is 4, 4, 4, en de MAE (1 + 2 + 3) / 3 = 2.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mase():
    mo.md(r"""
    ### Oefening 3. Wat de MASE toevoegt
    """)
    return


@app.cell(hide_code=True)
def choice_mase_cell():
    choice_mase = L.choice(
        "mase",
        question="Over 2012 heeft de seasonal naive forecast een week vooruit een MAE van 859 fietsers en een MASE van 1,68. Wat voegt de 1,68 toe aan de 859?",
        options={
            "right": "Hij vergelijkt de 859 met de naive forecast een dag vooruit op de trainingsdagen, 510,6: een week vooruit zit deze forecast er 1,68 keer zo ver naast",
            "percent": "De forecast zit er 1,68% van de fietsers naast",
            "large": "Hij telt grote missers zwaarder, zoals de RMSE",
        },
        correct="right",
        why={
            "percent": "Een aandeel van de fietsers is wat de sMAPE geeft, hier 22,6%. De MASE deelt door een MAE, dus heeft hij helemaal geen eenheid. Zie *MASE* in *Het idee*.",
            "large": "De RMSE kwadrateert de missers; de MASE deelt de MAE, die elke misser naar zijn grootte telt, door een andere MAE. Zie *MASE* in *Het idee*.",
            "right": "1,68 zegt hoe deze forecast het doet tegen een vaste maatstaf, zonder fietsers als eenheid, zodat een station dat 50 fietsen per dag verhuurt en een stad die er 5.000 verhuurt te vergelijken zijn.",
        },
    )
    choice_mase
    return (choice_mase,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De naive en de seasonal naive forecast zijn de baselines uit [Versla altijd een eenvoudige baseline](/course/open/ml-baselines?lang=nl) voor tijdreeksen, gescoord met de MAE uit [Hoe ver ernaast?](/course/open/ml-regression-metrics?lang=nl). Welke van de twee wint, stond al in de autocorrelatie: lag 1 is het sterkst een dag vooruit, lag 7 voor een week. De volgende les, *Vraag voorspellen met bomen*, maakt van het verleden kolommen, zoals de fietsers één en twee weken eerder, laat gradient boosting ervan leren, en test dat met deze backtest tegen de seasonal naive forecast.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_mase, result_backtest, result_seasonal_naive):
    L.wrap_up(result_seasonal_naive, result_backtest, choice_mase)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
