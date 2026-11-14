# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ts-trend-season. Route: course/route/lessen/klassiek-ml.yaml. Checks en plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ts-trend-season", "nl")
    C = L.checks   # de fietsers van elke dag, de decompositie voor elk window van de slider, en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Tijd als index: datums en resampling in pandas](/course/open/ts-time-index?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rolling():
    quick_rolling = L.choice(
        "quick-rolling",
        question="`y` heeft één waarde per dag, met de datum als index. Wat is `y.rolling(7).mean()` op zondag 8 januari 2012?",
        options={
            "right": "Het gemiddelde van de 7 dagen tot en met zondag 8 januari: maandag 2 tot en met zondag 8 januari",
            "around": "Het gemiddelde van de 3 dagen voor zondag 8 januari, die zondag zelf en de 3 dagen erna",
            "week": "Eén getal voor de hele week, op elke dag ervan hetzelfde",
        },
        correct="right",
        why={
            "around": "Een rolling window eindigt bij de dag zelf: het kent alleen het verleden. Vandaag zet `center=True` de dag in het midden van zijn window. Zie *Gladstrijken* in *Het idee* in [Tijd als index](/course/open/ts-time-index?lang=nl).",
            "week": "Eén getal per week is wat resampling geeft, `y.resample(\"W\").mean()`. Een rolling window geeft een getal voor elke dag. Zie *Gladstrijken* in *Het idee* in [Tijd als index](/course/open/ts-time-index?lang=nl).",
            "right": "Het window eindigt bij de dag zelf, dus loopt het gemiddelde achter. Vandaag schuift het window naar het midden, en vindt het de trend van een reeks.",
        },
    )
    quick_rolling
    return (quick_rolling,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Capital Bikeshare verhuurt fietsen vanaf stations door heel Washington, D.C. De tabel van deze les telt de fietsen die elke dag van 2011 en 2012 gehuurd zijn, door geregistreerde fietsers, leden die vooral naar hun werk fietsen, en door losse fietsers met een pas voor één dag. Hoeveel fietsen moeten er volgende maandag klaarstaan? In de getallen zitten drie dingen door elkaar: het systeem groeit, dus 2012 is drukker dan 2011; elke week gaat op dezelfde manier op en neer; en sommige dagen zijn een verrassing, zoals maandag 29 oktober 2012, toen orkaan Sandy de stad stillegde en geregistreerde fietsers 20 fietsen huurden. Om te voorspellen haal je de drie uit elkaar: wat groeit, wat elke week terugkomt, en wat niemand zag aankomen.

    **Een voorbeeld uit het dagelijks leven.** Het water in een haven stijgt en daalt met het getij, twee keer per dag. Over de jaren stijgt de zee langzaam. Op een dag met storm staat het water veel hoger dan het getij zegt. Getij, stijgende zee, storm: een patroon dat terugkomt, een langzame beweging, en een verrassing.
    """)
    return


@app.cell(hide_code=True)
def predict_two_weeks():
    guess_riders = L.guess(
        "two-kinds-guess",
        question="Vergeleken met hun eigen trend huren geregistreerde fietsers de meeste fietsen op donderdag, 411 meer, en de minste op zondag, 760 minder. Losse fietsers huren de meeste op zaterdag, 631 meer, en de minste op woensdag, 300 minder. En alle fietsers samen: hoe ver liggen de beste en de slechtste dag van de week uit elkaar?",
        options={
            "add": "Ongeveer 2.100: de twee patronen tellen op",
            "right": "Ongeveer 430: de twee patronen heffen elkaar grotendeels op",
            "zero": "0: de twee patronen heffen elkaar precies op",
        },
        reveal="**429.** Alle fietsers samen huren er op vrijdag 171 meer en op zondag 258 minder. Waar geregistreerde fietsers dalen, stijgen losse fietsers, en in de som verstoppen de twee sterke weekpatronen elkaar grotendeels. Een reeks die een som is, kan haar patronen verstoppen: splits haar waar dat kan. Deze les volgt de geregistreerde fietsers.",
    )
    guess_riders
    return (guess_riders,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Drie delen.** Veel tijdreeksen zijn een som van drie delen: reeks = trend + seizoen + rest. De **trend** is de langzame beweging: het aantal geregistreerde fietsers groeide van 2011 naar 2012. Het **seizoenspatroon**, kort het seizoen, is een patroon dat zich herhaalt met een vaste **periode**: hier een week van 7 dagen, dus het seizoen is 7 getallen, één per dag van de week. De **rest** is wat overblijft: weer, feestdagen, een orkaan. Een reeks in deze drie delen splitsen is een **decompositie**.

        **De trend: een gecentreerd voortschrijdend gemiddelde.** Neem voor elke dag het gemiddelde van de dagen eromheen: `half` dagen ervoor, de dag zelf en `half` dagen erna, een window van `2 * half + 1` dagen. Dat is een **gecentreerd voortschrijdend gemiddelde**. Een window van precies één periode, 7 dagen, bevat elke dag van de week één keer, dus middelt de week eruit, en wat overblijft is de trend. In pandas: `y.rolling(7, center=True).mean()`; `center=True` zet de dag in het midden van zijn window, waar `rolling` hem aan het eind zet. De eerste en de laatste 3 dagen krijgen geen trend, omdat hun window niet past. In NumPy maakt `np.full(n, np.nan)` een array van `n` waarden die allemaal `np.nan` zijn, de NaN van NumPy, om dag voor dag in te vullen.

        **Het seizoen: het gemiddelde per plek.** `rest = y - trend` is de reeks zonder haar trend. Het seizoen van de maandagen is het gemiddelde van `rest` over alle maandagen, en zo voor elke dag van de week. In NumPy neemt een slice met een stap elke 7e waarde: `rest[k::7]` zijn de waarden op plek `k`, `k + 7`, `k + 14` enzovoort, en `np.nanmean` is het gemiddelde dat NaN weglaat, zoals de dagen aan de randen zonder trend. Schuif als laatste de zeven gemiddelden zo op dat ze samen 0 zijn: het seizoen mag het niveau niet veranderen, dat is de taak van de trend. In pandas geeft `rest.index.dayofweek` elke datum zijn dag van de week, 0 voor maandag tot 6 voor zondag, en geeft `rest.groupby(rest.index.dayofweek).mean()` de zeven gemiddelden.

        **De rest.** rest = reeks − trend − seizoen. Een rest ver van 0 is een verrassing die een blik waard is. De laagste van allemaal is maandag 29 oktober 2012, orkaan Sandy: 3.799 onder trend en seizoen samen.

        **Een forecast.** Een **forecast** is een schatting van waarden die nog niet gebeurd zijn. De eenvoudigste uit een decompositie: zet de trend voort vanaf zijn laatste waarde, en tel het seizoen op van de dag die je voorspelt. De rest blijft weg: die is wat niemand zag aankomen.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **trend** | de langzame beweging van een reeks, zonder het patroon dat terugkomt | `y.rolling(7, center=True).mean()` |
        | **seizoenspatroon** | een patroon dat zich herhaalt met een vaste periode, zoals elke week; kort het seizoen | |
        | **periode** | de lengte van het patroon dat terugkomt: 7 dagen, 24 uur, 12 maanden | `period` |
        | **gecentreerd voortschrijdend gemiddelde** | voor elke stap het gemiddelde van een window met die stap in het midden | `center=True` |
        | **rest** | wat er van de reeks overblijft na de trend en het seizoen | `y - trend - season` |
        | **decompositie** | een reeks splitsen in trend, seizoen en rest | |
        | **forecast** | een schatting van waarden die nog niet gebeurd zijn | |
        | `::` | een slice met een stap: elke 7e waarde vanaf plek `k` | `rest[k::7]` |
        | `np.full` | een array van de gegeven lengte, met overal dezelfde waarde | `np.full(n, np.nan)` |
        | `np.nanmean` | het gemiddelde van een array, zonder NaN | `np.nanmean(rest[k::7])` |
        | `dayofweek` | de dag van de week van elke datum in een datetime index, 0 voor maandag tot 6 voor zondag | `y.index.dayofweek` |
        """),
        L.watch_out("Een seizoen is een gemiddelde, en een paar bijzondere dagen trekken eraan. Het seizoen van maandag is maar +15, ver onder dat van dinsdag, +291: 15 van de 21 feestdagen in deze twee jaar vallen op een maandag, en op die 15 maandagen huurden geregistreerde fietsers 440 fietsen minder dan de trend, op de andere 90 maandagen 92 meer. Zonder de feestdagen is het seizoen van maandag +74. Kijk wat aan een seizoen trekt voor je het vertrouwt."),
        mo.md(r"""
        Links is de dunne grijze lijn het aantal geregistreerde fietsers van elke dag van 2011 en 2012, en de dikke lijn de trend: het gecentreerde voortschrijdende gemiddelde over het window dat je kiest. Rechts het seizoen dat overblijft, één staaf per dag van de week. Schuif met de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_window_cell():
    def _frame(days):
        said = C.said(days, L.lang)
        alt = ("Links een lijngrafiek van de fietsen die geregistreerde fietsers elke dag van 2011 en 2012 huurden, tussen 0 en "
               "7.000, met de trend over het gekozen window als dikkere lijn. Rechts een staafdiagram van het seizoen, één staaf per "
               "dag van de week van maandag tot zondag. " + said)
        return (C.picture(days, {"left": "Geregistreerde fietsers en trend", "right": "Seizoen", "ylabel": "fietsen per dag",
                                 "months": ["jan 2011", "jul 2011", "jan 2012", "jul 2012"], "lang": L.lang, "alt": alt}), said)

    explore_window = L.explore("trend-window", "window (dagen)", _frame, start=7, values=C.WINDOWS, role="hyper",
                               task="Ga van een window van 3 dagen naar 35. Vanaf welk window blijft het seizoen hetzelfde, en wat gebeurt er met de trend?")
    explore_window
    return


@app.cell(hide_code=True)
def text_after_window():
    mo.md(r"""
    Een window korter dan een week laat een deel van de week in de trend: met 3 dagen springt de trend gemiddeld 333 per dag en beslaat het seizoen maar 553. Vanaf 7 dagen bevat elk window hele weken, en blijft het seizoen rond 1.170 tot 1.185: elk zo'n window vindt dezelfde week. Een langer window geeft een rustigere trend, 104, 38 en dan 25 per dag, en laat meer van de langzame beweging in de rest: de standaardafwijking ervan groeit van 596 naar 719. Eén periode is het kortste window dat de week uit de trend houdt.

    ### Recept

    Zo splits je een tijdreeks in trend, seizoen en rest:

    1. Kies de periode: de lengte van het patroon dat terugkomt, zoals 7 dagen, 24 uur of 12 maanden.
    2. Vind de trend met een gecentreerd voortschrijdend gemiddelde over één periode.
    3. Trek de trend eraf, neem het gemiddelde per plek in de periode, en schuif de gemiddelden zo op dat ze samen 0 zijn: dat is het seizoen.
    4. Trek ook het seizoen eraf: dat is de rest. Bekijk de grootste waarden ervan.
    5. Zet voor een forecast de trend voort en tel het seizoen op van de plek die je voorspelt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel splitst de geregistreerde fietsers in pandas. Ze print de trend op 1 juli van beide jaren, de zeven getallen van het seizoen van maandag tot zondag, en de laagste rest met zijn datum: `idxmin` geeft de index van de laagste waarde, zoals `np.argmin` er de plek van geeft. Om van elke dag het seizoen af te trekken, zoekt `_season.loc[...]` het seizoen van elke dag van de week op, en haalt `.to_numpy()` er de index af, zodat pandas rij voor rij aftrekt in plaats van indexlabels te koppelen. Als laatste voorspelt ze de laatste dag, maandag 31 december 2012, vanuit de laatste trend, `.iloc[-1]`, de waarde aan het eind van wat na `dropna` overblijft. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_decomposition():
    import pandas as pd

    # de geregistreerde fietsers van elke dag, met de datum als index
    _df = pd.read_csv("data/bike-sharing-day.csv.gz")
    _df["dteday"] = pd.to_datetime(_df["dteday"])
    _y = _df.set_index("dteday")["registered"]

    # 2. de trend: een gecentreerd voortschrijdend gemiddelde over één periode, 7 dagen
    _trend = _y.rolling(7, center=True).mean()
    print(round(_trend.loc["2011-07-01"]), round(_trend.loc["2012-07-01"]))

    # 3. het seizoen: het gemiddelde per dag van de week, opgeschoven tot het samen 0 is
    _rest = _y - _trend
    _season = _rest.groupby(_rest.index.dayofweek).mean()
    _season = _season - _season.mean()
    print(list(_season.round()))

    # 4. de rest, en zijn laagste dag
    _remainder = _rest - _season.loc[_rest.index.dayofweek].to_numpy()
    print(_remainder.idxmin(), round(_remainder.min()))

    # 5. een forecast voor de laatste dag: de laatste trend plus het seizoen van een maandag
    print(round(_trend.dropna().iloc[-1] + _season.loc[0]), _y.loc["2012-12-31"])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Op 1 juli was de trend 3.594 fietsen per dag in 2011 en 4.958 in 2012: het systeem groeide. Het seizoen loopt van +411 op donderdag tot −760 op zondag, en de laagste rest is orkaan Sandy, 3.799 onder trend en seizoen. De forecast voor maandag 31 december, de laatste trend, van vrijdag 28 december, plus het seizoen van een maandag, is 1.476; geregistreerde fietsers huurden er 2.290. De laatste trend valt in de rustige week van kerst, en zo'n eenvoudige forecast kan niet weten dat de fietsers terugkomen. Hoe goed een forecast is, en hoe je hem op het verleden test, is de les *Een forecast testen op het verleden*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_moving_average():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een gecentreerd voortschrijdend gemiddelde met de hand

        `moving_average(y, half)` moet de trend van de array `y` geven: voor elke dag `i` het gemiddelde van de `half` dagen ervoor, dag `i` zelf en de `half` dagen erna, en NaN voor de eerste en de laatste `half` dagen, waar het window niet past. De eerste regels maken met `np.full` een array van NaN zo lang als `y`, en lopen langs de dagen waarvan het window past. Vervang de `...`. De check roept `moving_average(weeks, 3)` aan, acht weken geregistreerde fietsers vanaf maandag 2 januari 2012, en probeert er nog vijf: een korte verzonnen array met `half` 0, 1, 2 en 3, en vijf gelijke waarden. Hij vergelijkt met `y.rolling(2 * half + 1, center=True).mean()`.
        """),
        L.task("Laat `moving_average` het gemiddelde geven van het window rond elke dag."),
    ])
    return


@app.cell(hide_code=True)
def exercise_moving_average():
    box_moving_average = L.code_box("moving-average", lines=7)
    box_moving_average
    return (box_moving_average,)


@app.cell(hide_code=True)
def check_moving_average(box_moving_average):
    result_moving_average = L.check("moving-average", box_moving_average.value)
    result_moving_average
    return (result_moving_average,)


@app.cell(hide_code=True)
def hints_moving_average():
    L.hints(
        "moving-average",
        [
            "De trend van dag `i` is een gemiddelde over een stuk van `y`. Welke dagen zitten in dat stuk?",
            "Het stuk loopt van `half` dagen voor `i` tot `half` dagen erna, allebei meegeteld. Een slice `y[a:b]` stopt vlak voor `b`.",
            "`trend[i] = np.mean(y[i - half:___])`.",
        ],
        why="Met `half` 3 krijgt dag 10 het gemiddelde van de dagen 7 tot en met 13: `y[7:14]`, 7 waarden.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_season_of():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Het seizoen, plek voor plek

        `season_of(rest, period)` moet het seizoen geven: voor elke plek `k` van 0 tot en met `period - 1` het gemiddelde van `rest[k]`, `rest[k + period]`, `rest[k + 2 * period]` enzovoort, zonder NaN, met de `period` gemiddelden zo opgeschoven dat ze samen 0 zijn. Vervang de twee `...`. De check roept `season_of(rest, 7)` aan, met `rest` de geregistreerde fietsers min hun trend over 7 dagen, alle 731 dagen, NaN voor de eerste en de laatste 3. Plek 0 is een zaterdag, want 1 januari 2011 was er een. Hij probeert er nog vier korte arrays, twee met NaN en één met `period` 1.
        """),
        L.task("Laat `season_of` het gemiddelde geven van elke plek in de periode, opgeschoven tot ze samen 0 zijn."),
    ])
    return


@app.cell(hide_code=True)
def exercise_season_of():
    box_season_of = L.code_box("season-of", lines=7)
    box_season_of
    return (box_season_of,)


@app.cell(hide_code=True)
def check_season_of(box_season_of):
    result_season_of = L.check("season-of", box_season_of.value)
    result_season_of
    return (result_season_of,)


@app.cell(hide_code=True)
def hints_season_of():
    L.hints(
        "season-of",
        [
            "`means[k]` is een gemiddelde over elke `period`-de waarde van `rest`, vanaf plek `k`. Daarna schuift elk gemiddelde evenveel op, zodat ze samen 0 zijn.",
            "*Het seizoen* in *Het idee* heeft een slice die elke 7e waarde neemt, en een gemiddelde dat NaN weglaat. Zeven getallen die samen 0 zijn: wat trek je van elk af?",
            "`means[k] = np.nanmean(rest[___::period])` en `return means - np.mean(___)`.",
        ],
        why="`rest[0::7]` is elke zaterdag, `rest[1::7]` elke zondag. De zeven gemiddelden zijn −560,6, −760,2, 14,1, 290,0, 332,2, 410,0 en 270,3, met een gemiddelde van −0,6; dat eraf trekken geeft −560,0, −759,6, 14,7, 290,6, 332,8, 410,6 en 270,9, en die zijn samen 0.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_window():
    mo.md(r"""
    ### Oefening 3. Hoe lang een window
    """)
    return


@app.cell(hide_code=True)
def choice_window_cell():
    choice_window = L.choice(
        "window",
        question="De trend van de geregistreerde fietsers is een gecentreerd voortschrijdend gemiddelde over 7 dagen. Waarom 7, of een veelvoud van 7, zoals 21 of 35?",
        options={
            "right": "Alleen dan bevat elk window elke dag van de week even vaak, zodat de week uit de trend middelt",
            "long": "Een korter window is te hobbelig; elk window van 7 dagen of meer werkt",
            "middle": "7 is oneven, dus heeft elke dag een midden en kan het window gecentreerd worden",
        },
        correct="right",
        why={
            "long": "Rust is niet het punt: een window van 9 dagen bevat twee dagen van de week twee keer, en laat een deel van de week terug in de trend. Dan beslaat het seizoen 1.337 in plaats van 1.170. Zie de slider in *Het idee*.",
            "middle": "Een oneven window heeft wel een midden, maar 5 is ook oneven: een window van 5 dagen mist twee dagen van de week, de trend neemt een deel van de week op, en het seizoen beslaat 942 in plaats van 1.170. Zie de slider in *Het idee*.",
            "right": "Eén hele periode bevat elke plek één keer, dus heeft het gemiddelde erover geen seizoen meer. 21 en 35 dagen bevatten ook hele weken, en geven een rustigere trend.",
        },
    )
    choice_window
    return (choice_window,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De trend is het rolling gemiddelde uit [Tijd als index](/course/open/ts-time-index?lang=nl), naar het midden van zijn window geschoven, en het seizoen is een gemiddelde per groep, zoals bij `groupby`. De rest is voor een decompositie wat een residu is voor een model: wat het niet verklaart. De volgende les, *Lijkt vandaag op gisteren? Autocorrelatie*, vindt de periode van een reeks uit de reeks zelf; de forecast van trend plus seizoen wordt getest in *Een forecast testen op het verleden*.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_window, result_moving_average, result_season_of):
    L.wrap_up(result_moving_average, result_season_of, choice_window)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
