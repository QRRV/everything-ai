# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ts-time-index. Route: course/route/lessen/klassiek-ml.yaml. Checks en plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ts-time-index", "nl")
    C = L.checks   # het verkeer van elk uur met de tijd als index, de rolling gemiddelden van de slider, en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Project: je tabel, verkend en schoongemaakt](/course/open/ml-project-table-explore?lang=nl) en één over [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_hours():
    quick_hours = L.choice(
        "quick-hours",
        question="De verkeerstabel heeft 48.204 rijen maar maar 40.575 verschillende uren in de kolom `date_time`. Welke regel houdt één rij per uur over?",
        options={
            "all": "`df.drop_duplicates()`",
            "right": "`df.drop_duplicates(subset=\"date_time\")`",
            "na": "`df.dropna()`",
        },
        correct="right",
        why={
            "all": "Zonder kolom laat `drop_duplicates` alleen rijen weg die in elke kolom hetzelfde zijn: hier 17. De meeste dubbele uren verschillen in hun weer. Zie *Het idee* in [Project: je tabel, verkend en schoongemaakt](/course/open/ml-project-table-explore?lang=nl).",
            "na": "`dropna` laat rijen met een ontbrekende waarde weg; een dubbel uur heeft er geen. Zie *Het idee* in [Project: je tabel, verkend en schoongemaakt](/course/open/ml-project-table-explore?lang=nl).",
            "right": "Eén rij per uur, de eerste van elk. Vandaag worden die uren de index van de tabel.",
        },
    )
    quick_hours
    return (quick_hours,)


@app.cell(hide_code=True)
def quick_check_groupby():
    quick_groupby = L.choice(
        "quick-groupby",
        question="`df` heeft een kolom `weekday`, 0 voor maandag tot 6 voor zondag, en een kolom `traffic_volume`. Wat geeft `df.groupby(\"weekday\")[\"traffic_volume\"].mean()`?",
        options={
            "right": "Zeven getallen: het gemiddelde verkeer van elke dag van de week",
            "rows": "Een getal voor elke rij: het gemiddelde van zijn dag",
            "one": "Eén getal: het gemiddelde van al het verkeer",
        },
        correct="right",
        why={
            "rows": "`groupby` geeft één getal per groep, niet één per rij: zeven groepen, zeven getallen. Zie *Het idee* in [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).",
            "one": "Dat is `df[\"traffic_volume\"].mean()`. `groupby(\"weekday\")` verdeelt de rijen eerst in zeven groepen. Zie *Het idee* in [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl).",
            "right": "Eén getal per groep. Vandaag zijn de groepen stukken tijd, zoals dagen, en maakt pandas ze voor je.",
        },
    )
    quick_groupby
    return (quick_groupby,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De verkeerstabel van je project telt elk uur de voertuigen op Interstate 94, van oktober 2012 tot september 2018. Hoe ziet een gewone dag eruit? Is het op vrijdag drukker dan op maandag? Groeit het verkeer door de jaren heen? Elk van die vragen gaat over tijd: een maand eruit knippen, de uren optellen tot dagen, de hobbels van losse uren wegstrijken, een uur vergelijken met hetzelfde uur een week eerder. pandas doet dat allemaal, zodra de tijd de index van de tabel is.

    **Een voorbeeld uit het dagelijks leven.** Een stappenteller laat je stappen per uur, per dag en per week zien. Op een dag dat je je telefoon thuis laat, staat er 0 stappen. Je hebt wel gelopen; alleen telde niets ze. Een tijdreeks heeft dezelfde valkuil.
    """)
    return


@app.cell(hide_code=True)
def predict_zero_days():
    guess_zero = L.guess(
        "zero-days-guess",
        question="Met de tijd als index telt `s.resample(\"D\").sum()` de uren van elke dag op, van 2 oktober 2012 tot 30 september 2018: 2.190 dagen. Op hoeveel daarvan is het totaal 0?",
        options={
            "none": "Geen: een snelweg is nooit een hele dag leeg",
            "few": "Een stuk of tien, op feestdagen en in sneeuwstormen",
            "right": "Ongeveer 330",
        },
        reveal="**330.** Op 330 dagen heeft de tabel helemaal geen rij, en de som van niets is 0. Een dag die niemand telde, lijkt een dag zonder één voertuig. `s.resample(\"D\").mean()` geeft voor die dagen NaN, en `s.resample(\"D\").count()` zegt op hoeveel uren elke dag rust: 1.214 dagen hebben alle 24, en 646 maar een deel.",
    )
    guess_zero
    return (guess_zero,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Tijd als index.** Een **tijdreeks** is een getal dat moment na moment gemeten wordt, zoals het verkeer van elk uur. Zet in pandas de tijden in de index: `s = df.set_index("date_time")["traffic_volume"]` is een Series met een **datetime index**, één moment per waarde. pandas leest datums dan zoals je ze schrijft: `s.loc["2017-10"]` is oktober 2017, 744 uren, en `s.loc["2017-10-02":"2017-10-08"]` is de week van maandag 2 tot zondag 8 oktober, beide dagen meegerekend.

        **Elk uur op zijn plek.** `s.asfreq("h")` geeft elk uur van het eerste tot het laatste, NaN waar de tabel geen rij heeft: 52.551 uren, waarvan 11.976 NaN. Een ontbrekend uur valt nu op in plaats van zich te verstoppen.

        **Resampling.** `s.resample("D")` verdeelt de waarden over bakjes van een dag, zoals `groupby` rijen over groepen verdeelt; `.mean()`, `.sum()` of `.count()` geeft dan één getal per dag. `"W"` maakt weken. Dat is **resampling**.

        **Gladstrijken.** Een **rolling window** geeft voor elk uur een getal over de uren vlak ervoor: `s.rolling("24h").mean()` is het gemiddelde van de laatste 24 uur, dat uur meegerekend. Het strijkt de hobbels van losse uren weg, en het loopt achter: het kent alleen het verleden. Een **exponentieel voortschrijdend gemiddelde** houdt één niveau bij en schuift dat bij elke nieuwe waarde een stap die kant op: nieuw niveau = α × nieuwe waarde + (1 − α) × oud niveau. Met α = 0,5 weegt de nieuwste waarde 0,5, die ervoor 0,25, die daarvoor 0,125: oude waarden vervagen, maar verdwijnen nooit helemaal. In pandas: `s.ewm(alpha=0.3, adjust=False).mean()`.

        **Opschuiven.** `s.shift(1)` schuift elke waarde één rij later, zodat elke rij de rij ervoor ziet. Op `s.asfreq("h")`, waar elke rij een uur is, is `shift(24)` hetzelfde uur een dag eerder.

        **Tijdzones.** De tijden in de tabel zijn plaatselijke tijden in Minneapolis. Twee keer per jaar springt de klok. In de nacht van 12 maart 2017 ging die van 1:59 meteen naar 3:00: de tabel heeft geen rij om 2:00, en `asfreq("h")` zet daar een NaN, op een uur dat nooit bestond. Op 6 november 2016 ging de klok van 1:59 terug naar 1:00: 1:00 gebeurde twee keer, en de tabel heeft er één rij voor. `s.tz_localize("America/Chicago")` vertelt pandas de **tijdzone**; op deze tabel stopt het met een fout, *Cannot infer dst time from 2012-11-04 01:00:00*: ook in die nacht gebeurde 1:00 twee keer, en pandas kan niet zien welke van de twee de rij is. Tijden in UTC, de tijd van de nulmeridiaan, springen nooit; `tz_convert("UTC")` zet gelokaliseerde tijden daarin om.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **tijdreeks** | een getal dat moment na moment gemeten wordt, zoals het verkeer van elk uur | |
        | **datetime index** | een index van momenten in de tijd, zodat pandas op tijd kan knippen, verdelen en opschuiven | `df.set_index("date_time")` |
        | `.loc` met een datum | de rijen van dat jaar, die maand, die dag of dat stuk, beide einden meegerekend | `s.loc["2017-10"]` |
        | `asfreq` | elke stap van de eerste tot de laatste, NaN waar geen rij was | `s.asfreq("h")` |
        | **resampling** | de waarden over bakjes tijd verdelen, zoals dagen, en één getal per bakje geven | `s.resample("D").mean()` |
        | **rolling window** | het stuk tijd vlak voor elk moment, hier om een gemiddelde over te nemen | `s.rolling("24h").mean()` |
        | **exponentieel voortschrijdend gemiddelde** | een niveau dat bij elke nieuwe waarde een deel α die kant op schuift | `s.ewm(alpha=0.3, adjust=False).mean()` |
        | α | het deel van de nieuwe waarde in het nieuwe niveau, tussen 0 en 1: groter volgt sneller | `alpha` |
        | `shift` | elke waarde zoveel rijen later gezet | `s.shift(24)` |
        | **tijdzone** | de regel die van een moment de plaatselijke kloktijd maakt, sprongen meegerekend | `s.tz_localize("America/Chicago")` |
        """),
        L.watch_out("Een window van 24 rijen is geen dag. `s.rolling(24)` neemt de laatste 24 rijen, en waar uren geen rij hebben, reiken die rijen verder terug: voor 11.503 van de 40.575 uren beslaan de laatste 24 rijen meer dan een dag, tot 320 dagen toe. `s.rolling(\"24h\")` neemt de uren van de laatste 24 uur, hoeveel daarvan er ook een rij hebben. Voor `shift` geldt hetzelfde: op de tabel zelf is `shift(24)` 24 rijen terug; alleen op `s.asfreq(\"h\")` is het een dag."),
        mo.md(r"""
        Het plaatje laat vier hele weken verkeer zien, van maandag 2 tot zondag 29 oktober 2017, zonder ontbrekend uur. De dunne grijze lijn is elk uur; de dikke lijn is `s.rolling(...).mean()` over het window dat jij kiest. Schuif met de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_window_cell():
    def _frame(hours):
        said = C.said(hours, L.lang)
        alt = ("Een lijndiagram van het verkeer op Interstate 94 per uur van maandag 2 tot zondag 29 oktober 2017, tussen 0 en "
               "7.500 voertuigen, met een streepje bij elke maandag, en het rolling gemiddelde over het gekozen window als dikkere lijn. " + said)
        return (C.picture(hours, {"title": "Verkeer, 2 tot 29 oktober 2017", "ylabel": "voertuigen per uur",
                                  "days": ["ma 2 okt", "ma 9 okt", "ma 16 okt", "ma 23 okt"], "alt": alt}), said)

    explore_window = L.explore("window", "window (uren)", _frame, start=24, values=C.WINDOWS, role="hyper",
                               task="Ga van een window van 1 uur naar 168. Bij welk window is de vorm van een dag verdwenen, en wat blijft er over?")
    explore_window
    return


@app.cell(hide_code=True)
def text_after_window():
    mo.md(r"""
    Elk window strijkt weg wat korter is dan het window zelf. Bij 3 uur blijven de ochtend- en de middagpiek, lager en een uur later. Bij 24 uur, een hele dag, is van de dag niets meer te zien: wat overblijft is de week, laag op zaterdag en zondag. Bij 168 uur, een hele week, blijft ook van de week bijna niets over. Welk window je nodig hebt, hangt af van je vraag: de vorm van een dag vraagt uren, een trend over de jaren weken of meer.

    ### Recept

    Om met een tijdreeks te werken in pandas:

    1. Maak van de tijd een tijd met `pd.to_datetime`, en zet hem in de index met `set_index`.
    2. Zet elke stap op zijn plek met `asfreq`, zoals `s.asfreq("h")`: een ontbrekend uur wordt NaN, dus je ziet het.
    3. Resample naar de stap die je vraag nodig heeft, met een gemiddelde in plaats van een som, en tel op hoeveel waarden elk bakje rust.
    4. Strijk glad met een window in tijd, `rolling("24h")`, of met `ewm`, en vergelijk met eerder met `shift` op het rooster van `asfreq`.
    5. Ken de tijdzone van je tijden: plaatselijke tijden springen twee keer per jaar.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel maakt de reeks, knipt oktober 2017 eruit, zet elk uur op zijn plek, resamplet naar dagen en telt hun uren, en vergelijkt maandag 2 oktober om 8:00 met hetzelfde uur een dag eerder. Als laatste print hij het gemiddelde van de laatste 24 uur en het exponentieel voortschrijdend gemiddelde met α = 0,3, allebei op maandag om 23:00. De opmerkingen met `#` zeggen welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_series():
    import pandas as pd

    # 1. de tijd als index, één rij per uur
    _df = pd.read_csv("data/metro-interstate-traffic.csv.gz")
    _df = _df.drop_duplicates(subset="date_time")
    _df["date_time"] = pd.to_datetime(_df["date_time"])
    _s = _df.set_index("date_time")["traffic_volume"]
    print(len(_s.loc["2017-10"]))

    # 2. elk uur op zijn plek
    _hours = _s.asfreq("h")
    print(len(_hours), _hours.isna().sum())

    # 3. per dag: het gemiddelde, en op hoeveel uren het rust
    _days = _s.resample("D").mean()
    _counts = _s.resample("D").count()
    print(len(_days), _days.isna().sum(), (_counts == 24).sum())

    # 4. een dag eerder, en twee manieren om glad te strijken
    _day_ago = _hours.shift(24)
    print(_hours.loc["2017-10-02 08:00"], _day_ago.loc["2017-10-02 08:00"])
    print(round(_s.rolling("24h").mean().loc["2017-10-02 23:00"]), round(_s.ewm(alpha=0.3, adjust=False).mean().loc["2017-10-02 23:00"]))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Oktober 2017 heeft al zijn 744 uren. Van de 52.551 uren van het eerste tot het laatste zijn er 11.976 NaN; van de 2.190 dagen hebben er 330 geen enkel uur en 1.214 alle 24. Om 8:00 op maandag 2 oktober reden er 6.054 voertuigen voorbij; een dag eerder, op zondag, 2.403. De laatste 24 uur tot maandag 23:00 zijn maandag zelf, dus hun gemiddelde, 3.443, is het gemiddelde van die dag. Het exponentieel voortschrijdend gemiddelde is 2.098: het laat de rustige late avond het zwaarst wegen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_week_change():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een week eerder

        `week_change(s)` moet voor elk uur geven hoeveel voertuigen er meer voorbijreden dan in hetzelfde uur een week eerder: NaN waar een van beide uren geen rij heeft, en voor de eerste week, die geen week ervoor heeft. `s` is een Series van uren met de tijd als index. Vervang de twee `...`. De check roept `week_change(may)` aan, de vier weken vanaf maandag 2 mei 2016, 672 uren waarvan er 654 een rij hebben, en probeert er nog drie: twee hele weken van oktober 2017, vier dagen, en tien verzonnen dagen waarin vier uren ontbreken.
        """),
        L.task("Laat `week_change` het verschil geven met hetzelfde uur een week eerder."),
    ])
    return


@app.cell(hide_code=True)
def exercise_week_change():
    box_week_change = L.code_box("week-change", lines=4)
    box_week_change
    return (box_week_change,)


@app.cell(hide_code=True)
def check_week_change(box_week_change):
    result_week_change = L.check("week-change", box_week_change.value)
    result_week_change
    return (result_week_change,)


@app.cell(hide_code=True)
def hints_week_change():
    L.hints(
        "week-change",
        [
            "Zorg eerst dat één rij één uur is, ook waar de tabel geen rij heeft. Trek dan van elk uur het uur een week ervoor af.",
            "*Elk uur op zijn plek* in *Het idee* zegt hoe je elk uur krijgt, NaN waar geen rij is. Een week heeft 7 × 24 uur.",
            "`hours = s.asfreq(___)` en `return hours - hours.shift(___)`.",
        ],
        why="In mei 2016 hebben 18 van de 672 uren geen rij. Op het rooster van `asfreq(\"h\")` is 168 rijen terug precies een week terug; op de tabel zelf niet, overal waar er een uur tussen ontbreekt.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_smooth():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een exponentieel voortschrijdend gemiddelde met de hand

        `smooth(values, alpha)` moet na elke waarde van de lijst `values` het exponentieel voortschrijdend gemiddelde geven, als lijst: het niveau begint bij de eerste waarde, en schuift bij elke waarde daarna een deel `alpha` die kant op. Vervang de `...` door het nieuwe niveau. De check roept `smooth([10, 20, 20, 0], 0.4)` aan, en probeert er nog vier: de 24 uren van maandag 2 oktober 2017 met `alpha` 0,3, één waarde, `alpha` 1, en een niveau dat naar 0 zakt. Hij vergelijkt met wat pandas geeft met `ewm(alpha=alpha, adjust=False).mean()`.
        """),
        L.task("Laat `smooth` het niveau bij elke nieuwe waarde een deel `alpha` die kant op schuiven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_smooth():
    box_smooth = L.code_box("smooth", lines=7)
    box_smooth
    return (box_smooth,)


@app.cell(hide_code=True)
def check_smooth(box_smooth):
    result_smooth = L.check("smooth", box_smooth.value)
    result_smooth
    return (result_smooth,)


@app.cell(hide_code=True)
def hints_smooth():
    L.hints(
        "smooth",
        [
            "Het nieuwe niveau bestaat uit twee delen: iets van de nieuwe waarde, en de rest van het oude niveau. `alpha` zegt hoeveel van de nieuwe waarde.",
            "In de lus is de nieuwe waarde `x` en het oude niveau `level`. *Gladstrijken* in *Het idee* geeft de regel in woorden.",
            "`level = alpha * ___ + (1 - alpha) * ___`.",
        ],
        why="Het niveau begint bij 10. Dan 0,4 · 20 + 0,6 · 10 = 14, 0,4 · 20 + 0,6 · 14 = 16,4, en 0,4 · 0 + 0,6 · 16,4 = 9,84: `[10, 14, 16.4, 9.84]`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_spring():
    mo.md(r"""
    ### Oefening 3. Een uur dat nooit bestond
    """)
    return


@app.cell(hide_code=True)
def choice_spring_cell():
    choice_spring = L.choice(
        "spring",
        question="In de nacht van 12 maart 2017 gaat de tabel van 1:00 meteen naar 3:00, en `s.asfreq(\"h\")` zet een NaN op 2:00. Hoeveel verkeer miste het meetpunt om 2:00?",
        options={
            "right": "Geen: die nacht ging de klok van 1:59 meteen naar 3:00, dus 2:00 bestond niet in Minneapolis",
            "sensor": "Een uur verkeer: het meetpunt miste het, zoals de andere losse uren zonder rij",
            "duplicate": "Geen: de rij is weggelaten omdat hij een uur herhaalde",
        },
        correct="right",
        why={
            "sensor": "De meeste ontbrekende uren zijn uren die niemand telde, maar deze niet: op de tweede zondag van maart springt de klok in Minneapolis van 1:59 naar 3:00. In plaatselijke tijd bestaat 2:00 die nacht niet, en `asfreq(\"h\")` kan dat niet weten. Zie *Tijdzones* in *Het idee*.",
            "duplicate": "De eerste rij van elk uur houden laat een tweede rij van hetzelfde uur weg; de eerste blijft. Om 2:00 is er nooit een rij geweest, want die nacht sprong de klok van 1:59 naar 3:00. Zie *Tijdzones* in *Het idee*.",
            "right": "In plaatselijke tijd bestond 2:00 die nacht niet. `asfreq(\"h\")` telt uren op een klok zonder sprongen, dus voegt het een uur toe dat nooit bestond. In UTC heeft dezelfde nacht geen gat.",
        },
    )
    choice_spring
    return (choice_spring,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Resampling is de `groupby` uit [Eén getal per groep: groupby](/course/open/data-groupby?lang=nl) met groepen van tijd, en `asfreq` laat als NaN de 11.976 uren zien die je telde in [Project: je tabel, verkend en schoongemaakt](/course/open/ml-project-table-explore?lang=nl). Het rolling gemiddelde over één seizoen komt terug in de volgende les, *Trend, seizoen en de rest*, waar het de trend van een reeks vindt; `shift` komt terug wanneer een reeks met zichzelf een stap eerder vergeleken wordt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_spring, result_smooth, result_week_change):
    L.wrap_up(result_week_change, result_smooth, choice_spring)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
