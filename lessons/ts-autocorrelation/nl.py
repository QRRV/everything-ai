# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ts-autocorrelation. Route: course/route/lessen/klassiek-ml.yaml. Checks en plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ts-autocorrelation", "nl")
    C = L.checks   # de fietsers van elk uur op een rooster van uren, hun autocorrelatie bij elke lag, en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Trend, seizoen en de rest](/course/open/ts-trend-season?lang=nl) en één over [Twee kolommen samen: een verband is geen oorzaak](/course/open/data-two-columns?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_season():
    quick_season = L.choice(
        "quick-season",
        question="`rest` is een reeks dagen min haar trend, en de eerste dag is een maandag. Wat is `np.nanmean(rest[2::7])`?",
        options={
            "right": "Het gemiddelde van alle woensdagen: het seizoen van woensdag, voordat de zeven gemiddelden worden opgeschoven tot ze samen 0 zijn",
            "stretch": "Het gemiddelde van de dagen op plek 2 tot 7",
            "tuesday": "Het gemiddelde van alle dinsdagen",
        },
        correct="right",
        why={
            "stretch": "`rest[2:7]` zou één stuk zijn. `::7` neemt stappen: plek 2, 9, 16 enzovoort, elke 7e dag vanaf plek 2. Zie *Het seizoen* in *Het idee* in [Trend, seizoen en de rest](/course/open/ts-trend-season?lang=nl).",
            "tuesday": "Plekken tellen vanaf 0: plek 0 is de maandag, 1 de dinsdag, 2 de woensdag. Zie *Het seizoen* in *Het idee* in [Trend, seizoen en de rest](/course/open/ts-trend-season?lang=nl).",
            "right": "Elke 7e dag vanaf plek 2: alle woensdagen. Vandaag wordt een reeks vergeleken met zichzelf een aantal stappen eerder, en komt dezelfde week weer tevoorschijn.",
        },
    )
    quick_season
    return (quick_season,)


@app.cell(hide_code=True)
def quick_check_association():
    quick_association = L.choice(
        "quick-association",
        question="Op warme dagen verkoopt de ijskraam meer, en worden er meer fietsen gehuurd. Laat een fiets huren mensen ijs kopen?",
        options={
            "right": "Dat kun je hieruit niet zeggen: warm weer stuurt allebei, dus bewegen ze hoe dan ook samen",
            "yes": "Ja: de twee kolommen bewegen samen, dus veroorzaakt de een de ander",
            "never": "Nee: twee kolommen die samen bewegen, hebben nooit iets met elkaar te maken",
        },
        correct="right",
        why={
            "yes": "Samen bewegen is geen veroorzaken: een derde kolom, hier het weer, kan allebei sturen. Zie *Het idee* in [Twee kolommen samen](/course/open/data-two-columns?lang=nl).",
            "never": "Kolommen die samen bewegen, hebben vaak wel iets met elkaar te maken, soms via een derde kolom. Alleen zegt samen bewegen niet wat wat veroorzaakt. Zie *Het idee* in [Twee kolommen samen](/course/open/data-two-columns?lang=nl).",
            "right": "Het weer is een confounder. Vandaag zegt één getal hoe sterk twee kolommen samen bewegen, en daarna wordt een reeks naast zichzelf een paar stappen eerder gezet.",
        },
    )
    quick_association
    return (quick_association,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Om te voorspellen hoeveel fietsen geregistreerde fietsers het komende uur huren, kijk je natuurlijk naar het verleden. Maar naar welk deel? Het uur ervoor, hetzelfde uur gisteren, of hetzelfde uur vorige week? Het antwoord zit in hoeveel een reeks op zichzelf lijkt, een aantal stappen eerder. Dat laat ook het seizoen van een reeks zien zonder dat je de periode hoeft te weten, en het waarschuwt als een trend alles op elkaar laat lijken.

    **Een voorbeeld uit het dagelijks leven.** Wordt het morgen warm? De temperatuur van vandaag is een goede gok: het weer verandert langzaam, dus lijkt een dag op de dag ervoor. Of het morgen regent, is veel moeilijker uit vandaag te raden. Sommige reeksen onthouden hun verleden goed, andere nauwelijks.
    """)
    return


@app.cell(hide_code=True)
def predict_earlier_hour():
    guess_hour = L.guess(
        "earlier-hour-guess",
        question="Zet voor elk uur van 2011 en 2012 de fietsen die geregistreerde fietsers huurden naast de fietsen die ze een aantal uur eerder huurden. Welk eerder uur beweegt het sterkst samen met het uur nu?",
        options={
            "hour": "Het uur ervoor",
            "day": "Hetzelfde uur een dag eerder",
            "right": "Hetzelfde uur een week eerder",
        },
        reveal="**Hetzelfde uur een week eerder.** Met het getal van deze les, r, is het 0,886 voor een week eerder, 0,815 voor een dag eerder en 0,802 voor een uur eerder. Het uur ervoor ligt dicht bij in de tijd, maar op een werkdag klimmen de fietsers 's ochtends snel: gemiddeld 98 om 6:00, 278 om 7:00 en 455 om 8:00. Maandag 8:00 vorige week was ook een maandagochtend.",
    )
    guess_hour
    return (guess_hour,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **De correlatiecoëfficiënt.** Voor paren getallen, zoals de temperatuur en de fietsers van elke dag, zegt de **correlatiecoëfficiënt** r hoe goed een rechte lijn erbij past, stijgend of dalend:

        $$r = \frac{\operatorname{mean}\big((x - \operatorname{mean}(x)) \cdot (y - \operatorname{mean}(y))\big)}{\operatorname{std}(x) \cdot \operatorname{std}(y)}$$

        Elk paar telt het product van zijn afstanden tot de twee gemiddelden mee. Dat product is positief als beide waarden boven hun gemiddelde liggen of allebei eronder, en negatief als de een erboven ligt en de ander eronder. Delen door de twee standaardafwijkingen maakt r vrij van eenheden: het ligt tussen −1 en 1. r = 1 betekent dat elk paar op één stijgende lijn ligt, −1 op één dalende lijn, en 0 dat er geen rechte lijn past, wat nog steeds een kromme kan zijn: `x` en `x ** 2` van −3 tot 3 geven 0. Temperatuur en geregistreerde fietsers geven 0,540. In NumPy geeft `np.corrcoef(x, y)` een tabel van 2 bij 2 correlaties, van `x` en `y` met zichzelf en met elkaar, dus is `np.corrcoef(x, y)[0, 1]` gelijk aan r.

        **Lag.** Een **lag** van `k` stappen is de waarde `k` stappen eerder, wat `y.shift(k)` naast elke waarde zet.

        **Autocorrelatie.** De **autocorrelatie** bij lag `k` is de correlatie van een reeks met zichzelf `k` stappen eerder: elke waarde vanaf plek `k`, gekoppeld aan de waarde `k` stappen ervoor. In pandas: `y.autocorr(k)`. Teken haar voor elke lag als een staaf, en het seizoen laat zich zien: voor de fietsers van elk uur pieken de staven bij 24 uur, 0,815, en nog hoger bij 168 uur, een week, 0,886, terwijl 12 uur, van dag naar nacht, −0,021 geeft.

        **Een trend laat elke lag op elkaar lijken.** De geregistreerde fietsers van elke dag hebben een autocorrelatie tussen 0,57 en 0,83 bij elke lag van 1 tot 28 dagen: een dag in een drukke maand ligt dicht bij alle dagen eromheen. De rest uit de vorige les, de fietsers min hun trend, laat de week zien: 0,390 bij 7 dagen en −0,501 bij 3.

        **Stationariteit.** Een reeks is **stationair** als haar gemiddelde en haar spreiding over de tijd gelijk blijven: elk stuk ervan lijkt op elk ander stuk. De geregistreerde fietsers zijn dat niet: ze huurden in 2011 gemiddeld 2.728 fietsen per dag, en in 2012 4.581. De meeste methoden om te voorspellen gaan uit van een stationaire reeks, en daarom gaan de trend en het seizoen er eerst af.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **correlatiecoëfficiënt** | hoe goed een rechte lijn bij paren getallen past, van −1 (dalend) via 0 (geen lijn) tot 1 (stijgend) | `np.corrcoef(x, y)[0, 1]` |
        | r | de correlatiecoëfficiënt | |
        | **lag** | een aantal stappen terug in de tijd | `y.shift(k)` |
        | **autocorrelatie** | de correlatie van een reeks met zichzelf een lag eerder | `y.autocorr(k)` |
        | **stationair** | een reeks waarvan het gemiddelde en de spreiding over de tijd gelijk blijven | |
        | `np.corrcoef` | de correlaties van twee arrays, met zichzelf en met elkaar, als tabel van 2 bij 2 | `np.corrcoef(x, y)` |
        """),
        L.watch_out("Een lag van `k` rijen is alleen op een rooster een lag van `k` stappen. De uren van de fietsdata hebben 165 gaten; in de tabel zelf ligt een rij 168 rijen terug alleen een week terug als er geen uur tussenin ontbreekt. Op de tabel geeft `autocorr(168)` 0,823; op `asfreq(\"h\")`, waar elk uur zijn plek heeft, 0,886. Zet een reeks op haar rooster voor je haar met haar verleden vergelijkt."),
        mo.md(r"""
        Links is elke stip één uur: de fietsers `k` uur eerder naar rechts, en de fietsers in dat uur omhoog. Rechts de autocorrelatie bij elke lag tot twee weken, de lag die je kiest dik getekend. Schuif met de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_lag_cell():
    def _frame(lag):
        said = C.said(lag, L.lang)
        alt = ("Links een scatter plot van de fietsen die geregistreerde fietsers in een uur huurden tegen de fietsen die ze het "
               "gekozen aantal uur eerder huurden, allebei tussen 0 en 900. Rechts de autocorrelatie als dunne staaf voor elke lag van 1 tot "
               "336 uur, met pieken bij elke 24 uur en de hoogste bij 168, de gekozen lag dik getekend. " + said)
        return (C.picture(lag, {"left": "Fietsers nu en k uur eerder", "right": "Autocorrelatie", "xlabel": "k uur eerder",
                                "ylabel": "nu", "lag": "lag (uren)", "alt": alt}), said)

    explore_lag = L.explore("lag-hours", "lag (uren)", _frame, start=24, values=C.LAGS, role="hyper",
                            task="Ga van een lag van 1 uur naar 336. Bij welke lags liggen de stippen dicht bij een lijn, en bij welke zijn ze een wolk?")
    explore_lag
    return


@app.cell(hide_code=True)
def text_after_lag():
    mo.md(r"""
    Bij 1 uur liggen de stippen dicht bij een lijn, bij 3 uur waaieren ze uit, en bij 6 en 12 uur zijn ze een wolk: r is ongeveer 0. Dan komt de dag terug: bij 24 uur is r weer 0,815, bij 167 uur 0,717, en bij 168, een hele week, 0,886, de hoogste van alle lags tot twee weken. 336 uur, twee weken, is bijna even hoog: 0,884. De pieken van de staven bij elke 24 uur en elke 168 zijn de twee seizoenen van deze reeks, de dag en de week.

    ### Recept

    Zo zie je hoe een reeks samenhangt met haar verleden:

    1. Zet de reeks op een rooster van gelijke stappen, zoals `asfreq("h")`, zodat een lag van `k` rijen `k` stappen is.
    2. Bereken de autocorrelatie voor elke lag tot een paar periodes: `y.autocorr(k)`.
    3. Pieken bij een lag en de veelvouden ervan laten een seizoen met die periode zien.
    4. Een autocorrelatie die bij elke lag hoog blijft, wijst op een trend: haal de trend eraf en kijk opnieuw.
    5. Vergelijk het gemiddelde en de spreiding van stukken, zoals jaren, om te zien of de reeks stationair is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel zet de fietsers van elk uur op een rooster van uren: `pd.to_timedelta(_h["hr"], unit="h")` maakt van de kolom met uurnummers tijdsduren, die bij de datums worden opgeteld. Ze print de autocorrelatie bij 1, 24 en 168 uur. Daarna neemt ze de fietsers van elke dag en hun rest na een trend over 7 dagen, en print van allebei de autocorrelatie bij 3 en 7 dagen. Als laatste vergelijkt ze 2011 met 2012: de eerste 365 dagen en de rest, hun gemiddelden en de standaardafwijkingen van de rest. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_autocorrelation():
    import pandas as pd

    # 1. de fietsers van elk uur, op een rooster van uren
    _h = pd.read_csv("data/bike-sharing-hour.csv.gz")
    _when = pd.to_datetime(_h["dteday"]) + pd.to_timedelta(_h["hr"], unit="h")
    _s = _h.set_index(_when)["registered"].asfreq("h")

    # 2. en 3. de autocorrelatie een uur, een dag en een week terug
    print(round(_s.autocorr(1), 3), round(_s.autocorr(24), 3), round(_s.autocorr(168), 3))

    # 4. de fietsers van elke dag, met en zonder hun trend
    _y = pd.read_csv("data/bike-sharing-day.csv.gz")["registered"]
    _rest = _y - _y.rolling(7, center=True).mean()
    print(round(_y.autocorr(3), 3), round(_y.autocorr(7), 3))
    print(round(_rest.autocorr(3), 3), round(_rest.autocorr(7), 3))

    # 5. stationair? het gemiddelde van de fietsers en de spreiding van de rest, 2011 tegen 2012
    print(round(_y[:365].mean()), round(_y[365:].mean()))
    print(round(_rest[:365].std()), round(_rest[365:].std()))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Een uur, een dag en een week terug geven 0,802, 0,815 en 0,886. De fietsers van elke dag geven 0,646 bij 3 dagen en 0,764 bij 7: de trend houdt allebei hoog. Zonder de trend geeft 3 dagen −0,501 en 7 dagen 0,390: een donderdag lijkt niet op de zondag 3 dagen later, en wel op de donderdag een week later. Het gemiddelde groeide van 2.728 per dag in 2011 naar 4.581 in 2012. De trend eraf halen zet het gemiddelde vast, maar de spreiding van de rest groeide ook, van 529 naar 901: de rest is ook niet stationair.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_correlation():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De correlatiecoëfficiënt met de hand

        `correlation(x, y)` moet de correlatiecoëfficiënt geven van twee even lange arrays. De eerste regels geven de afstanden van elke waarde tot haar gemiddelde, `dx` en `dy`. Vervang de `...` door r. De check roept `correlation(temp, riders)` aan, de temperatuur in °C en de geregistreerde fietsers van alle 731 dagen, en probeert er nog vier: een stijgende lijn, een dalende lijn, de kom van `x` en `x ** 2`, en vijf verzonnen paren. Hij vergelijkt met `np.corrcoef`.
        """),
        L.task("Laat `correlation` r geven: hoe goed een rechte lijn bij de paren past."),
    ])
    return


@app.cell(hide_code=True)
def exercise_correlation():
    box_correlation = L.code_box("correlation", lines=6)
    box_correlation
    return (box_correlation,)


@app.cell(hide_code=True)
def check_correlation(box_correlation):
    result_correlation = L.check("correlation", box_correlation.value)
    result_correlation
    return (result_correlation,)


@app.cell(hide_code=True)
def hints_correlation():
    L.hints(
        "correlation",
        [
            "Boven de streep van de formule: één gemiddelde, van producten. Eronder: twee standaardafwijkingen.",
            "De producten zijn `dx * dy`, één per paar. `np.std` geeft een standaardafwijking.",
            "`return np.mean(___) / (np.std(x) * ___)`.",
        ],
        why="Voor de temperatuur en de fietsers is het gemiddelde van de producten 6.314,8, en zijn de standaardafwijkingen 7,5 °C en 1.559,2 fietsers: 6.314,8 / (7,5 × 1.559,2) = 0,540.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_autocorrelation():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De autocorrelatie bij lag k

        `autocorrelation(y, k)` moet de autocorrelatie van de array `y` bij lag `k` geven: `now` is elke waarde vanaf plek `k`, `before` de waarde `k` stappen voor elk daarvan, en de laatste regel geeft de correlatie van die twee. Vervang de twee `...` door slices. De check roept `autocorrelation(riders, 7)` aan, de geregistreerde fietsers van alle 731 dagen, en probeert er nog vijf: lag 1 van de fietsers, een stijgende reeks, een reeks die elke stap op en neer gaat bij lag 1 en 2, en lag 0. Hij vergelijkt met wat pandas geeft, `pd.Series(y).autocorr(k)`.
        """),
        L.task("Laat `autocorrelation` elke waarde koppelen aan de waarde `k` stappen eerder."),
    ])
    return


@app.cell(hide_code=True)
def exercise_autocorrelation():
    box_autocorrelation = L.code_box("autocorrelation", lines=6)
    box_autocorrelation
    return (box_autocorrelation,)


@app.cell(hide_code=True)
def check_autocorrelation(box_autocorrelation):
    result_autocorrelation = L.check("autocorrelation", box_autocorrelation.value)
    result_autocorrelation
    return (result_autocorrelation,)


@app.cell(hide_code=True)
def hints_autocorrelation():
    L.hints(
        "autocorrelation",
        [
            "`now` en `before` moeten even lang zijn, en `before[i]` moet de waarde `k` stappen voor `now[i]` zijn. De eerste `k` waarden hebben niets voor zich; de laatste `k` waarden liggen nooit `k` stappen voor een andere.",
            "`now` laat de eerste `k` waarden weg, `before` de laatste `k`. Een slice `y[a:b]` kan elk van beide einden weglaten; `len(y)` is waar `y` ophoudt.",
            "`now = y[___:]` en `before = y[:len(y) - ___]`.",
        ],
        why="Met `y` = [1, 2, 3, 4, 5] en `k` = 2 is `now` gelijk aan [3, 4, 5] en `before` aan [1, 2, 3], dus hoort 3 bij 1, 4 bij 2 en 5 bij 3.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_stationary():
    mo.md(r"""
    ### Oefening 3. Stationair of niet
    """)
    return


@app.cell(hide_code=True)
def choice_stationary_cell():
    choice_stationary = L.choice(
        "stationary",
        question="De rest van de geregistreerde fietsers, na de trend over 7 dagen, heeft in elk kwartaal van 2011 en 2012 een gemiddelde dicht bij 0. De standaardafwijking ervan groeit van 342 in het eerste kwartaal van 2011 naar 1.035 in het laatste van 2012. Is de rest stationair?",
        options={
            "right": "Nee: de spreiding verandert over de tijd, en een stationaire reeks houdt zowel haar gemiddelde als haar spreiding",
            "mean": "Ja: het gemiddelde blijft dicht bij 0",
            "trend": "Ja: de trend is weg, en dat is wat stationair betekent",
        },
        correct="right",
        why={
            "mean": "Een vast gemiddelde is de helft ervan. Een stationaire reeks houdt ook haar spreiding, en deze spreiding verdrievoudigt, van 342 naar 1.035: eind 2012 liggen de dagen veel verder van hun trend dan begin 2011. Zie *Stationariteit* in *Het idee*.",
            "trend": "De trend eraf halen zet het gemiddelde vast, maar de spreiding van de rest groeit nog steeds over de twee jaar. Stationair vraagt allebei. Zie *Stationariteit* in *Het idee*.",
            "right": "De spreiding van de rest groeide over de twee jaar, van 342 in het eerste kwartaal tot 1.035 in het laatste. Een forecast die overal dezelfde spreiding aanneemt, is te zeker waar de spreiding groot is en te voorzichtig waar ze klein is.",
        },
    )
    choice_stationary
    return (choice_stationary,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De correlatiecoëfficiënt zet een getal bij de scatter plots uit [Twee kolommen samen](/course/open/data-two-columns?lang=nl), en hij zegt net zo weinig over oorzaak als zij. De autocorrelatie vindt de periode die [Trend, seizoen en de rest](/course/open/ts-trend-season?lang=nl) moest krijgen, en de rest uit die les is wat ze zonder trend leest. De volgende les, *Een forecast testen op het verleden*, maakt van de twee sterkste lags de twee eenvoudigste forecasts: de waarde een stap eerder, en de waarde een seizoen eerder.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_stationary, result_autocorrelation, result_correlation):
    L.wrap_up(result_correlation, result_autocorrelation, choice_stationary)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
