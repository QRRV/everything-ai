# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les ml-project-table-explore. Route: course/route/lessen/klassiek-ml.yaml. Checks, de tabel en het speelblok: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("ml-project-table-explore", "nl")
    C = L.checks   # de verkeerstabel, de feiten erover, de kalender van het speelblok, en de demo

    MONTHS = ["januari", "februari", "maart", "april", "mei", "juni", "juli", "augustus", "september", "oktober",
              "november", "december"]
    SHORT = ["jan", "feb", "mrt", "apr", "mei", "jun", "jul", "aug", "sep", "okt", "nov", "dec"]

    def names_in(names):
        """Namen in woorden: september, oktober en november."""
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" en {names[-1]}"


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Vragen stellen aan data](/course/open/data-explore?lang=nl) en [Gaten in een tabel: dropna en fillna](/course/open/data-missing-values?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_describe():
    quick_describe = L.choice(
        "quick-describe",
        question="`df.describe()` van een jaar weer zegt over de kolom `temp`, in graden Celsius: `count` 365, `mean` 11,2, `min` -999,0, `max` 34,1. Wat vertelt de `min` je?",
        options={
            "cold": "Dat het op één dag heel koud was",
            "right": "Dat een waarde niet kan kloppen: een code of een fout, die je bekijkt voordat je iets anders doet",
            "fine": "Niets: `min` en `max` laten alleen het bereik zien",
        },
        correct="right",
        why={
            "cold": "Geen plek waar mensen meten komt in de buurt van -999 °C: dat ligt ver onder het koudste dat er bestaat, -273,15 °C. Zo'n waarde is een code voor *niet gemeten*, of een fout. Zie [Vragen stellen aan data](/course/open/data-explore?lang=nl).",
            "fine": "Juist in het bereik zie je een waarde die niet kan het eerst. -999 °C ligt ver onder het koudste dat er bestaat, dus het is een code of een fout. Zie [Vragen stellen aan data](/course/open/data-explore?lang=nl).",
            "right": "In `min` en `max` zie je onmogelijke waarden het eerst. Vandaag laat `describe()` van een echte tabel er twee zien, en je schrijft voor elk een regel.",
        },
    )
    quick_describe
    return (quick_describe,)


@app.cell(hide_code=True)
def quick_check_fill():
    quick_fill = L.choice(
        "quick-fill",
        question="Een kolom met temperaturen heeft gaten, en je vult ze in met het gemiddelde van de kolom. Wat gebeurt er met het gemiddelde en de standaardafwijking?",
        options={
            "both": "Allebei blijven ze gelijk",
            "right": "Het gemiddelde blijft gelijk; de standaardafwijking wordt kleiner",
            "mean": "Het gemiddelde verandert; de standaardafwijking blijft gelijk",
        },
        correct="right",
        why={
            "both": "Het gemiddelde blijft gelijk, maar elke ingevulde waarde ligt precies op het gemiddelde, dus de waarden liggen minder ver eromheen verspreid en de standaardafwijking wordt kleiner. Zie [Gaten in een tabel: dropna en fillna](/course/open/data-missing-values?lang=nl).",
            "mean": "Waarden toevoegen die gelijk zijn aan het gemiddelde, laat het gemiddelde zoals het was. Wat verandert is de spreiding: elke nieuwe waarde ligt op het gemiddelde, dus de standaardafwijking wordt kleiner. Zie [Gaten in een tabel: dropna en fillna](/course/open/data-missing-values?lang=nl).",
            "right": "Invullen verandert de getallen, ook als het gemiddelde gelijk blijft. Vandaag kom je gaten van een andere soort tegen: hele uren zonder rij, waar geen cel is om in te vullen.",
        },
    )
    quick_fill
    return (quick_fill,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Een model leert van de tabel die je het geeft, fouten en al: het ziet geen verschil tussen een meting en een tikfout. Dus voordat er een model aan te pas komt, bekijk je de tabel. Met deze stap begint je project voor klassieke machine learning: één echte tabel, die je stap voor stap van een eerste blik naar een model brengt waar je op kunt vertrouwen. De tabel is het verkeer op Interstate 94, een snelweg in Minnesota, in westelijke richting, elk uur geteld door meetpunt 301 van het verkeersdepartement van die staat, ongeveer halverwege Minneapolis en Saint Paul, van oktober 2012 tot september 2018. Bij elk uur staat het weer van dat uur, en de feestdagen. John Hogue stelde de tabel samen en deelde hem in 2019 in de UCI Machine Learning Repository, onder de licentie CC BY 4.0, dus de course levert hem mee, met zijn naam erbij.

        Vandaag bekijk je hem. Je vindt temperaturen van 0 K, die niets ooit bereikt, een uur met meer regen dan er ooit ergens viel, uren die meer dan eens in de tabel staan, en maanden zonder één rij. Voor elk probleem schrijf je een regel, en je bewaart de regels in je project: elke latere stap begint met dezelfde schone tabel.

        **Een voorbeeld uit het dagelijks leven.** Je telt de bezoekers van een museum aan de hand van de lijst met kaartjes. Sommige kaartjes zijn twee keer gescand, de scanner was in de winter een week kapot, en op één dag staan 90.000 bezoekers. Je neemt niet het gemiddelde van die lijst zoals hij is: eerst herstel je wat te herstellen is, en je schrijft op wat niet te herstellen is.
        """),
        draw.project_steps(["Je tabel, verkend en schoongemaakt", "Bomen op je tabel, eerlijk getest"], 0,
                           here_label="← vandaag", title="Waar je project staat: je tabel"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_repeats():
    guess_repeats = L.guess(
        "repeats",
        question="De tabel heeft 48.204 rijen, voor elk uur één, zou je denken, van 2 oktober 2012 tot 30 september 2018. Hoeveel van die rijen herhalen een uur dat al een rij heeft?",
        options={
            "none": "Geen: elk uur staat er één keer in",
            "few": "Zo'n 20",
            "right": "Zo'n 7.600",
        },
        reveal="**7.629**, verdeeld over 5.445 uren; twee uren hebben er zelfs 6. Slechts 17 daarvan zijn in elke kolom gelijk aan een andere rij. De meeste herhalen een uur met een andere soort weer: om 7:00 op 10 oktober 2012 zegt de ene rij *Rain* en de andere *Drizzle*, met dezelfde temperatuur en hetzelfde verkeer.",
    )
    guess_repeats
    return (guess_repeats,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **De tabel.** Hij heeft 9 kolommen: `holiday`, de naam van een feestdag, alleen in het eerste uur van die dag; `temp`, de temperatuur in kelvin; `rain_1h` en `snow_1h`, de regen en de sneeuw in dat uur, in mm; `clouds_all`, hoeveel van de lucht bewolkt is, in %; `weather_main` en `weather_description`, het weer in één woord en in een paar; `date_time`, het uur, in plaatselijke tijd; en `traffic_volume`, hoeveel verkeer het meetpunt in dat uur in westelijke richting telde. Kelvin is een temperatuurschaal die begint bij het koudste dat er bestaat: 0 K is -273,15 °C, en graden Celsius zijn kelvin min 273,15.

        **Onmogelijke waarden.** `df.describe()` laat ze zien in zijn `min` en `max`. De laagste temperatuur is 0,00 K: 10 uren, tussen 03:00 en 08:00, op 31 januari en 2 februari 2014. De laagste daarna is 243,39 K, -29,76 °C, een koude decemberochtend in Minnesota. De meeste regen in een uur is 9.831,3 mm, om 17:00 op 11 juli 2016: bijna 10 meter water in één uur, hoger dan een huis. De meeste daarna is 55,63 mm. Geen van beide kan waar zijn, dus allebei gaan ze eruit, met voor elk een regel: een mask dat de rijen houdt die kunnen, zoals `df[df["temp"] > 0]`. Kies de grens uit wat mogelijk is, niet uit die ene rij die je zag: een grens van 100 mm regen in een uur ligt ver boven alles wat dit meetpunt mat, de 55,63 mm, en ver onder de 9.831,3 die niet kan. Een regel die zo geschreven is, werkt ook op uren die de tabel nog niet heeft.

        **Dubbele rijen.** In een tabel van uren hoort elk uur één keer te staan. Deze heeft 48.204 rijen, maar slechts 40.575 verschillende uren: 7.629 rijen herhalen een uur. `df.duplicated()` vindt er daarvan maar 17, omdat hij zoekt naar rijen die in elke kolom gelijk zijn. De meeste dubbele uren verschillen in hun weer: het meetpunt meldde twee soorten tegelijk, en elk kreeg een eigen rij. Noem de kolom die uniek hoort te zijn, en `df.duplicated(subset="date_time")` vindt ze alle 7.629. De regel van deze les houdt de eerste rij van elk uur, `df.drop_duplicates(subset="date_time")`: dan is de tweede soort weer van een uur weg, en dat is de prijs van één rij per uur.

        **Ontbrekende uren.** De pagina van de tabel in de UCI-repository zegt *Has Missing Values? No*. Voor de cellen klopt dat: `df.isna().sum()` geeft 0 voor elke kolom behalve `holiday`, waar pandas het woord `None`, die dag geen feestdag, als ontbrekend leest. Maar van het eerste uur tot het laatste zijn het 52.551 uren, en maar 40.575 daarvan hebben een rij: 11.976 uren hebben helemaal geen rij. `isna()` kan ze niet vinden, want er is geen cel die kan ontbreken. Je vindt ze door de uren van het eerste tot het laatste te tellen, en daarvoor moet `date_time` momenten in de tijd zijn in plaats van tekst: `pd.to_datetime(df["date_time"])`. Het langste gat is 307 dagen, van 8 augustus 2014 om 01:00 tot 11 juni 2015 om 20:00. Geen regel brengt die uren terug. Imputatie vult een cel in van een rij die er is; 307 dagen invullen zou betekenen dat je duizenden uren verkeer verzint die niemand telde.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **data cleaning** | de problemen van ruwe data verhelpen voordat je die gebruikt, zoals onmogelijke waarden, dubbele rijen en verkeerde types | |
        | **dubbele rij** | een rij die herhaalt wat maar één keer in de tabel hoort, zoals een tweede rij voor hetzelfde uur | `df.duplicated(subset="date_time")` |
        | `drop_duplicates` | de tabel zonder zijn dubbele rijen: van de rijen met dezelfde waarde in de kolom `subset` blijft alleen de eerste | `df.drop_duplicates(subset="date_time")` |
        | `pd.to_datetime` | een kolom met tekst zoals `"2012-10-02 09:00:00"` als momenten in de tijd, die je kunt vergelijken en tellen | `pd.to_datetime(df["date_time"])` |
        | `pd.date_range` | elk uur van `eerste` tot `laatste`, die twee meegeteld | `pd.date_range(eerste, laatste, freq="h")` |
        | **kelvin**, K | een temperatuurschaal die begint bij het koudste dat er bestaat: 0 K is -273,15 °C | `df["temp"] - 273.15` |
        """),
        L.watch_out("`df.duplicated()` en `df.drop_duplicates()` zonder kolom zoeken alleen naar rijen die in elke kolom gelijk zijn: hier 17. Twee rijen van hetzelfde uur met een andere soort weer zijn niet gelijk, dus noem de kolom die uniek hoort te zijn: `subset=\"date_time\"`."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Eén jaar van de tabel als kalender: een kolom per maand, een rij per dag. Hoe voller een dag, hoe meer van zijn 24 uren een rij hebben; een gestippelde dag heeft er geen. Een stip markeert een dag met een uur dat meer dan één rij heeft. Schuif door de jaren, van 2012 tot 2018.
    """)
    return


@app.cell(hide_code=True)
def explore_year_cell():
    _legend = ["alle 24 uren hebben een rij", "sommige uren hebben geen rij", "helemaal geen rij", "een uur met meer dan één rij"]

    def _frame(y):
        y = int(y)
        _f = C.year_facts(y)
        _since = {2012: " (vanaf 2 oktober, 09:00)", 2018: " (tot 30 september)"}.get(y, "")
        _gone = [MONTHS[m - 1] for m in _f["months_gone"]]
        _no = f" Helemaal geen rij in {names_in(_gone)}." if _gone else " Elke maand heeft rijen."
        _said = (f"**{y}**{_since}: {C.grouped(_f['have'], L.lang)} van zijn {C.grouped(_f['hours'], L.lang)} uren hebben een rij, "
                 f"en {C.grouped(_f['repeats'], L.lang)} rijen herhalen een uur.{_no}")
        _alt = (f"Een kalender van {y}: een kolom per maand en een rij per dag, elke dag gevuld naar het deel van zijn uren dat "
                f"een rij heeft, met een stip op de dagen met een uur dat meer dan één rij heeft. " + _said.replace("**", ""))
        return C.calendar(y, SHORT, _legend, f"{y}", _alt).text, _said

    explore_year = L.explore("year", "jaar", _frame, start=2015, values=C.YEARS, role="",
                             task="Zoek het gat van 306 lege dagen op rij. In welke jaren is geen enkele dag leeg?")
    explore_year
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Het grote gat loopt van augustus 2014 tot juni 2015; 63 van de 72 maanden hebben rijen. Eromheen blijven kleinere gaten over, van één ontbrekend uur tot tien dagen in 2013: in totaal 2.588 sprongen van meer dan een uur. En de stippen staan overal: elk jaar heeft honderden dubbele uren, de meeste met mist, regen of sneeuw, als het meetpunt meer dan één soort weer meldde.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Voordat er een model aan te pas komt, voor elke nieuwe tabel:

    1. Lees hem in en kijk: `df.describe()` voor waarden die niet kunnen, en `len(df)` tegenover het aantal verschillende waarden in de kolom die uniek hoort te zijn.
    2. Schrijf voor elke onmogelijke waarde een regel: een mask dat de rijen houdt die kunnen, zoals `df = df[df["temp"] > 0]`, met een grens die je kiest uit wat mogelijk is.
    3. Eén rij voor elk uur: `df.drop_duplicates(subset="date_time")` houdt van elk uur de eerste rij.
    4. Maak van tekst die een tijd is een tijd: `pd.to_datetime(df["date_time"])`, en zet een meting in een eenheid die je kunt lezen, zoals graden Celsius in plaats van kelvin. Tel de uren van het eerste tot het laatste, en vergelijk dat met de uren die een rij hebben.
    5. Tel wat elke regel overlaat, en schrijf op wat geen regel kan herstellen, zoals de ontbrekende uren, in de datasheet van de tabel.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel leest de tabel in, print de getallen van `describe()` voor drie kolommen, telt de onmogelijke waarden en de dubbele uren, en telt de uren van het eerste tot het laatste. Voer de cel uit.
    """)
    return


@app.cell
def example_look():
    import pandas as pd

    # 1. lees de tabel in, en bekijk drie van zijn kolommen
    _df = pd.read_csv("data/metro-interstate-traffic.csv.gz")
    print(len(_df))
    print(_df[["temp", "rain_1h", "traffic_volume"]].describe())

    # 2. de onmogelijke waarden: 0 K, en meer dan 100 mm regen in een uur
    print((_df["temp"] == 0).sum(), (_df["rain_1h"] > 100).sum())

    # 3. dubbele rijen: gelijk in elke kolom, en hetzelfde uur
    print(_df.duplicated().sum(), _df.duplicated(subset="date_time").sum())
    print(_df[_df["date_time"] == "2012-10-10 07:00:00"])

    # 4. de uren van het eerste tot het laatste, en de uren die een rij hebben
    _times = pd.to_datetime(_df["date_time"])
    _every = pd.date_range(_times.min(), _times.max(), freq="h")
    print(_times.min(), _times.max())
    print(len(_every), len(_times.drop_duplicates()))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `describe()` laat `min` `0.000000` zien voor `temp`, en `max` `9831.300000` voor `rain_1h`, met een punt als decimaalteken, zoals Python print: 10 uren bij 0 K en 1 uur met meer dan 100 mm regen. 17 rijen zijn in elke kolom gelijk aan een andere rij, en 7.629 herhalen een uur. De twee rijen van 10 oktober 2012 om 7:00 verschillen alleen in hun weer. Van het eerste uur tot het laatste zijn het 52.551 uren; 40.575 daarvan hebben een rij.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_impossible():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een regel voor elke onmogelijke waarde

        `drop_impossible(df)` moet de rijen weglaten die niet kunnen kloppen: een temperatuur van 0 K, en de regen van 9.831,3 mm. Houd als regel voor de regen de uren met hoogstens 100 mm, ver boven alles wat dit meetpunt mat. Vervang de twee `...` elk door een regel van de vorm `df = df[...]`. De check roept `drop_impossible(TRAFFIC)` aan, met de hele tabel zoals `read_csv` hem geeft, en probeert vier kleine tabellen van zichzelf, waarvan één zonder één foute rij. Als hij klopt, bewaart de check hem in je project, in `work/projects/tabular/impossible.py`.
        """),
        L.task("Houd de rijen waarvan `temp` boven 0 ligt en `rain_1h` hoogstens 100 is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_impossible():
    box_impossible = L.code_box("impossible", lines=6)
    box_impossible
    return (box_impossible,)


@app.cell(hide_code=True)
def check_impossible(box_impossible):
    result_impossible = L.check("impossible", box_impossible.value)
    result_impossible
    return (result_impossible,)


@app.cell(hide_code=True)
def hints_impossible():
    L.hints(
        "impossible",
        [
            "Stap 2 van het recept. Een mask in `df[...]` houdt de rijen waar het True is: welke rijen moeten blijven?",
            "`df[\"temp\"] > 0` is True voor elke temperatuur die kan; voor de regen is het `<=` en 100.",
            "`df = df[df[\"temp\"] ___ 0]` en `df = df[df[\"rain_1h\"] ___ 100]`.",
        ],
        why="Elke regel houdt de rijen die door zijn toets komen, en de volgende regel werkt op wat er over is. Omdat de grenzen regels zijn en geen rijnummers, werken ze ook op de kleine tabellen van de check, en op uren die de tabel nog niet heeft.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_hours():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Eén rij voor elk uur

        `one_per_hour(df)` moet de tabel teruggeven met één rij voor elk uur: van de rijen met dezelfde `date_time` alleen de eerste. Vervang de `...`. De check roept `one_per_hour(TRAFFIC)` aan en twee kleine tabellen, waarvan één zonder één dubbel uur, en bewaart je functie in `work/projects/tabular/hours.py`.
        """),
        L.task("Laat elke rij weg waarvan de `date_time` al in de tabel staat, en houd de eerste."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hours():
    box_hours = L.code_box("hours", lines=4)
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
            "Stap 3 van het recept. Welke kolom hoort uniek te zijn, en hoe zeg je tegen `drop_duplicates` dat hij alleen naar die ene moet kijken?",
            "`subset=\"date_time\"` laat `drop_duplicates` alleen het uur vergelijken; de eerste rij houdt hij vanzelf.",
            "`return df.drop_duplicates(subset=___)`",
        ],
        why="Zonder `subset` verdwijnen alleen rijen die in elke kolom gelijk zijn: 17. Met `subset` verdwijnt elke tweede, derde of latere rij van een uur: 7.629.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_clean():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. clean: je regels samen

        `clean(df)` haalt je twee regels uit je project, `from tabular.impossible import drop_impossible`, en maakt daarna twee kolommen makkelijker in het gebruik: `date_time` als momenten in de tijd, en `temp` in graden Celsius. Vervang de twee `...`. De check roept `clean(TRAFFIC)` aan en vier kleine tabellen. Waar een module van jou ontbreekt of kapot is, springt de versie van de course in, zoals de melding bij *Waar je project staat* zegt. Als hij klopt, bewaart de check hem in `work/projects/tabular/clean.py`: de volgende stap van je project begint ermee.
        """),
        L.task("Maak van `date_time` een kolom momenten in de tijd, en van `temp` graden Celsius."),
    ])
    return


@app.cell(hide_code=True)
def exercise_clean():
    box_clean = L.code_box("clean", lines=12)
    box_clean
    return (box_clean,)


@app.cell(hide_code=True)
def check_clean(box_clean):
    result_clean = L.check("clean", box_clean.value)
    result_clean
    return (result_clean,)


@app.cell(hide_code=True)
def hints_clean():
    L.hints(
        "clean",
        [
            "Stap 4 van het recept, na je regels. `df[\"date_time\"] = ...` zet een nieuwe versie op de plek van de kolom: wat maakt van tekst een tijd, en hoe kom je van kelvin op graden Celsius?",
            "`pd.to_datetime` krijgt één kolom; graden Celsius zijn kelvin min 273,15.",
            "`df[\"date_time\"] = pd.to_datetime(df[___])` en `df[\"temp\"] = df[\"temp\"] ___ 273.15`.",
        ],
        why="Je regels gaan eerst: de regel voor 0 K heeft de temperatuur in kelvin nodig, want na de omrekening zou het -273,15 zijn. Daarna maakt `pd.to_datetime` van de tekst momenten in de tijd, en krijgt de temperatuur de schaal van een weerbericht.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel haalt je `clean` uit je project, voert hem uit op de hele tabel, en telt wat elke stap overlaat. Uit je schone tabel tekent hij het gemiddelde verkeer voor elk uur van de dag, op werkdagen en in het weekend.
    """)
    return


@app.cell(hide_code=True)
def demo_cell(result_clean):
    _r = result_clean   # zodat deze cel na elke check van oefening 3 opnieuw draait
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Maak eerst oefening 3 af: dan staat je `clean` in je project, en gebruikt deze cel hem.*")
    elif isinstance(_result, str):
        _out = mo.md("*Je `clean` in `work/projects/tabular/clean.py` stopt met een fout, of geeft niet de tabel terug met `date_time` als momenten in de tijd. Heb je het bestand veranderd? Verwijder het, en check oefening 3 opnieuw: de check bewaart je code dan opnieuw.*")
    else:
        _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de melding bij *Waar je project staat* zegt."
        _first = "" if _result["first"] is None else f" laat je `drop_impossible` er **{C.grouped(_result['first'], L.lang)}** over, en"
        _twice = "geen uur dubbel" if _result["repeats"] == 0 else f"{C.grouped(_result['repeats'], L.lang)} rijen die nog een uur herhalen"
        _low, _high = _result["temp"]
        _labels = {"title": "Gemiddeld verkeer per uur van de dag", "x": "uur van de dag", "y": "verkeer per uur",
                   "work": "werkdagen", "weekend": "weekend",
                   "alt": "Het gemiddelde verkeer voor elk uur van de dag in je schone tabel: op werkdagen doorgetrokken, met pieken om 7:00 en om 16:00, en in het weekend gestippeld, met één brede heuvel rond het middaguur."}
        _out = mo.vstack([
            mo.md(f"Van de {C.grouped(_result['rows'], L.lang)} rijen{_first} laat je `clean` er **{C.grouped(_result['clean'], L.lang)}** over: {_twice}, met temperaturen van {C.number(_low, 1, L.lang)} tot {C.number(_high, 1, L.lang)} °C. Je schone tabel springt nog op {C.grouped(_result['gaps'], L.lang)} plekken meer dan een uur: die kan geen regel vullen.{_who}"),
            mo.Html(C.hours_chart(_result["by_hour"], _labels, L.lang)),
            mo.md("Op werkdagen heeft het verkeer twee pieken, de ochtendspits en de middagspits; in het weekend één brede heuvel rond het middaguur. Die vorm moeten de modellen van de latere stappen van je project leren."),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_gap():
    mo.md(r"""
    ### Oefening 4. Het gat van tien maanden
    """)
    return


@app.cell(hide_code=True)
def choice_gap_cell():
    choice_gap = L.choice(
        "gap",
        question="Van 8 augustus 2014 om 01:00 tot 11 juni 2015 om 20:00 heeft de tabel geen rijen: bijna 308 dagen. Wat doe je met dat gat?",
        options={
            "fill": "Invullen: geef elk ontbrekend uur het gemiddelde verkeer van dat uur van de dag",
            "drop": "2014 en 2015 helemaal weglaten, zodat geen jaar een gat heeft",
            "right": "Laten zoals het is: verzin geen uren, en schrijf het gat op in de datasheet, zodat latere stappen weten dat het er is",
        },
        correct="right",
        why={
            "fill": "Dan verzin je 7.386 uren verkeer die niemand telde, en een model leert ervan alsof het metingen zijn. Imputatie vult een cel in van een rij die er is, met een missing indicator om de gok van een meting te onderscheiden; hier ontbreken hele rijen.",
            "drop": "Dan gooi je 8.094 uren weg die wel geteld zijn, en het gat is er nog steeds, nu tussen 2013 en 2016. Een gat is geen fout in de rijen die je hebt: die blijven.",
            "right": "De rijen die je hebt blijven zoals ze zijn, en er wordt niets verzonnen. Omdat het gat in de datasheet staat, kent elke latere stap het: als een stap naar uren achter elkaar kijkt, weet hij dat het uur voor 11 juni 2015 om 20:00 bijna 308 dagen eerder is.",
        },
    )
    choice_gap
    return (choice_gap,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** `describe()` uit [Vragen stellen aan data](/course/open/data-explore?lang=nl) liet de onmogelijke waarden zien; de gaten van [Gaten in een tabel: dropna en fillna](/course/open/data-missing-values?lang=nl) waren cellen, en die van vandaag hele rijen, waar geen `fillna` bij kan. De datasheet uit [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl) is de plek voor het gat van 307 dagen. Je regels staan in een package van jezelf, zoals in [Je eigen module: projectcode in een bestand bewaren](/course/open/py-own-modules?lang=nl): je project `tabular` heeft nu `impossible.py`, `hours.py` en `clean.py`. Het project komt terug aan het eind van de basis van klassieke machine learning, in *Project: bomen op je tabel, eerlijk getest*, waar je schone tabel modellen krijgt. Eerst leer je in de volgende les, *Nu leren of later: de nearest centroid*, meer manieren om van een tabel te leren.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_gap, result_clean, result_hours, result_impossible):
    L.wrap_up(result_impossible, result_hours, result_clean, choice_gap)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
