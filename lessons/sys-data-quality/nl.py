# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-data-quality. Route: course/route/lessen/systemen.yaml. Checks, de metingen, de maand van de demo en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-data-quality", "nl")
    C = L.checks   # de metingen van vier stations, de weerdienst, de maand van de demo, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Gaten in een tabel: dropna en fillna](/course/open/data-missing-values?lang=nl) en [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_gaps():
    quick_gaps = L.choice(
        "quick-gaps",
        question='Een kolom `temp` heeft 10 gaten in 200 rijen. Wat geeft `df["temp"].isna().sum()` na `df["temp"] = df["temp"].fillna(df["temp"].median())`?',
        options={
            "right": "0, want elk gat bevat nu de mediaan",
            "ten": "10, want fillna markeert alleen waar de gaten zaten",
            "rows": "190, het aantal ingevulde waarden",
            "all": "200, want isna telt elke rij van de kolom, ingevuld of niet, zolang de rij bestaat",
        },
        correct="right",
        why={
            "right": "`fillna` zet de mediaan in elk gat, dus vindt `isna` er geen meer: na het invullen lijkt de tabel volledig. In deze les tel je de gaten op de data zoals de data binnenkomt, voordat iets de gaten invult.",
            "ten": "`fillna` vult de gaten in; markeren is het werk van een missing-indicatorkolom. Wat blijft er over voor `isna`? Zie [Gaten in een tabel: dropna en fillna](/course/open/data-missing-values?lang=nl).",
            "rows": "`isna().sum()` telt de gaten, niet de ingevulde waarden. Hoeveel gaten zijn er over na fillna? Zie [Gaten in een tabel: dropna en fillna](/course/open/data-missing-values?lang=nl).",
            "all": "`isna` is alleen `True` waar een waarde ontbreekt. Waar ontbreekt er een na fillna? Zie [Gaten in een tabel: dropna en fillna](/course/open/data-missing-values?lang=nl).",
        },
    )
    quick_gaps
    return (quick_gaps,)


@app.cell(hide_code=True)
def quick_check_box():
    quick_box = L.choice(
        "quick-box",
        question="Hoe groot is het blok ML-code in een echt ML-systeem, in de afbeelding van Sculley en anderen (2015)?",
        options={
            "big": "Het grootste blok, met afstand, want het hele systeem is gebouwd rond het model en de rest voedt het model alleen",
            "half": "Ongeveer de helft, met het geven van de antwoorden als de andere helft",
            "right": "Klein, tussen veel grotere blokken, zoals het verzamelen en controleren van de data",
            "same": "Even groot als elk van de andere blokken",
        },
        correct="right",
        why={
            "right": "Het blok ML-code is klein; het verzamelen van de data, het controleren, features, de antwoorden geven en monitoring zijn veel groter. Deze les gaat over een van de grote blokken: controleren of de data goed is.",
            "big": "In de afbeelding is het blok ML-code een van de kleinste. Welke blokken zijn groter? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
            "half": "De antwoorden geven is één blok van de vele, en het blok ML-code is klein. Welke blokken zijn groot? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
            "same": "De blokken verschillen sterk in grootte, en het blok ML-code hoort bij de kleinste. Welke zijn groter? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
        },
    )
    quick_box
    return (quick_box,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model antwoordt vanuit data, en antwoordt op slechte data net zo zeker als op goede. In de praktijk gaat data vaker kapot dan code: een sensor gaat stuk, een leverancier verandert een formaat, een feed stopt 's nachts. De monitoring van de antwoorden merkt dat pas als de schade er is, dus controleert een team de data zelf, voordat de data de training of het model bereikt. Goed is alleen niet één ding. Een tabel kan elke cel gevuld hebben en toch fout zijn, oud, of op twee manieren geschreven. Datakwaliteit heeft dimensies die je kunt meten, elk met een eigen vraag, en een tabel krijgt op elke dimensie een score.

    **Een voorbeeld uit het dagelijks leven.** Een receptkaart voor brood kan op vier manieren fout gaan. Er kan een regel ontbreken: hoeveel zout? Een regel kan 50 gram zeggen waar 5 bedoeld was. De ene regel kan grammen gebruiken en de volgende theelepels. En de kaart kan de oude zijn, van voordat de oven vervangen werd. Het zijn vier verschillende problemen, en elk probleem vang je met een andere blik.
    """)
    return


@app.cell(hide_code=True)
def predict_catch():
    guess_catch = L.guess(
        "catch",
        question="In één maand gaan vier dingen mis met de weermetingen die het fietsmodel gebruikt: een windmeter gaat kapot, een station schrijft de stad op een nieuwe manier, een thermometer meet 3 graden te warm, en de feed van twee stations stopt. Een check telt elke avond de lege cellen. Hoeveel van de vier vangt de check, denk je?",
        options={
            "none": "Geen, want een kapotte sensor stuurt een nul, geen lege cel",
            "right": "Eén: de kapotte windmeter, want de metingen laten de wind weg",
            "two": "Twee: de kapotte sensor, en de gestopte feed, want daardoor blijven de cellen van twee stations leeg",
            "all": "Alle vier, want elk probleem eindigt in een lege of foute cel",
        },
        reveal="**Eén.** De gestopte feed laat geen lege cellen achter: er komen helemaal geen nieuwe rijen, en de nieuwste rijen van de twee stations zijn even vol als altijd, alleen oud. De nieuwe schrijfwijze en de warme thermometer vullen elke cel, met waarden die er goed uitzien. Elk probleem vraagt een eigen manier van kijken, en de demo zet de vier naast elkaar.",
    )
    guess_catch
    return (guess_catch,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Vier vragen.** **Datakwaliteit** is hoe geschikt data is voor het gebruik. Datakwaliteit heeft dimensies, en elke dimensie is een vraag met een aandeel van 0 tot 1 als antwoord. Deze les gebruikt er vier.

        - **Volledigheid**: staat de waarde er? Het aandeel van de cellen dat gevuld is. Een sleutel die een meting weglaat, of een lege cel in een tabel, telt als gat.
        - **Juistheid**: is de waarde waar? Het aandeel van de waarden dat klopt met een bron die het team vertrouwt, binnen een tolerantie: hier de weerdienst, hoogstens 1 graad ernaast. Veel teksten noemen dit accuracy; in deze course blijft accuracy het aandeel goede antwoorden van een model.
        - **Consistentie**: staat hetzelfde overal op dezelfde manier? Het aandeel van de waarden dat op de afgesproken manier geschreven is: Houten en niet houten, dezelfde eenheid in elke rij, dezelfde naam voor een station in elke tabel.
        - **Actualiteit**: is de waarde nieuw genoeg? Het aandeel van de bronnen waarvan de nieuwste data vers genoeg is: hier, van vandaag. Oude data kan volledig, juist en consistent zijn, en toch fout voor vandaag.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **Een score per dimensie.** Je geeft een tabel een score op elke dimensie en kijkt naar de laagste: daar breekt de data, en daar kijk je eerst. Elke dimensie telt wat bij de eigen vraag hoort. Consistentie en juistheid tellen alleen de ingevulde waarden, want een lege cel telt al bij de volledigheid.

        **Eerst tellen, dan invullen.** Meet de volledigheid op de data zoals de data binnenkomt. Na `fillna` bevat elk gat een waarde en lijkt de tabel volledig; de missing-indicatorkolom van [Gaten in een tabel: dropna en fillna](/course/open/data-missing-values?lang=nl) bewaart wat de score nodig heeft. In [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl) hoorden het verzamelen en het controleren van de data bij de grootste blokken rond het model: dit is het controleren.

        **Meer dimensies.** De vier zijn niet de enige. De Government Data Quality Hub van het Verenigd Koninkrijk raadde op 24 juni 2021 zes dimensies aan, zoals DAMA UK de dimensies definieert: accuracy, completeness, uniqueness, consistency, timeliness en validity, in het Nederlands juistheid, volledigheid, uniciteit, consistentie, actualiteit en validiteit. De norm ISO/IEC 25012 (2008) heeft 15 eigenschappen, waaronder accuracy, completeness, consistency en currentness. Beide bronnen zijn gelezen op 28 november 2026. Uniciteit (geen rij twee keer) en validiteit (een waarde van het goede type en binnen het toegestane bereik) komen terug in de volgende les, als regels.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **datakwaliteit** | | hoe geschikt data is voor het gebruik, gemeten op dimensies als volledigheid, juistheid, consistentie en actualiteit | |
        | **volledigheid** | | het aandeel van de cellen dat gevuld is | `column in row` |
        | **juistheid** | | het aandeel van de waarden dat klopt met een bron die je vertrouwt, binnen een tolerantie; vaak accuracy genoemd | |
        | **consistentie** | | het aandeel van de waarden dat op de afgesproken manier geschreven is: één schrijfwijze, één eenheid, één naam | |
        | **actualiteit** | | of data nieuw genoeg is: het aandeel van de bronnen waarvan de nieuwste data vers genoeg is | |
        """),
        L.watch_out("Een score van 1,0 betekent alleen dat deze ene manier van kijken niets vond. Een tabel vol metingen van vorige week kan 1,0 scoren op volledigheid, juistheid en consistentie."),
        L.watch_out("Herstel geen data binnen de check. Een check die spaties weghaalt of hoofdletters verandert en dan pas telt verbergt juist het probleem dat de check moet melden. Tel eerst, en herstel in een eigen stap, waar je ziet wat er veranderde."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Hoe goed is een tabel, en waar breekt de data:

    1. Bepaal wat een rij is, en welke kolommen het model nodig heeft.
    2. Volledigheid: per kolom het aandeel van de rijen met een waarde.
    3. Juistheid: vergelijk de waarden met een bron die je vertrouwt, binnen een tolerantie.
    4. Consistentie: per kolom het aandeel van de ingevulde waarden dat op de afgesproken manier geschreven is.
    5. Actualiteit: per bron de dag van de nieuwste rij; het aandeel van de bronnen dat vers genoeg is.
    6. Kijk eerst naar de laagste score, en zoek de bron en de dag waar de data brak.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.READINGS` bevat de metingen van vier stations om twaalf uur 's middags, op drie dagen, als dicts: een meting zonder waarde laat de sleutel weg. Volledigheid per kolom: tel de rijen die de sleutel hebben, en deel door het aantal rijen. `_column in _row` vraagt of de dict `_row` de sleutel `_column` heeft. De cel print eerst één meting. Voer de cel uit.
    """)
    return


@app.cell
def example_complete():
    print(C.READINGS[5])
    for _column in ["city", "temp", "wind"]:
        _filled = 0
        for _row in C.READINGS:
            if _column in _row:
                _filled = _filled + 1
        print(_column, round(_filled / len(C.READINGS), 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De stad ontbreekt één keer, bij west op dag 3, en de wind twee keer, bij east op dag 2 en 3: volledigheid 0,917 en 0,833. De temperatuur staat er altijd: 1,0. Maar kijk naar de geprinte meting: south schrijft houten, waar dag 1 Houten zegt. De volledigheid ziet dat niet, want de cel is gevuld. Oefening 1 telt het.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_consistency():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Zoals afgesproken?

        `consistency(rows, column, agreed)` krijgt de rijen, de naam van een kolom, en een lijst `agreed` met de afgesproken schrijfwijze van elke waarde. De `for`-loop loopt de rijen langs. Een `row` met de kolom telt mee in `filled`, en `same` telt de waarden die op een afgesproken manier geschreven zijn. Een rij zonder de kolom telt voor geen van beide: de lege cel telt bij de volledigheid. De functie geeft het aandeel `same / filled` terug. Vervang de `...` door de vraag of de waarde van `row` onder `column` een van de schrijfwijzen van `agreed` is.

        `C.CITIES` is `["Utrecht", "Houten", "Zeist", "Nieuwegein"]`. De check roept `consistency(C.READINGS, "city", C.CITIES)` aan. Daarna de namen van de stations, een kortere lijst, een paar eigen rijen, en een kolom met eenheden.
        """),
        L.task("Tel de ingevulde waarden die zoals afgesproken geschreven zijn, en geef het aandeel terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_consistency():
    box_consistency = L.code_box("consistency", lines=10)
    box_consistency
    return (box_consistency,)


@app.cell(hide_code=True)
def check_consistency(box_consistency):
    result_consistency = L.check("consistency", box_consistency.value)
    result_consistency
    return (result_consistency,)


@app.cell(hide_code=True)
def hints_consistency():
    L.hints(
        "consistency",
        [
            '`row` is één meting, een dict, en `column` de naam van een sleutel, zoals `"city"`. De vraag na `if` is of de waarde van `row` onder de sleutel `column` een van de strings van `agreed` is. Hoe zoek je een waarde op in een dict, en hoe vraag je of een waarde een van de waarden van een lijst is?',
            '`"Zeist" in ["Utrecht", "Zeist"]` geeft `True`, en `"zeist" in ["Utrecht", "Zeist"]` geeft `False`: de schrijfwijze moet precies hetzelfde zijn, met hoofdletters en spaties.',
            "`            if row[column] ___:`",
        ],
        why='`if row[column] in agreed:` is waar voor een waarde die precies als een van de afgesproken schrijfwijzen geschreven is. Van de 11 ingevulde steden zijn "houten" (twee keer) en "Utrecht " (één keer) dat niet: 8 van 11, 0,727.',
    )
    return


@app.cell(hide_code=True)
def text_exercise_correctness():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Waar binnen een tolerantie?

        Nu met NumPy, voor elke dag tegelijk. `correctness(readings, reference, tolerance)` krijgt twee arrays van dezelfde lengte: de metingen van een station, en de referentie, de waarden van een bron die het team vertrouwt, op dezelfde momenten. `diff` bevat hoe ver elke meting van de referentie ligt: positief als de meting te hoog is, negatief als de meting te laag is. `close` moet een array van `True` en `False` worden, met `True` voor een meting die hoogstens `tolerance` van de referentie ligt. `close.mean()` geeft dan het aandeel `True`, want `True` telt als 1. Vervang de `...`.

        `C.TEMPS` bevat de temperaturen om twaalf uur 's middags van station west op 12 dagen, en `C.REFERENCE` de temperaturen van de weerdienst. De check roept `correctness(C.TEMPS, C.REFERENCE, 1.0)` aan. Daarna een tolerantie van `0.5` en van `3.0`, metingen gelijk aan de referentie, en twee eigen kleine arrays.
        """),
        L.task("Tel, voor elke dag tegelijk, de metingen die hoogstens de tolerantie ernaast zitten, eronder of erboven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_correctness():
    box_correctness = L.code_box("correctness", lines=5)
    box_correctness
    return (box_correctness,)


@app.cell(hide_code=True)
def check_correctness(box_correctness):
    result_correctness = L.check("correctness", box_correctness.value)
    result_correctness
    return (result_correctness,)


@app.cell(hide_code=True)
def hints_correctness():
    L.hints(
        "correctness",
        [
            "Een meting telt als goed als de meting hoogstens `tolerance` van de referentie ligt, eronder of erboven. Daar moet `close` `True` zijn: een vergelijking met `<=` van hoe ver elke meting ernaast zit, tegen `tolerance`. Hoe ver zit een meting met een `diff` van -3,5 ernaast?",
            "Een `diff` van -3,5 ligt onder 1,0, en toch zit de meting 3,5 graad ernaast. De functie `abs` van Python maakt van -3,5 gewoon 3,5, en werkt ook op een NumPy-array, getal voor getal: `abs(np.array([-2.0, 0.5]))` geeft `array([2. , 0.5])`.",
            "`    close = ___ <= tolerance`",
        ],
        why="`close = abs(diff) <= tolerance` is `True` waar een meting hoogstens 1 graad ernaast zit, eronder of erboven. West heeft 12 dagen, en 9 daarvan zijn zo: 0,75. Zonder `abs` zouden de twee dagen ver eronder als goed tellen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_problems():
    mo.md(r"""
    ### Oefening 3. Welke dimensie?

    Acht problemen uit de data van een fietsverhuur. Zet elk probleem onder de dimensie die het probleem meet.
    """)
    return


@app.cell(hide_code=True)
def sort_problems_cell():
    sort_problems = L.sort("problems")
    sort_problems
    return (sort_problems,)


@app.cell(hide_code=True)
def check_problems(sort_problems):
    result_problems = L.check("problems", sort_problems)
    result_problems
    return (result_problems,)


@app.cell(hide_code=True)
def hints_problems():
    L.hints(
        "problems",
        [
            "De vier vragen, onder *Vier vragen*: staat de waarde er, is de waarde waar, staat de waarde er op de afgesproken manier, en is de waarde nieuw genoeg?",
            "In elke kolom horen twee problemen.",
            "Volledigheid: de wind en \\_\\_\\_. Juistheid: de zon en \\_\\_\\_. Consistentie: houten en \\_\\_\\_. Actualiteit: vrijdag en \\_\\_\\_.",
        ],
        why="**Volledigheid:** een week geen wind, en rijen zonder stad. **Juistheid:** drie graden te warm, en tien keer de wind. **Consistentie:** houten en Houten, en twee namen voor één station. **Actualiteit:** stil sinds vrijdag, en zes uur achter.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_first():
    mo.md(r"""
    ### Oefening 4. Waar kijk je eerst?

    Op maandagochtend zitten de antwoorden van het fietsmodel ver ernaast. De check van zondagavond gaf: volledigheid 0,99, juistheid 0,98, consistentie 1,00, actualiteit 0,25.
    """)
    return


@app.cell(hide_code=True)
def choice_first_cell():
    choice_first = L.choice(
        "first",
        question="Waar kijk je eerst naar?",
        options={
            "retrain": "Het model opnieuw trainen op de nieuwste data van de stations, want de wereld is veranderd",
            "sensor": "Een thermometer die te warm meet",
            "spelling": "Een stad die op een nieuwe manier geschreven wordt",
            "right": "Of drie van de vier stations de metingen nog sturen",
        },
        correct="right",
        why={
            "right": "Actualiteit 0,25: van de vier stations stuurde alleen één station zondag metingen. De andere drie scores zijn hoog, want de scores kijken naar de nieuwste rijen die er zijn, en de rijen zijn oud. Zoek eerst uit waarom drie feeds stopten.",
            "retrain": "De nieuwste data van drie stations is oud, dus zou opnieuw trainen van dezelfde oude rijen leren. Welke score is laag?",
            "sensor": "De juistheid is 0,98: de metingen kloppen met de weerdienst. Welke score is laag?",
            "spelling": "De consistentie is 1,00: elke waarde staat er zoals afgesproken. Welke score is laag?",
        },
    )
    choice_first
    return (choice_first,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Een maand metingen

    De vier stations van deze les, 30 dagen lang, met elk uur een meting. Vier dingen gaan mis, elk op een paar dagen. Elke avond geeft een check de vier scores over de nieuwste metingen van elk station: de volledigheid van de temperatuur en de wind, de juistheid tegen de weerdienst, de consistentie van de stad, en de actualiteit, het aandeel van de stations met metingen van de dag zelf. Kies een dag.
    """)
    return


@app.cell(hide_code=True)
def demo_day():
    L.explore("day", "dag van de maand", lambda _v: C.frame(_v, L.lang), start=7, values=C.DEMO_DAYS, role="",
              task="Ga de maand langs: op welke dagen zakt een score onder 0,95, en bewegen de andere drie mee?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Elk incident beweegt één score en laat de andere drie waar de scores waren. De kapotte windmeter, op dag 6 tot en met 9, beweegt alleen de volledigheid, tot 0,85. De nieuwe schrijfwijze van south, op dag 13 tot en met 17, beweegt alleen de consistentie, tot 0,75. De thermometer in de zon, op dag 20 tot en met 23, beweegt alleen de juistheid, tot 0,72. En de gestopte feed van north en east, vanaf dag 26, beweegt alleen de actualiteit, tot 0,50. Op elke andere dag blijven alle vier boven 0,97. Eén getal voor de hele tabel, het gemiddelde van de vier, had de windmeter gemist: het gemiddelde blijft op de dagen van de windmeter op 0,96 of hoger, boven 0,95. En de gestopte feed kunnen de rijen nooit uit zichzelf laten zien: de nieuwste rijen van north en east zijn vol, juist en consistent, alleen oud.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Gaten in een tabel: dropna en fillna](/course/open/data-missing-values?lang=nl) vulde je de gaten van een tabel in; hier tel je de gaten eerst, met drie manieren van kijken erbij. In [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl) was het controleren van de data een van de grote blokken rond het model: het controleren gebeurt eerst, voordat er getraind wordt en voordat het model antwoordt. In de volgende les, *Regels voor data: schema's en expectations*, wordt het kijken een poort: regels per kolom en voor de hele tabel houden een slechte partij tegen, voordat de partij de training bereikt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_first, result_consistency, result_correctness, result_problems):
    L.wrap_up(result_consistency, result_correctness, result_problems, choice_first)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
