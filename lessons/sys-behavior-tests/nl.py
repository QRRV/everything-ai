# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-behavior-tests. Route: course/route/lessen/systemen.yaml. Checks, de vijf versies van het model, de demo en de plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-behavior-tests", "nl")
    C = L.checks   # het model van de course en vier versies met een bug, de checks van je tests, de demo en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Tests voor ML-code](/course/open/sys-code-tests?lang=nl) en [Project: een model van de vraag, bijgehouden en ingepakt](/course/open/sys-project-tracked-model?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_tests():
    quick_tests = L.choice(
        "quick-tests",
        question="Een test die je schreef slaagt voor de goede versie van je code. Wat moet er nog meer waar zijn voordat je de test vertrouwt?",
        options={
            "fast": "De test moet binnen een seconde klaar zijn, want pytest stopt elke test die langer duurt",
            "right": "De test moet falen voor een versie van de code met een bekende bug, anders slaagt de test mogelijk altijd",
            "many": "De test moet minstens drie regels met `assert` hebben, één voor elke soort bug die een functie heeft",
            "random": "De test moet ook slagen op willekeurige inputs, want één voorbeeld met een bekend antwoord laat niets zien",
        },
        correct="right",
        why={
            "right": "Een test die voor elke versie slaagt vangt niets. In Tests voor ML-code moest elke test falen voor elke versie met een bug, zoals een `inputs` die het uur 8 in kolom 9 zette.",
            "fast": "pytest stopt geen test omdat de test de tijd neemt; een trage test is alleen lastig: wat laat zien dat een test een bug kan vangen? Zie [Tests voor ML-code](/course/open/sys-code-tests?lang=nl).",
            "many": "Eén regel met `assert` kan meerdere bugs vangen, en drie regels kunnen er geen enkele vangen: wat laat zien dat een test een bug vangt? Zie [Tests voor ML-code](/course/open/sys-code-tests?lang=nl).",
            "random": "Eén voorbeeld met een bekend antwoord is juist waar een unit test uit bestaat: wat laat zien dat de test kan falen? Zie [Tests voor ML-code](/course/open/sys-code-tests?lang=nl).",
        },
    )
    quick_tests
    return (quick_tests,)


@app.cell(hide_code=True)
def quick_check_rmse():
    quick_rmse = L.choice(
        "quick-rmse",
        question="Je verpakte model heeft een validation RMSE van 53,04 fietsen per uur. Wat vertelt dat ene getal je?",
        options={
            "every": "Dat geen enkel uur van een validatiedag meer dan 53,04 fietsen naast zit, 's nachts of in de drukte",
            "new": "Dat het model in 2012 ook ongeveer 53,04 fietsen per uur naast zit, omdat de validatiedagen nieuw zijn",
            "half": "Dat ongeveer de helft van de uren van de validatiedagen meer dan 53,04 fietsen naast zit, en de helft minder",
            "right": "Hoe ver het model naast zit over alle uren van de validatiedagen, in één getal, niet waar of wanneer",
        },
        correct="right",
        why={
            "right": "De RMSE kwadrateert elke fout, neemt het gemiddelde over de 1.733 validatie-uren en dan de wortel: één getal voor alle uren samen. Of het model de nacht goed doet, of regen, zegt het getal niet, en dat test deze les.",
            "every": "Sommige uren zitten veel verder naast en veel uren dichterbij: de RMSE is de wortel van de gemiddelde kwadratische fout. Wat laat één getal over 1.733 uren weg? Zie [Project: een model van de vraag, bijgehouden en ingepakt](/course/open/sys-project-tracked-model?lang=nl).",
            "new": "De validatiedagen zijn nieuw voor het model, maar het zijn dagen van 2011, met het weer en de fietsers van dat jaar: zegt een score op 2011 hoe 2012 gaat? Zie [Project: een model van de vraag, bijgehouden en ingepakt](/course/open/sys-project-tracked-model?lang=nl).",
            "half": "Dat zou de mediaan van de fouten zijn, niet de RMSE, de wortel van de gemiddelde kwadratische fout: wat vertelt één getal over alle uren, en wat niet? Zie [Project: een model van de vraag, bijgehouden en ingepakt](/course/open/sys-project-tracked-model?lang=nl).",
        },
    )
    quick_rmse
    return (quick_rmse,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Je verpakte model heeft een validation RMSE van 53,04 fietsen per uur: één getal over 1.733 uren. Het getal vertelt niet of het model de eenvoudige dingen goed doet, zoals een rustige nacht, of het model bij regen minder fietsen verwacht, of dat het antwoord verandert als er iets verandert wat niet zou mogen uitmaken. De model assertion van [Tests voor ML-code](/course/open/sys-code-tests?lang=nl) vond 235 uren van 2011 waarvoor het model minder dan 0 fietsen verwacht, en de RMSE gaf daar geen enkel teken van.

    In 2020 testten Marco Tulio Ribeiro en anderen modellen die tekst lezen, van grote bedrijven en uit onderzoek, met hoge scores op de bijbehorende testsets. Met kleine tests van elk één vaardigheid vonden de onderzoekers in elk model fouten. De methode, CheckList, won dat jaar de prijs voor het beste paper op de conferentie ACL. Een model testen zoals engineers software testen, één vaardigheid tegelijk, heet behavioral testing. Deze les schrijft zulke tests voor je model van de vraag, van de drie soorten die CheckList gebruikt.

    **Een voorbeeld uit het dagelijks leven.** Een rijexamen is geen score, gemiddeld over duizend kilometer. De examinator controleert een paar dingen die elke bestuurder goed moet doen, zoals stoppen voor een rood licht; dingen die niet mogen uitmaken, zoals de kleur van de auto; en dingen die het rijden de goede kant op moeten veranderen, zoals langzamer rijden in de mist.
    """)
    return


@app.cell(hide_code=True)
def predict_later():
    guess_later = L.guess(
        "later",
        question="Een versie van het model kreeg ook de kolom instant: het rijnummer van elk uur in de tabel, 1 voor het eerste uur van 2011 en 8.645 voor het laatste. De validation RMSE van deze versie is 49,98 fietsen per uur, lager dan de 53,04 van het model van de course. Geef de versie dezelfde uren van 2011, met alleen de rijnummers een jaar opgeschoven, zoals bij de uren van 2012. Wat doet de versie, denk je?",
        options={
            "same": "De versie verwacht dezelfde fietsen als eerst, want aan de uren zelf is niets veranderd",
            "right": "De antwoorden schuiven, tot tientallen fietsen per uur, terwijl alleen het rijnummer veranderde",
            "error": "De versie stopt met een fout, want tijdens het leren zag de versie nooit een rijnummer boven 8.645",
            "more": "De versie verwacht in elk uur meer fietsen, want in 2011 hadden de latere rijen meer fietsen",
        },
        reveal="**De antwoorden schuiven.** Voor dezelfde uren, een jaar later in de tabel, verwacht de versie gemiddeld 12,4 fietsen minder, en in één uur tot 66,7 meer of minder: 8.454 van de 8.645 uren schuiven meer dan 1 fiets. Op de echte uren van 2012 doet de versie het slechter dan het model van de course, met een RMSE van 145,0 tegenover 130,7. Allebei doen het op 2012 veel slechter dan op 2011, omdat 2012 veel meer fietsers had; *Project: je model als service, bewaakt op drift* komt daarop terug. De validatiedagen konden het probleem niet laten zien: elke vijfde dag van het jaar is er een, tussen de trainingsdagen in, dus leek het rijnummer daar een prima feature. Een invariance test laat het in een seconde zien, zoals oefening 2 doet.",
    )
    guess_later
    return (guess_later,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Behavioral testing: het model van buitenaf.** De tests van [Tests voor ML-code](/course/open/sys-code-tests?lang=nl) kijken naar binnen: één functie, één rij, één batch. Een behavioral test kijkt van buitenaf naar het model, zoals het model gebruikt wordt: er gaat een tabel met uren in, en er komen fietsen per uur uit. De test controleert één vaardigheid tegelijk, met gevallen die je zelf kiest. Een echt aantal fietsen is niet nodig: je zegt wat eruit moet komen, of hoe het ene antwoord zich tot het andere moet verhouden. Zo kan de test op elk uur draaien, ook op uren die nooit gebeurd zijn, zoals een zomerdag met regen in elk uur.

        **Drie soorten tests.** CheckList noemt er drie:

        - Een **minimum functionality test** (MFT) vraagt om eenvoudige gevallen die elk bruikbaar model goed moet doen, met een bekend antwoord. Van 1 tot 4 uur 's nachts op een werkdag huurt bijna niemand een fiets: in 2011 minder dan 40 fietsen in zo'n uur, op twee uren na, met 50 en 114. Een bruikbaar model verwacht daar weinig fietsen.
        - Een **invariance test** (INV) verandert iets wat niet mag uitmaken, en vraagt hetzelfde antwoord. De kolom instant is het rijnummer van elk uur in de tabel; een ander rijnummer verandert niets aan het weer, het uur of de dag, dus mag het ook niets veranderen aan wat het model verwacht.
        - Een **directional expectation test** (DIR) verandert iets wat wel uitmaakt, en vraagt dat het antwoord de bekende kant op beweegt. Met regen in plaats van helder weer pakken minder mensen een fiets: het model moet minder fietsen verwachten, niet meer.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        Het plaatje laat er twee zien op één dag. De lijn van streepjes, regen in elk uur, ligt de hele dag onder de doorgetrokken lijn: de richting klopt. Van 1 tot 4 uur 's nachts blijft het model onder het korte lijntje op 50 fietsen: het minimum klopt.

        **Verander een kopie.** Een invariance test en een directional expectation test veranderen een kolom van een tabel, en vergelijken. `day.copy()` geeft een nieuwe tabel met dezelfde rijen en kolommen; een wijziging daarin laat `day` zoals het was. `rainy = day` maakt geen nieuwe tabel, alleen een tweede naam voor dezelfde tabel: een wijziging via `rainy` verandert ook `day`, en dan vergelijkt een test een tabel met zichzelf.

        **De fixture `expect`.** Je tests krijgen het model als functie, `expect`: er gaat een tabel met uren in, en er komt een tensor uit met één getal per uur, wat het model in fietsen verwacht. pytest geeft een testfunctie de waarde van de fixture met dezelfde naam als het argument, dus krijgt een test die begint met `def test_rain(expect):` de fixture `expect`. Deze les zet het bestand `conftest.py` in je project, dat pytest als eerste leest; `@pytest.fixture` boven de functie `expect` maakt van de functie een fixture, en de functie laadt je pakket uit `ops/model` met `load` uit `ops/serving.py`. De checks van deze les geven je tests in plaats daarvan een versie van het model.

        **Bugs rond het model.** Een behavioral test draait op het model zoals het gebruikt wordt, dus ziet de test ook bugs in de code eromheen, waar de RMSE van de training nooit langskomt. De checks proberen je tests op het model van de course en op vier versies met een bug. Het uur in UTC, de klok van de hele wereld, in de winter 5 uur later dan in Washington. De twee temperaturen in graden, terwijl de data de temperaturen deelt door 41 en 50. Het weer omgekeerd, 1 voor zware regen en 4 voor helder weer. En de versie met het rijnummer uit de gok. Een goede test slaagt voor het model van de course en faalt voor de bug die de test moet vangen.

        **Tel de fouten.** Sommige tests zijn regels waar het model voor moet slagen. Andere zijn vragen: CheckList telt per test hoeveel gevallen falen, en een hoog aantal laat zien waar het model zwak is, wat de moeite waard is om op te schrijven voor wie het model gebruikt.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **behavioral testing** | | een model van buitenaf testen, één vaardigheid tegelijk, met gevallen die je zelf kiest, in plaats van één score over een testset | `expect(hours)` |
        | **minimum functionality test** | MFT | een test van eenvoudige gevallen die elk bruikbaar model goed moet doen, met een bekend antwoord | `expect(night)` |
        | **invariance test** | INV | een test die iets verandert wat niet mag uitmaken, en hetzelfde antwoord vraagt | `moved["instant"]` |
        | **directional expectation test** | DIR | een test die iets verandert wat wel uitmaakt, en vraagt dat het antwoord de bekende kant op beweegt | `rain["weathersit"] = 3` |
        | `.copy()` | copy | een nieuwe tabel met dezelfde rijen en kolommen; een wijziging daarin laat de tabel waar de kopie vandaan komt zoals de tabel was | `day.copy()` |
        | `.between(1, 4)` | between | True voor elke waarde van 1 tot en met 4 | `work["hr"].between(1, 4)` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Behavioral tests van een model:

    1. Schrijf in woorden op wat het model moet doen: eenvoudige gevallen die het model goed moet doen, wijzigingen die niet mogen uitmaken, en wijzigingen die het antwoord één kant op moeten bewegen.
    2. Kies voor elk de uren: echte uren uit de data, of uren die je bewust verandert.
    3. Leg voor een minimum functionality test een grens vast waar elk bruikbaar model aan voldoet, met ruimte over.
    4. Verander voor een invariance test een kopie, en leg vast dat het antwoord hetzelfde blijft.
    5. Verander voor een directional expectation test een kopie, en leg vast dat het antwoord de bekende kant op beweegt, gemiddeld of per geval.
    6. Zorg dat elke test faalt voor een versie met een bekende bug.
    7. Tel per test hoeveel gevallen falen, en schrijf op waar het model zwak is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Donderdag 14 juli 2011 was de hele dag helder. Om 3 uur 's nachts verwacht het model van de course −6,3 fietsen: onder de 50, dus klopt de rustige nacht voor dit uur, al is minder dan 0 een fout op zich. De versie met het uur in UTC telt er 5 uur bij op, zoals in de winter, dus krijgt het model in plaats daarvan uur 8, en dan verwacht het model 425,6: ver boven de 50, dus faalt een test van de rustige nacht voor deze versie.

    Om 8 uur 's ochtends verwacht het model 431,2 fietsen. Met regen in dat uur, weathersit 3 in plaats van 1, verwacht het model 373,5: 57,7 minder, de bekende kant op. De versie met het rijnummer verwacht op deze dag om 8 uur 440,4, en 417,5 voor hetzelfde uur een jaar later in de tabel: 22,9 minder, terwijl er aan het uur niets veranderde. Voor het model van de course geven allebei 431,2, omdat het model het rijnummer nooit krijgt.
    """)
    return


@app.cell(hide_code=True)
def text_day():
    mo.md(r"""
    ### 14 juli, zoals de dag was en met regen

    De cel leest de uren van 2011 met `read_hours` uit je project en neemt donderdag 14 juli, dag 194 als 1 januari dag 0 is. De cel maakt een kopie met `.copy()`, en zet lichte regen, weathersit 3, in elk uur van de kopie. `C.expect` is het verpakte model van de course voor je project, het model dat de checks het model van de course noemen: het geeft wat het model voor elk uur verwacht, en `.round()` rondt af op hele fietsen. `print` laat beide zien. Voer de cel uit.
    """)
    return


@app.cell
def example_day():
    from ops.demand import read_hours

    hours = read_hours(2011)
    day = hours[hours["day"] == 194]   # donderdag 14 juli 2011
    rainy = day.copy()
    rainy["weathersit"] = 3            # lichte regen in elk uur van de kopie
    print("zoals de dag was:", C.expect(day).round())
    print("met regen:", C.expect(rainy).round())
    return


@app.cell(hide_code=True)
def text_exercise_quiet_night():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een rustige nacht

        `test_quiet_night(expect)` is een minimum functionality test. pytest geeft de test `expect`, de fixture: een functie die een tabel met uren krijgt en geeft wat het model in elk uur verwacht, als tensor met één getal per uur. De test leest elk uur van 2011 met `read_hours` uit `ops.demand` in `HOURS`. `work` houdt de uren van de werkdagen, en `night` de uren van 1 tot 4 uur 's nachts van deze dagen, met `.between(1, 4)`: 939 uren. `pred` bevat wat het model voor elk van deze uren verwacht. In 2011 werden in zo'n uur minder dan 40 fietsen gehuurd, op twee uren na, met 50 en 114.

        Voeg eronder een `assert` toe, net zo ver ingesprongen als de andere regels: een regel die faalt als het model in een van deze uren veel fietsen verwacht. Een vergelijking van een tensor met een getal, zoals `pred > 10`, geeft een tensor met een True of False voor elk uur; `.all()` geeft alleen True als elke waarde True is.

        De check voert je test uit op het model van de course, waar de test moet slagen, en op twee versies met een bug, waar de test moet falen: het uur in UTC en de temperaturen in graden. Onder de check zie je hoe je test het op elke versie doet. Als de test klopt, bewaart de check je test als `work/projects/ops/test_night.py`.
        """),
        L.task("Voeg een `assert` toe: een regel die faalt als het model in een van deze nachturen veel fietsen verwacht."),
    ])
    return


@app.cell(hide_code=True)
def exercise_quiet_night():
    box_quiet_night = L.code_box("quiet-night", lines=10)
    box_quiet_night
    return (box_quiet_night,)


@app.cell(hide_code=True)
def check_quiet_night(box_quiet_night):
    result_quiet_night = L.check("quiet-night", box_quiet_night.value)
    result_quiet_night
    return (result_quiet_night,)


@app.cell(hide_code=True)
def hints_quiet_night():
    L.hints(
        "quiet-night",
        [
            "Vergelijk `pred` met een grens: de vergelijking geeft een True of False voor elk van de 939 uren, en `.all()` maakt daaruit de ene True of False die `assert` nodig heeft.",
            "Kies een grens waar een goed model 's nachts onder blijft, met ruimte over: in 2011 hadden zulke uren minder dan 40 fietsen, op twee na, en het model van de course verwacht tot 47,8. Een versie die de ochtend in de nacht ziet verwacht honderden.",
            "`    assert (___).all()`",
        ],
        why="`assert (pred < 50).all()`, ingesprongen in `test_quiet_night`: elk nachtuur onder de 50 fietsen. Elke grens boven 47,8 en tot een paar honderd slaagt voor het model van de course en faalt voor beide bugs.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_row_number():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een test die altijd slaagt

        `test_row_number(expect)` is een invariance test: als het rijnummer van elk uur een jaar opschuift, 8.645 rijen, mag dat niets veranderen aan wat het model verwacht. De test leest `HOURS` met `read_hours` uit `ops.demand`, net als in oefening 1. `moved = HOURS` moet de tabel geven om te veranderen, de regel erna telt 8.645 op bij de kolom instant, en de `assert` vergelijkt wat het model voor `moved` verwacht met wat het model voor `HOURS` verwacht: `==` voor elk uur, en `.all()` over alle uren.

        De test heeft een bug. De test slaagt voor beide versies die de check probeert: het model van de course, en de versie met het rijnummer, de versie die de test moest vangen. Zoek de bug, en verander één regel. Als de test klopt, bewaart de check je test als `work/projects/ops/test_row_number.py`.
        """),
        L.task("Verander één regel, zodat de test faalt voor de versie met het rijnummer en nog steeds slaagt voor het model van de course."),
    ])
    return


@app.cell(hide_code=True)
def exercise_row_number():
    box_row_number = L.code_box("row-number", lines=9)
    box_row_number
    return (box_row_number,)


@app.cell(hide_code=True)
def check_row_number(box_row_number):
    result_row_number = L.check("row-number", box_row_number.value)
    result_row_number
    return (result_row_number,)


@app.cell(hide_code=True)
def hints_row_number():
    L.hints(
        "row-number",
        [
            "Voer de test in je hoofd uit. Hoeveel tabellen zijn er na de regel met `moved`? Welke tabel verandert de regel erna, en wat vergelijkt de `assert` dan?",
            "`moved = HOURS` maakt geen nieuwe tabel: de ene tabel krijgt een tweede naam. 8.645 optellen via `moved` verandert ook `HOURS`, dus krijgt het model twee keer dezelfde rijen. De test heeft een eigen tabel nodig om te veranderen.",
            "`    moved = ___`",
        ],
        why="`moved = HOURS.copy()`: een nieuwe tabel met dezelfde rijen, zodat de regel erna alleen de rijnummers van `moved` verandert. De versie met het rijnummer verwacht dan andere fietsen voor `moved` dan voor `HOURS`, en de test faalt voor deze versie, zoals het hoort.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rain():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Regen

        `test_rain(expect)` is een directional expectation test. De test leest `HOURS` met `read_hours` uit `ops.demand`, net als in oefening 1. `clear` bevat de 5.645 uren van 2011 met helder weer, weathersit 1, en `rain` is een kopie van `clear`, gemaakt met `.copy()`, met weathersit 3, lichte regen, in elk uur.

        Voeg eronder een `assert` toe: een regel die faalt, tenzij het model met regen minder fietsen verwacht dan in dezelfde uren met helder weer. `expect(rain)` en `expect(clear)` geven elk een tensor van 5.645 getallen, en `.mean()` geeft het gemiddelde van een tensor.

        De check voert je test uit op het model van de course en op een jaar met 50% meer fietsers, waar de test moet slagen, en op het weer omgekeerd, waar de test moet falen. Als de test klopt, bewaart de check je test als `work/projects/ops/test_rain.py`.
        """),
        L.task("Voeg een `assert` toe: een regel die faalt, tenzij het model met regen minder fietsen verwacht."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rain():
    box_rain = L.code_box("rain", lines=10)
    box_rain
    return (box_rain,)


@app.cell(hide_code=True)
def check_rain(box_rain):
    result_rain = L.check("rain", box_rain.value)
    result_rain
    return (result_rain,)


@app.cell(hide_code=True)
def hints_rain():
    L.hints(
        "rain",
        [
            "Vergelijk één getal van `expect(rain)` met één getal van de andere tensor: wat het model gemiddeld verwacht, met `.mean()`.",
            "De twee tensors uur voor uur vergelijken zou vragen dat elk van de 5.645 uren daalt, en 's nachts verwacht het model met of zonder regen ongeveer hetzelfde. Het gemiddelde van een tensor is één getal, en `assert` heeft één True of False nodig.",
            "`    assert expect(rain).mean() < ___`",
        ],
        why="`assert expect(rain).mean() < expect(clear).mean()`: voor het model van de course 121,6 tegenover 157,4 fietsen per uur, 22,8% minder; voor het weer omgekeerd 137,7 tegenover 109,2.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Je tests op vijf versies

    De checks bewaarden je drie tests in je project, en nu draaien de tests op alle vijf versies: het model van de course en de vier versies met een bug. Waar een test van jou nog niet in je project staat, draait de test van de course zelf. Ga met de slider langs de versies: het plaatje laat zien wat elke versie op 14 juli verwacht, zoals de dag was, met regen in elk uur, en een jaar later in de tabel, en eronder hoe je tests het deden.
    """)
    return


@app.cell(hide_code=True)
def demo_versions(result_quiet_night, result_rain, result_row_number):
    (result_quiet_night, result_row_number, result_rain)   # opnieuw zodra een check een test bewaart
    _r = C.demo()
    _out = mo.vstack([
        L.explore("versions", "versie", lambda _v: C.version_frame(_r, _v, L.lang), start=1, low=1, high=5, step=1,
                  role="",
                  task="Ga langs de vijf versies. Welke versie schuift de drukke ochtend naar de nacht, welke verwacht in elk uur duizenden fietsen, en bij welke gaat de paarse lijn weg van de doorgetrokken lijn?"),
        mo.Html(C.demo_key(L.lang)),
        mo.Html(C.matrix_picture(_r, L.lang)),
        mo.md(C.verdicts(_r, L.lang)),
    ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.vstack([
        mo.md(r"""
        Elke versie met een bug faalt voor een van de tests. Het uur in UTC schuift de drukke ochtend naar de nacht: om 3 uur 's nachts op 14 juli verwacht deze versie 425,6 fietsen. De temperaturen in graden zetten bijna elk uur in de duizenden, en het korte lijntje van de rustige nacht ligt onder in het plaatje. Het weer omgekeerd legt de lijn van streepjes boven de doorgetrokken lijn. De versie met het rijnummer is de enige waarvan de paarse lijn weggaat van de doorgetrokken lijn, en alleen de test van het rijnummer vangt deze versie. Geen enkele test vangt alle vier bugs; een test van elke soort wel.

        ### Waar het model van de course faalt

        De checks vragen om tests die slagen voor het model van de course. CheckList stelt ook een andere vraag: voor hoeveel gevallen faalt een test? Het plaatje laat vijf tests los op het model van de course, geval voor geval.
        """),
        mo.Html(C.report_picture(L.lang)),
        mo.md(r"""
        De rustige nacht en het rijnummer kloppen in elk geval. Minder dan 0 fietsen in 235 van de 8.645 uren is de fout die de model assertion vond. Regen, uur voor uur getest in plaats van gemiddeld, faalt in 272 van de 5.645 heldere uren, waarvan 261 tussen middernacht en 7 uur 's ochtends, als het model hoe dan ook bijna niets verwacht. En één invariance test faalt in 333 van de 2.495 weekenduren: of een weekenduur een zaterdag of een zondag is, verandert wat het model daar verwacht met meer dan 20 fietsen, tot 38,0, terwijl de twee dagen in 2011 ongeveer evenveel fietsen per uur hadden, 142,2 en 143,8. Oefening 4 vraagt waarom.
        """),
        L.watch_out("Een model dat voor al je behavioral tests slaagt is daarmee nog niet goed: de tests controleren alleen waar je aan dacht. Het advies van CheckList is om de vaardigheden een voor een langs te gaan, zoals het uur van de dag, het weer en de dag van de week, en elke soort test op elke vaardigheid te proberen."),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_weekend():
    mo.md(r"""
    ### Oefening 4. Zaterdag of zondag

    De invariance test van het weekend vraagt dat het model van de course voor elk van de 2.495 weekenduren van 2011 met zaterdag bijna hetzelfde verwacht als met zondag, binnen 20 fietsen. In 333 uren liggen de twee antwoorden verder uit elkaar, tot 38,0, de meeste tussen 8 uur 's ochtends en 4 uur 's middags, en laat op de avond. De inputs van het model zijn dezelfde als in [Project: een model van de vraag, bijgehouden en ingepakt](/course/open/sys-project-tracked-model?lang=nl): het uur als 24 one-hot-kolommen, en elke feature als één gestandaardiseerd getal, ook de dag van de week.
    """)
    return


@app.cell(hide_code=True)
def choice_weekend_cell():
    choice_weekend = L.choice(
        "weekend",
        question="Waarom behandelt het model van de course een zaterdag en een zondag zo verschillend?",
        options={
            "floats": "Twee voorspellingen zijn nooit precies gelijk, want floats verschillen in de laatste cijfers, dus faalt ook een test met marge",
            "right": "De dag van de week is één getal, van 0 voor zondag tot 6 voor zaterdag, dus liggen de twee voor het model ver uit elkaar",
            "data": "In 2011 hadden de zaterdagen duidelijk meer fietsen dan de zondagen, en het model leerde dat verschil goed",
            "small": "Een netwerk van 577 getallen is te klein om iets over weekenden te leren, dus gokt het model daar",
        },
        correct="right",
        why={
            "right": "De kolom weekday gaat het netwerk in als één gestandaardiseerd getal: −1,50 voor zondag en 1,49 voor zaterdag, de twee uiteinden van de week. Elk neuron ziet de dag via één weight op dat getal, dus ligt zondag voor het model naast maandag en het verst van zaterdag. Met de dag als 7 one-hot-kolommen, zoals het uur er 24 heeft, zou het model elke dag apart kunnen leren. De validation RMSE van 53,04 laat dit niet zien; een behavioral test wel, en de volgende stap is het opschrijven voor wie het model gebruikt.",
            "floats": "Floats verschillen in de laatste cijfers, met miljoensten van een fiets, en de test staat 20 fietsen toe: wat in de inputs maakt zaterdag en zondag zo verschillend voor het model?",
            "data": "In 2011 hadden de twee dagen ongeveer evenveel fietsen per uur, 142,2 op zaterdag en 143,8 op zondag: waar komt het verschil dat het model maakt dan vandaan?",
            "small": "Hetzelfde netwerk doet de drukke ochtenden van de werkdagen en de rustige nachten goed: wat in de inputs zet zaterdag en zondag zo ver uit elkaar?",
        },
    )
    choice_weekend
    return (choice_weekend,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Tests voor ML-code](/course/open/sys-code-tests?lang=nl) keken je tests naar de code; hier keken de tests naar het model, van buitenaf, zoals CheckList dat doet: een minimum dat goed moet, een antwoord dat niet mag veranderen, en een antwoord dat de goede kant op moet bewegen. Je project *ops* heeft nu drie tests meer, `test_night.py`, `test_row_number.py` en `test_rain.py`, en `conftest.py` geeft deze tests je verpakte model. De fouten die overblijven, zoals minder dan 0 fietsen 's nachts en zaterdag tegenover zondag, zijn wat een model card opschrijft: *Voor wie werkt het? Model cards en system cards* laat zien hoe, en in *Project: tests en een model card* krijgt je model van de vraag allebei. In *Project: je model als service, bewaakt op drift* beantwoordt het model verzoeken, en komen de uren van 2012 terug, met veel meer fietsers dan in 2011.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_weekend, result_quiet_night, result_rain, result_row_number):
    L.wrap_up(result_quiet_night, result_row_number, result_rain, choice_weekend)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
