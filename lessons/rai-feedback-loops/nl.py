# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-feedback-loops. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-feedback-loops", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Bias in elke stap](/course/open/rai-bias-lifecycle?lang=nl) en [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_measure():
    quick_measure = L.choice(
        "quick-measure",
        question="Een model gebruikt arrestaties als label voor criminaliteit, terwijl in sommige wijken veel meer politie rondloopt dan in andere. Welke soort bias is dat?",
        options={
            "historical": "Historical bias",
            "right": "Measurement bias",
            "deployment": "Deployment bias, want de politie gebruikt het",
        },
        correct="right",
        why={
            "historical": "De wereld kan ook ongelijk zijn, maar hier meet het label zelf criminaliteit op sommige plekken slechter. Zie *De data maken* in [Bias in elke stap](/course/open/rai-bias-lifecycle?lang=nl).",
            "right": "Een arrestatie is een vervanger voor criminaliteit, en meet die slechter waar meer politie is.",
            "deployment": "Deployment bias gaat over een model gebruiken voor iets waarvoor het niet gebouwd is. Zie *De data maken* in [Bias in elke stap](/course/open/rai-bias-lifecycle?lang=nl).",
        },
    )
    quick_measure
    return (quick_measure,)


@app.cell(hide_code=True)
def quick_check_seed():
    quick_seed = L.choice(
        "quick-seed",
        question="Je voert `np.random.default_rng(7).random(size=3)` twee keer uit. Wat krijg je?",
        options={
            "different": "Twee verschillende rijtjes van drie getallen",
            "right": "Twee keer dezelfde drie getallen",
            "sevens": "Drie getallen rond de 7",
        },
        correct="right",
        why={
            "different": "Met dezelfde seed geeft een generator elke keer dezelfde getallen. Zie *Het idee* in [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).",
            "right": "De seed legt de getallen vast: iedereen die dit uitvoert, krijgt dezelfde drie.",
            "sevens": "De seed is geen waarde om op te mikken: `rng.random` geeft getallen van 0 tot 1. Zie *Het idee* in [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).",
        },
    )
    quick_seed
    return (quick_seed,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model bepaalt waar de politie patrouilleert. Waar ze patrouilleert, ziet ze meer misdrijven, en wat ze ziet, gaat in de data waar het model daarna van leert. Zo lijkt het model gelijk te hebben, en stuurt het er nog meer politie heen. In [Waar labels vandaan komen](/course/open/data-labels?lang=nl) was de feedback loop de weg van de voorspellingen van een model terug naar de training data ervan. Hier zie je dat die loop uit zichzelf een verschil kan laten groeien, zelfs tussen twee wijken met precies evenveel misdrijven.

    **Een voorbeeld uit het dagelijks leven.** Je zoekt je sleutels alleen in je jaszak, omdat je ze daar de vorige keer vond. Je vindt ze daar steeds vaker, ook omdat je nergens anders kijkt. Volgens je geheugen liggen de sleutels meestal in je jaszak, maar het laat alleen zien waar je zocht.
    """)
    return


@app.cell(hide_code=True)
def predict_winner():
    guess_winner = L.guess(
        "winner",
        question="Twee wijken. Op een dag dat de politie er is, vindt die een misdrijf met een kans van 10 procent in wijk A en 11 procent in wijk B. Elke dag gaat de politie naar een wijk met een kans die gelijk is aan het deel van de tot nu toe gevonden misdrijven in die wijk. Welk deel van de dagen gaat op den duur naar wijk B, denk je?",
        options={
            "fair": "Zo'n 52 procent, iets meer dan de helft",
            "right": "Uiteindelijk allemaal: 100 procent",
            "same": "De helft, het verschil is klein",
        },
        reveal="**Allemaal.** Ensign en anderen (2018) lieten met een model van deze loop zien dat hij bij kansen van 10 en 11 procent met kans 1 bij wijk B uitkomt: een klassiek geval van de winnaar volgen. Ze namen zelfs aan dat de politie misdrijven precies zo vaak vindt als ze gebeuren. Met dezelfde kans in beide wijken is het niet beter: waar het deel uitkomt, is toeval. De demo hieronder laat dat zien.",
    )
    guess_winner
    return (guess_winner,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    **Een voorspelling die zichzelf waarmaakt.** In 1948 gaf Robert K. Merton dit verschijnsel een naam, in het Engels *self-fulfilling prophecy*: een onjuiste kijk op de situatie roept nieuw gedrag op, en dat gedrag maakt de oorspronkelijk onjuiste kijk waar. In zijn voorbeeld, dat zich afspeelt in 1932, gaat een gezonde bank failliet omdat de klanten een gerucht geloven dat de bank in de problemen zit, en allemaal hun geld komen halen. Het gebeurt alleen tussen mensen, voegde hij eraan toe: voorspellingen over de terugkeer van de komeet van Halley veranderen zijn baan niet. Het boek *Fairness and Machine Learning* beschrijft iets vergelijkbaars bij modellen, een zichzelf waarmakende voorspelling: politie die naar de gebieden gaat die een model riskant noemt, vindt daar meer misdrijven, dus lijkt de voorspelling te kloppen, ook als die alleen op scheve data berustte. De arrestaties uit die gebieden gaan terug in de training data, en het verschil kan met de tijd groeien.

    **Oakland.** Kristian Lum en William Isaac (2016) lieten het algoritme van PredPol, een grote Amerikaanse leverancier van zulke systemen, los op de drugsarrestaties van de politie van Oakland uit 2010. Het algoritme gebruikt maar drie dingen: de soort, de plek en de tijd van eerdere misdrijven. Drugsgebruik, geschat uit een landelijke enquête, was vrij gelijk over de stad verdeeld, maar de arrestaties zaten geconcentreerd in twee gebieden met vooral niet-witte bewoners met een laag inkomen, met ongeveer 200 keer zoveel drugsarrestaties als elders. In hun simulatie zouden zwarte mensen ongeveer twee keer zo vaak het doelwit zijn als witte mensen. Toen ze elke plek waar de politie heen ging 20 procent meer misdrijven lieten registreren, werd het model steeds zekerder dat de misdrijven zaten waar het de politie heen had gestuurd. Het was een simulatie, geen verslag van wat de politie van Oakland deed.

    **Gevonden of gemeld.** Ensign en anderen (2018) splitsen politiedata in tweeën. Wat de politie ontdekt waar het model haar heen stuurde, voedt de loop. Wat mensen melden, zoals een telefoontje naar het alarmnummer, komt binnen of het model nu iemand stuurde of niet: dat kan een loop dempen, maar niet helemaal weghalen. En ze leggen uit waarom een loop uit de hand loopt: als je de politie 90 procent van de tijd naar gebied A stuurt, moet het je niet verbazen dat ze daar negen keer zoveel misdrijven ontdekt, ook als er in beide gebieden evenveel gebeurt. Hun oplossing telt een ontdekking alleen mee met een kans die kleiner wordt naarmate de politie er vaker heen gestuurd is. Een eenvoudige versie van hetzelfde idee: deel wat gevonden is door hoe vaak iemand keek.

    **Probeer het.** De cel hieronder draait de loop van Ensign en anderen vijf keer, elke keer met een andere seed. Twee wijken hebben evenveel misdrijven: op een dag dat de politie in een wijk is, vindt die er een misdrijf met een kans van 0,3. Elke dag gaat de politie naar wijk 0 met een kans die gelijk is aan het deel van de tot nu toe gevonden misdrijven in wijk 0. De cel laat de seed zien, de gevonden misdrijven in wijk 0 en 1 na 1000 dagen, en het deel van wijk 0. Elke telling begint bij 1, zodat beide wijken op de eerste dag een kans hebben. Voer de cel uit.
    """)
    return


@app.cell
def demo_loop():
    import numpy as np

    # twee wijken met evenveel misdrijven: op een dag met politie vindt die er een misdrijf met kans 0,3
    _rate = 0.3
    for _seed in [1, 2, 3, 4, 5]:
        _rng = np.random.default_rng(_seed)
        _found = [1, 1]   # tot nu toe gevonden misdrijven in wijk 0 en 1
        for _day in range(1000):
            # de politie gaat naar wijk 0 met een kans gelijk aan het deel van de gevonden misdrijven daar
            _share = _found[0] / (_found[0] + _found[1])
            if _rng.random() < _share:
                _sent = 0
            else:
                _sent = 1
            if _rng.random() < _rate:
                _found[_sent] = _found[_sent] + 1
        print(_seed, _found, round(_found[0] / (_found[0] + _found[1]), 2))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.Html(C.picture({"title": "Vijf keer de loop", "xlabel": "dag", "ylabel": "deel gevonden in wijk 0",
                           "true": "deel van de misdrijven: 0,5", "decimal": ",",
                           "alt": "Vijf oranje lijnen van het deel van de gevonden misdrijven dat in wijk 0 ligt, over 1000 dagen. Na 200 dagen blijft elke lijn in een eigen smalle band; op dag 1000 staan ze op {finals}. Een gestippelde lijn op 0,5 is het deel van de misdrijven dat echt in wijk 0 gebeurt."})),
        mo.md(r"""
        Na 200 dagen blijft elke run in een eigen smalle band, ver van de helft, terwijl de misdrijven half om half verdeeld zijn. Wat de politie vindt, groeit met waar ze heen gestuurd wordt, dus nemen de aantallen de verdeling over in plaats van de misdrijven.

        **In Europa en Nederland.** Het Bureau van de Europese Unie voor de grondrechten (FRA) simuleerde in 2022 twee wijken met evenveel misdrijven, die begonnen met 20 en 80 procent van de patrouilles. De patrouilles verdelen naar verhouding van de geregistreerde misdrijven hield die verdeling voor altijd vast; een eenvoudig lerend model stuurde, in één run, na 40 weken alle politie naar één wijk. Feedback loops kunnen makkelijk ontstaan als alles automatisch gaat, concludeerde het FRA. De Nederlandse politie gebruikte het Criminaliteits Anticipatie Systeem (CAS) van halverwege de jaren 2010 tot december 2025. Het voorspelde misdrijven in vakken van 125 bij 125 meter, uit aangiften en meldingen van burgers en uit bekende verdachten die in de buurt wonen; in de termen van Ensign en anderen dempen meldingen een loop. Amnesty International (2020) beschreef een loop in een ander Nederlands systeem, rond Roermond: een auto die het systeem markeert en waar de politie op afgaat, blijft in politiedatabases aan het profiel gekoppeld, ook als de controle niets oplevert, dus de volgende keer dat hij daar rijdt, schat het systeem het risico hoger in.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **zichzelf waarmakende voorspelling** | een voorspelling die verandert wat er gebeurt of wat er gemeten wordt, zodat de data haar gelijk lijkt te geven; in het Engels *self-fulfilling prediction* |
        | **ontdekt incident** | een misdrijf dat de politie vond waar een systeem haar heen stuurde; het voedt de loop |
        | **gemeld incident** | een misdrijf dat mensen zelf melden, bijvoorbeeld met een telefoontje; het komt binnen zonder het systeem |
        | **runaway feedback loop** | een feedback loop die een verschil steeds groter maakt, tot bijna alles naar één plek gaat |
        """),
        L.watch_out("Een feedback loop heeft geen scheve data nodig om te beginnen. Ensign en anderen namen aan dat de politie misdrijven precies zo vaak vindt als ze gebeuren, en toch liep hun loop uit de hand. Een scheef begin, zoals de arrestaties in Oakland, maakt het alleen erger."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een feedback loop op te sporen:

    1. Volg de beslissingen van een model: wat doen mensen ermee?
    2. Vraag je af of dat de data verandert die terugkomt: naar wie wordt gekeken, wie krijgt een beslissing en dus een uitkomst, en wie nooit?
    3. Splits de data in wat gevonden is door het model, en wat zonder het model binnenkwam.
    4. Deel wat gevonden is door hoe vaak iemand keek: per dag met politie, per controle, per aangenomen persoon.
    5. Groeit een verschil terwijl de wereld niet veranderde, vertrouw dan de totalen niet. Tel ontdekkingen naar verhouding van de moeite, en gebruik data die niet van het model afhangt, in de wetenschap dat die een loop dempt maar niet weghaalt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De gegevens van één jaar voor twee wijken, verzonnen door de course. `_found` bevat de misdrijven die de politie in elke wijk vond. De loop rekent voor elke wijk het deel uit van alle gevonden misdrijven, zoals een rapport of een model het zou zien. Voer de cel uit.
    """)
    return


@app.cell
def example_share():
    # verzonnen gegevens van één jaar: misdrijven die de politie vond, per wijk
    _found = {"North": 99, "South": 11}
    _total = _found["North"] + _found["South"]

    _share = {}
    for _district in _found:
        _share[_district] = round(_found[_district] / _total, 2)
    print(_share)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    North heeft 90 procent van de gevonden misdrijven: negen keer zoveel als South. Een model dat hiervan leert, stuurt nog meer politie naar North. Maar hoe vaak was de politie daar? Oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_per_day():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Per dag met politie

        `days` bevat op hoeveel dagen van het jaar de politie in elke wijk was, en `found` de misdrijven die ze daar vond. Vervang de `...` door een loop over de wijken, zoals in het uitgewerkte voorbeeld, met de namen zonder liggend streepje: `per_day` en `district`. Zet in `per_day` voor elke wijk de gevonden misdrijven gedeeld door de dagen met politie, afgerond op twee decimalen. De laatste regel is `per_day`, zodat de check het ziet. De check probeert ook andere getallen.
        """),
        L.task("Reken `per_day` uit: voor elke wijk de gevonden misdrijven per dag met politie."),
    ])
    return


@app.cell(hide_code=True)
def exercise_per_day():
    box_per_day = L.code_box("per-day", lines=10)
    box_per_day
    return (box_per_day,)


@app.cell(hide_code=True)
def check_per_day(box_per_day):
    result_per_day = L.check("per-day", box_per_day.value)
    result_per_day
    return (result_per_day,)


@app.cell(hide_code=True)
def hints_per_day():
    L.hints(
        "per-day",
        [
            "Gebruik de loop van het uitgewerkte voorbeeld. Alleen de deling verandert: niet door het totaal van alle gevonden misdrijven, maar door de dagen dat de politie in die wijk was.",
            "De dagen dat de politie in een wijk was, staan in `days`, onder de naam van de wijk: `days[district]`.",
            "`per_day[district] = round(found[district] / ___, 2)`.",
        ],
        why="`per_day = {}`, dan `for district in found:` met daaronder `per_day[district] = round(found[district] / days[district], 2)`. Dat geeft `{'North': 0.3, 'South': 0.31}`: 99 misdrijven in 330 dagen, en 11 in 35.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_sources():
    mo.md(r"""
    ### Oefening 2. Gevonden of gemeld?

    Zes manieren waarop een misdrijf bij de politie komt. Deel elk geval in: gevonden omdat een systeem iemand stuurde, of door mensen zelf gemeld.
    """)
    return


@app.cell(hide_code=True)
def sort_sources_cell():
    sort_sources = L.sort("sources")
    sort_sources
    return (sort_sources,)


@app.cell(hide_code=True)
def check_sources(sort_sources):
    result_sources = L.check("sources", sort_sources)
    result_sources
    return (result_sources,)


@app.cell(hide_code=True)
def hints_sources():
    L.hints(
        "sources",
        [
            "Vraag je bij elk geval af: was dit misdrijf ook bij de politie gekomen als geen systeem iemand had gestuurd?",
            "De noodoproep en de aangifte op het bureau komen binnen zonder systeem. Welk geval, naast de patrouille en de controle, is gevonden omdat een systeem ernaar wees?",
            "Gevonden omdat het systeem iemand stuurde: op patrouille, een controle en \\_\\_\\_. Door mensen zelf gemeld: \\_\\_\\_, op het bureau en online.",
        ],
        why="**Gevonden omdat het systeem iemand stuurde:** op patrouille, een controle en een gemarkeerde auto. **Door mensen zelf gemeld:** de noodoproep, op het bureau en online.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_hiring():
    mo.md(r"""
    ### Oefening 3. Buiten de politie
    """)
    return


@app.cell(hide_code=True)
def choice_hiring_cell():
    choice_hiring = L.choice(
        "hiring",
        question="Een model voor werving leert alleen van hoe het de mensen die het aannam op hun werk verging. In de loop van de jaren neemt het steeds minder mensen uit groep X aan. Is dit een feedback loop?",
        options={
            "no": "Nee, hier wordt niemand opgepakt",
            "right": "Ja, zijn keuzes bepalen wat het leert",
            "named": "Alleen als groep X een kolom in de data is",
        },
        correct="right",
        why={
            "no": "Een feedback loop heeft geen politie nodig: Ensign en anderen noemen ook werving, leningen en toelating tot een opleiding. Zie *Gevonden of gemeld* in *Het idee*.",
            "right": "Het ziet alleen hoe het de aangenomen mensen verging. De mensen uit groep X die het afwees, krijgen nooit een uitkomst, dus kan het nooit leren dat het zich in hen vergiste.",
            "named": "De loop heeft geen kolom nodig: hij loopt via wie een beslissing krijgt, en dus een uitkomst. Zie *Oakland* in *Het idee*.",
        },
    )
    choice_hiring
    return (choice_hiring,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Bias in elke stap](/course/open/rai-bias-lifecycle?lang=nl) liep een gestippelde pijl van het gebruik van een model terug naar de wereld; hier zag je wat die pijl kan doen. De beslissingen van een model veranderen wat er daarna gemeten wordt, dus drijven zelfs twee wijken met evenveel misdrijven uit elkaar, en lijkt de data het model gelijk te geven. In [Waar labels vandaan komen](/course/open/data-labels?lang=nl) zag je dat een model alleen leert over wat eerdere keuzes doorlieten; hier zag je hoe ver die loop kan gaan. Jouw deling door de dagen met politie is een eerste hulpmiddel ertegen. De volgende les, *Een kolom weglaten is niet genoeg*, laat een andere manier zien waarop een model groepen verschillend kan behandelen zonder te horen wie in welke groep zit.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_hiring, result_per_day, result_sources):
    L.wrap_up(result_per_day, result_sources, choice_hiring)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
