# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-mlops. Route: course/route/lessen/systemen.yaml. Checks, de teams, de drift van twee jaar en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-mlops", "nl")
    C = L.checks   # de teams, de drift van twee jaar, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Project: je model als service, bewaakt op drift](/course/open/sys-project-service?lang=nl) en [Een doos die overal draait: containers](/course/open/sys-containers?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_retrain():
    quick_retrain = L.choice(
        "quick-retrain",
        question="In het project noemde de functie train_again de maanden waarin het fietsmodel opnieuw getraind moest worden. Wanneer telde een maand mee?",
        options={
            "right": "Als de RMSE meer was dan 1,5 keer de RMSE op de validatieset",
            "psi": "Als de PSI van de antwoorden van het model boven 0,1 kwam",
            "every": "Elke maand, want de wereld verandert de hele tijd",
            "slow": "Als het 99e percentiel van de antwoorden langer dan een seconde duurde, want een traag model moet weg",
        },
        correct="right",
        why={
            "right": "train_again vergeleek de RMSE van elke maand van 2012 met 1,5 × 53,04 = 79,56, en noemde februari tot en met december. In het project moest een mens de lijst nog lezen en de training starten. Deze les zet de stap in een kringloop die uit zichzelf draait.",
            "psi": "De PSI van de antwoorden bleef het hele jaar onder 0,1, terwijl de fout groeide. Welk getal vergeleek train_again? Zie [Project: je model als service, bewaakt op drift](/course/open/sys-project-service?lang=nl).",
            "every": "Elke maand trainen kost tijd en compute, ook in maanden waarin het model nog goed is. Welk getal besliste? Zie [Project: je model als service, bewaakt op drift](/course/open/sys-project-service?lang=nl).",
            "slow": "Een traag antwoord is een zaak van de service, niet van de training. Welk getal van het model besliste? Zie [Project: je model als service, bewaakt op drift](/course/open/sys-project-service?lang=nl).",
        },
    )
    quick_retrain
    return (quick_retrain,)


@app.cell(hide_code=True)
def quick_check_layers():
    quick_layers = L.choice(
        "quick-layers",
        question="In de snelle Dockerfile verandert alleen service.py. Welke stappen worden opnieuw gebouwd?",
        options={
            "all": "Alle stappen, want een nieuw image begint bij niets",
            "right": "Alleen de laatste stap: het kopiëren van service.py",
            "pip": "De stap met pip en elke stap daarna, want packages kunnen op elk moment veranderen",
            "none": "Geen: het oude image draait nog",
        },
        correct="right",
        why={
            "right": "Elke stap van een Dockerfile is een laag, en een laag wordt alleen opnieuw gebouwd als de eigen stap of een laag eronder verandert. Met service.py als laatste gekopieerd bouwt een wijziging daarin één kleine laag opnieuw, in een tiende seconde. In deze les is het image bouwen één stap van een pipeline, na de tests.",
            "all": "Docker bewaart elke laag waarvan de stap en de bestanden niet veranderden. Welke lagen veranderden wel? Zie [Een doos die overal draait: containers](/course/open/sys-containers?lang=nl).",
            "pip": "De packages staan in requirements.txt, en dat bestand veranderde niet, dus blijft de laag met pip staan. Welke laag bevat service.py? Zie [Een doos die overal draait: containers](/course/open/sys-containers?lang=nl).",
            "none": "Het oude image bevat de oude service.py. Wat doet een build met de ene stap waarvan het bestand veranderde? Zie [Een doos die overal draait: containers](/course/open/sys-containers?lang=nl).",
        },
    )
    quick_layers
    return (quick_layers,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In deze module bouwde je elke stap rond een model met de hand: bijgehouden runs, versies, tests, een service met een log, een meting van drift, een container. In veel teams is elke stap het werk van een mens: iemand traint in een notebook, leest de getallen, kopieert een bestand naar een server, en kijkt af en toe naar de monitoring. Dat werkt voor één model, één keer. Het breekt als de wereld elke maand verandert en het model moet bijblijven: dan is elke stap met de hand wachten, en zolang het team wacht antwoordt een slechter model. MLOps zet de stappen samen in één kringloop die uit zichzelf draait, en een volwassenheidsmodel vertelt een team hoe ver het is en welke stap het als volgende automatiseert.

    **Een voorbeeld uit het dagelijks leven.** Een thuiskok bakt brood met de hand: wegen, kneden, bakken, proeven. Een bakkerij die elke dag brood verkoopt zet dezelfde stappen in machines, en controleert elke partij voordat de partij naar de winkel gaat: een partij die afgekeurd wordt komt nooit in het schap. Als het meel verandert ziet de bakkerij dat in de eigen metingen en past het recept aan voordat klanten klagen. En de weg van keuken naar bakkerij gaat machine voor machine.
    """)
    return


@app.cell(hide_code=True)
def predict_wait():
    guess_wait = L.guess(
        "wait",
        question="De fout van een model groeit elke maand een beetje, doordat de wereld verandert. Is de fout meer dan 1,5 keer de fout net na de training, dan gaat er een alert af. Team A traint dan met de hand: er gaan nog twee maanden voorbij tot het nieuwe model in productie staat. De pipeline van team B traint, test en rolt het nieuwe model zelf uit, al in de volgende maand. Over twee jaar draait bij team B in 4 maanden een model boven de grens. En bij team A, denk je?",
        options={
            "same": "Ook 4 maanden, want dezelfde alert start beide teams",
            "two": "6 maanden: twee maanden meer, maar alleen bij de eerste alert",
            "right": "9 maanden: ongeveer drie maanden boven de grens bij elke alert",
            "all": "Alle 24 maanden, want een team dat met de hand werkt haalt de achterstand nooit meer in",
        },
        reveal="**9 maanden.** Team A krijgt drie alerts, en elke alert kost drie maanden boven de grens: de maand van de alert en de twee maanden wachten, waarin de fout verder groeit. Team B krijgt vier alerts, elk van één maand. De maanden wachten tellen veel zwaarder dan het aantal alerts: met drie maanden wachten zou de helft van de twee jaar boven de grens liggen. Oefening 2 telt dit, en de demo laat het maand voor maand zien.",
    )
    guess_wait
    return (guess_wait,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Eén kringloop.** **MLOps**, kort voor machine learning operations, is de manier van werken die een model goed houdt in productie, met de tools erbij. De kern is één kringloop van de stappen van deze module. De training draait bijgehouden en herhaalbaar, zoals in [Je runs bijhouden: experiment tracking](/course/open/sys-experiment-tracking?lang=nl). Tests controleren de code en wat het model doet, zoals in [Tests voor ML-code](/course/open/sys-code-tests?lang=nl) en [Doet het model wat het moet? Behavioral tests](/course/open/sys-behavior-tests?lang=nl). Een model dat slaagt gaat naar buiten als versie in een registry en als image, zoals in [Welk model is dit? Versies en een model registry](/course/open/sys-model-versions?lang=nl). Het model antwoordt achter een API, en monitoring bewaakt de antwoorden en de inputs, zoals in [Een draaiend model in de gaten houden: metrics, logs en alerts](/course/open/sys-monitoring?lang=nl) en [Als de wereld verandert: drift](/course/open/sys-drift?lang=nl). Ziet de monitoring het model slechter worden, dan begint de kringloop opnieuw bij de training, met nieuwe data.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **CI/CD.** De woorden komen uit de softwarewereld. Bij continuous integration wordt elke wijziging in de code vanzelf gebouwd en getest; bij continuous delivery of deployment gaat een wijziging die de tests haalt vanzelf naar productie, of met één klik. **CI/CD** is het paar. Bij ML controleren de tests meer dan code: ook de data en het model. En er wordt meer geleverd dan een service: ook de pipeline die het model traint.

        **Continuous training.** Bij **continuous training** traint de pipeline het model vanzelf opnieuw. De pagina van Google noemt wanneer: op verzoek, volgens een rooster, als er nieuwe data binnenkomt, als het model slechter wordt, en als de data verschuift. Het nieuwe model haalt dezelfde tests als het eerste, en vervangt het model in productie alleen als het nieuwe model beter is: de pipeline moet nee kunnen zeggen.

        **Hoe ver is een team?** Een **volwassenheidsmodel** beschrijft de weg van alles met de hand naar een kringloop die uit zichzelf draait, in niveaus. De pagina van Google Cloud heeft er drie: niveau 0, een handmatig proces; niveau 1, ML pipeline automation, met continuous training; niveau 2, CI/CD pipeline automation, waarin ook wijzigingen aan de pipeline zelf vanzelf getest en geleverd worden. De pagina van Microsoft, voor het laatst bijgewerkt op 3 juni 2026, heeft er vijf:

        | Niveau | Naam | Wat vanzelf draait |
        |---|---|---|
        | 0 | No MLOps | niets: training, uitrollen en de applicatie allemaal met de hand |
        | 1 | DevOps but no MLOps | de code van de applicatie wordt vanzelf gebouwd, getest en uitgerold; het model wordt met de hand getraind en overgedragen |
        | 2 | Automated training | de training draait als pipeline, met bijgehouden runs; een model uitrollen gaat nog met de hand |
        | 3 | Automated model deployment | een model dat de tests haalt gaat via CI/CD naar productie |
        | 4 | Full MLOps automated operations | getallen uit productie starten vanzelf een nieuwe training |

        Beide pagina's zijn gelezen op 28 november 2026. Microsoft merkt op dat een team vaak trekken van meer dan één niveau tegelijk heeft. Deze les gebruikt een eenvoudige lezing: een team staat op het hoogste niveau waarvan de stap, en elke stap eronder, vanzelf draait. Bij een volwassenheidsmodel gaat het niet om het getal: het gaat om de volgende stap, de laagste stap die een mens nog doet.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **MLOps** | | de manier van werken die een model goed houdt in productie, met de tools erbij: bijgehouden training, tests, uitrollen en monitoring in één kringloop | |
        | **CI/CD** | | elke wijziging in de code vanzelf gebouwd en getest (continuous integration), en uitgerold als de tests slagen (continuous delivery of deployment) | |
        | **continuous training** | | een pipeline die het model vanzelf opnieuw traint, volgens een rooster, op nieuwe data of op een alert, en het model pas uitrolt na dezelfde tests | |
        | **volwassenheidsmodel** | | een reeks niveaus van alles met de hand naar alles vanzelf, om te zien waar een team staat en wat er nu komt | |
        """),
        L.watch_out("Automatiseer je een stap die je nooit met de hand deed, dan maakt de pipeline dezelfde fouten, alleen sneller. Doe elke stap eerst met de hand, zoals in deze module, en automatiseer de stap als je weet hoe goed eruitziet."),
        L.watch_out("Continuous training zonder de tests van de stap uitrollen kan uit zichzelf een slechter model in productie zetten, bijvoorbeeld als een partij nieuwe data kapot is. Zet bij elke automatische uitrol een test die de uitrol kan tegenhouden."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Hoe ver is een team, en wat is de volgende stap:

    1. Schrijf de stappen van de kringloop op: trainen, testen, uitrollen, draaien, bewaken, en terug naar trainen.
    2. Vraag bij elke stap: draait de stap vanzelf, of doet een mens de stap, en hoe lang duurt dat?
    3. Tel vanaf onderen: het niveau is het aantal stappen op een rij dat vanzelf draait.
    4. Automatiseer de eerste stap die een mens nog doet, met een test die de pipeline kan stoppen.
    5. Meet wat het oplevert: de maanden van een alert tot een nieuw model, en de maanden boven de grens.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De stap uitrollen van een pipeline, voor het fietsmodel: een nieuw model gaat alleen in productie als elke test slaagt en de RMSE op de validatieset lager is dan de RMSE van het model in productie. In de cel bevat `tests` de uitkomst van elke test, `new_rmse` de RMSE van het nieuwe model, en `old_rmse` de 53,04 van het model in productie. De loop zet `passed` op `False` bij een test die faalt. Voer de cel uit.
    """)
    return


@app.cell
def example_gate():
    tests = {"vormen": True, "loss daalt": True, "warmer, meer fietsen": True}
    new_rmse = 51.2      # het nieuwe model, op de validatieset
    old_rmse = 53.04     # het model in productie
    passed = True
    for _name in tests:
        if not tests[_name]:
            passed = False
    release = passed and new_rmse < old_rmse
    print("alle tests geslaagd:", passed)
    print("het nieuwe model uitrollen:", release)
    return new_rmse, old_rmse, passed, release, tests


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Alle drie de tests slagen, en 51,2 is lager dan 53,04, dus gaat het nieuwe model naar buiten. Zet één test op `False`, en `passed` wordt `False`: het model blijft binnen, hoe goed de RMSE ook is. Zet `new_rmse` op `53.04`, en het oude model blijft: een nieuw model moet beter zijn, niet even goed, want elke uitrol brengt een risico mee. Deze poort maakt continuous training veilig. Oefening 1 zet eerst een team op het volwassenheidsmodel.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_level():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Hoe ver is een team?

        `level(team, steps)` krijgt een team en de stappen van de niveaus, op volgorde. `C.STEPS` is `["code", "training", "release", "retraining"]`: de code van de applicatie, automatisch gebouwd en getest (niveau 1), de training als pipeline (niveau 2), het uitrollen door de pipeline (niveau 3), en opnieuw trainen op een seintje van de monitoring (niveau 4). Een team is een dict met `True` voor een automatische stap en `False` voor een stap die een mens doet. Het niveau is het aantal stappen op een rij vanaf het begin dat vanzelf draait. De `for`-loop loopt de stappen langs, `n` telt de stappen, en `break` stopt de loop. Vervang de `...` door de vraag of een mens deze `step` nog doet.

        `C.TEAMS` bevat vier teams. De check roept `level(C.TEAMS["shop"], C.STEPS)` aan: een winkel waar de code en de training vanzelf draaien, maar waar iemand het model met de hand naar de server kopieert. Daarna een fietsverhuur die automatisch uitrolt maar met de hand traint, een bank, een lab, een team met elke stap automatisch, en twee teams met andere stappen.
        """),
        L.task("Tel de automatische stappen, tot de eerste stap die een mens doet."),
    ])
    return


@app.cell(hide_code=True)
def exercise_level():
    box_level = L.code_box("level", lines=8)
    box_level
    return (box_level,)


@app.cell(hide_code=True)
def check_level(box_level):
    result_level = L.check("level", box_level.value)
    result_level
    return (result_level,)


@app.cell(hide_code=True)
def hints_level():
    L.hints(
        "level",
        [
            "Een stap die een mens doet heeft `False` in `team`. De loop moet daar stoppen: de vraag na `if` is waar als de waarde van `step` in `team` `False` is. Hoe zoek je een waarde op in een dict, en hoe maak je van `False` een vraag die waar is?",
            "In het uitgewerkte voorbeeld was `if not tests[_name]:` waar voor een test die faalde. Bij de fietsverhuur ziet de loop eerst `True` bij de code, dan `False` bij de training: daar moet de loop stoppen, met `n` op 1.",
            "`        if not ___:`",
        ],
        why="`if not team[step]:` is waar bij de eerste stap die een mens doet, en `break` stopt daar het tellen. De winkel heeft `True`, `True`, `False`: niveau 2. De fietsverhuur heeft `True`, `False`, `True`: niveau 1, want automatisch uitrollen telt niet boven een training met de hand.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_bad_months():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Maanden boven de grens

        `bad_months(drift, limit, delay)` volgt een model in productie, maand na maand. `drift` bevat voor elke maand hoeveel de fout groeit: `1.06` is 6% meer. De `for`-loop neemt de groei van elke maand als `grow`. `error` begint bij `1.0`, de fout net na de training. Ligt `error` in een maand boven `limit`, dan zit het model te vaak fout: `bad` telt de maand mee, en de alert gaat af. Het team heeft dan nog `delay` maanden nodig tot het nieuwe model in productie staat: `waited` telt de maanden sinds de alert, en zodra het nieuwe model er is gaat `error` terug naar `1.0` en `waited` naar 0. Vervang de `...` door de vraag of het team lang genoeg heeft gewacht.

        `C.DRIFT` bevat de 24 maanden van twee jaar. De check roept `bad_months(C.DRIFT, 1.5, 2)` aan: een team dat met de hand traint en na een alert nog 2 maanden nodig heeft. Daarna 0, 1 en 3 maanden wachten, een lagere grens, een wereld die niet verandert, een wereld die snel verandert, en één enkele maand.
        """),
        L.task("Laat het nieuwe model komen zodra het team lang genoeg heeft gewacht."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bad_months():
    box_bad_months = L.code_box("bad_months", lines=15)
    box_bad_months
    return (box_bad_months,)


@app.cell(hide_code=True)
def check_bad_months(box_bad_months):
    result_bad_months = L.check("bad_months", box_bad_months.value)
    result_bad_months
    return (result_bad_months,)


@app.cell(hide_code=True)
def hints_bad_months():
    L.hints(
        "bad_months",
        [
            "Bij elke maand boven de grens is de vraag: is het nieuwe model er nu? `waited` telt de maanden wachten tot nu toe, en `delay` is hoeveel maanden het team nodig heeft. Wanneer zijn de twee gelijk?",
            "Met `delay` op 0 komt het nieuwe model meteen: bij de eerste maand boven de grens is `waited` 0, en gaat `error` terug naar `1.0`. Met `delay` op 2 gaan eerst de maand van de alert en nog twee maanden voorbij.",
            "`            if waited ___:`",
        ],
        why="`if waited == delay:` laat het nieuwe model komen na precies `delay` maanden wachten. Met 2 maanden kost elk van de drie alerts van `C.DRIFT` drie maanden boven de grens: 9 in totaal. Met 0 maanden zijn er vier alerts van elk één maand: 4.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_loop():
    mo.md(r"""
    ### Oefening 3. Welke stap van de kringloop?

    Acht werkwijzen uit deze module. Zet elke werkwijze onder de stap van de kringloop waar de werkwijze bij hoort: trainen, testen, uitrollen of bewaken.
    """)
    return


@app.cell(hide_code=True)
def sort_loop_cell():
    sort_loop = L.sort("loop")
    sort_loop
    return (sort_loop,)


@app.cell(hide_code=True)
def check_loop(sort_loop):
    result_loop = L.check("loop", sort_loop)
    result_loop
    return (result_loop,)


@app.cell(hide_code=True)
def hints_loop():
    L.hints(
        "loop",
        [
            "De afbeelding van de kringloop, onder *Eén kringloop*. Vraag bij elke werkwijze: gebeurt dit terwijl een model gemaakt wordt, voordat het model naar buiten gaat, terwijl het model naar buiten gaat, of terwijl het model antwoordt?",
            "In elke kolom horen twee werkwijzen.",
            "Trainen: de seeds en \\_\\_\\_. Testen: de loss op één batch en \\_\\_\\_. Uitrollen: de registry en \\_\\_\\_. Bewaken: het 99e percentiel en \\_\\_\\_.",
        ],
        why="**Trainen:** seeds en vaste versies, en een regel per run. **Testen:** de loss op één batch, en warmer weer met meer fietsen. **Uitrollen:** een versie in de registry, en een container image. **Bewaken:** het 99e percentiel, en de PSI per maand. Een pipeline draait deze stappen na elkaar, en de stap bewaken start de training opnieuw.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_next():
    mo.md(r"""
    ### Oefening 4. De volgende stap

    Een team laat de code van de eigen service vanzelf bouwen en testen. De training draait elke nacht als pipeline, en elke run wordt bijgehouden. Eens per maand leest een data scientist de getallen en kopieert het beste model met de hand naar de server.
    """)
    return


@app.cell(hide_code=True)
def choice_next_cell():
    choice_next = L.choice(
        "next",
        question="Wat is de volgende stap van dit team?",
        options={
            "retrain": "Opnieuw trainen dat vanzelf start als de monitoring drift ziet",
            "code": "Automatisch bouwen en testen van de code van de service",
            "hourly": "Elk uur trainen in plaats van elke nacht, zodat het model altijd vers is en nooit achterloopt",
            "right": "Uitrollen door de pipeline: een model dat de tests haalt gaat vanzelf naar buiten",
        },
        correct="right",
        why={
            "right": "De code en de training draaien vanzelf: niveau 2. De eerste stap die een mens nog doet is het uitrollen, dus is dat de volgende stap, naar niveau 3. Pas daarna loont opnieuw trainen uit zichzelf, want anders wacht elk nieuw model op de kopie met de hand.",
            "retrain": "Vanzelf opnieuw trainen is niveau 4, alleen wacht elk nieuw model dan nog op de kopie met de hand. Welke stap doet een mens nog?",
            "code": "De code wordt al vanzelf gebouwd en getest. Welke stap doet een mens nog?",
            "hourly": "Vaker trainen brengt het model niet sneller naar de server. Welke stap doet een mens nog?",
        },
    )
    choice_next
    return (choice_next,)


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Hoe lang mag een team wachten?

    Het model van oefening 2, twee jaar in productie, met de drift van `C.DRIFT` en de grens op 1,5. Kies hoeveel maanden er na een alert voorbijgaan tot het nieuwe model er is: 0 voor een pipeline die uit zichzelf traint, test en uitrolt, meer voor een team dat met de hand werkt.
    """)
    return


@app.cell(hide_code=True)
def demo_delay():
    L.explore("delay", "maanden wachten na een alert", lambda _v: C.frame(_v, L.lang), start=2,
              values=C.DELAYS, role="", task="Ga de vier waarden langs: wat voegt elke maand wachten toe, en waarom krijgt een team dat wacht minder alerts?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Zonder wachten liggen 4 van de 24 maanden boven de grens; met 3 maanden wachten 12: de helft van de twee jaar. Een team dat wacht krijgt minder alerts, drie in plaats van vier, want de nieuwe modellen komen later, alleen duurt elke alert langer. En een vast plan dat elke 6 maanden traint, alert of niet, traint ook 4 keer, even vaak als de pipeline, en heeft toch 10 maanden boven de grens: het plan traint als de kalender dat zegt, niet als de fout dat vraagt. Continuous training traint op de momenten waarop de fout erom vraagt, en snel.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Project: je model als service, bewaakt op drift](/course/open/sys-project-service?lang=nl) noemde train_again de maanden voor een nieuwe training, en moest een mens met de lijst aan de slag; in [Een doos die overal draait: containers](/course/open/sys-containers?lang=nl) werd de service een image dat elke server kan starten. MLOps zet deze stappen, en de andere van deze module, in één kringloop die uit zichzelf draait, en een volwassenheidsmodel laat zien welke stap je als volgende automatiseert. In de volgende les, *Is de data goed? Vier manieren om te kijken*, krijgt de kringloop een stap aan het begin: de data controleren voordat de data de training in gaat.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_next, result_bad_months, result_level, result_loop):
    L.wrap_up(result_level, result_bad_months, result_loop, choice_next)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
