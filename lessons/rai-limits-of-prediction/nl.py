# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-limits-of-prediction. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-limits-of-prediction", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Automatisch over mensen beslissen](/course/open/rai-automated-decisions?lang=nl), en één over [Hoe ver ernaast? MAE, RMSE en R²](/course/open/ml-regression-metrics?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_fact():
    quick_fact = L.choice(
        "quick-fact",
        question="Welk van deze besluiten kun je het veiligst automatiseren?",
        options={
            "right": "Controleren of iemand 18 jaar of ouder is",
            "fraud": "Beslissen of iemand fraude heeft gepleegd",
            "child": "Beslissen bij welke ouder een kind woont",
        },
        correct="right",
        why={
            "right": "Het controleert een feit, elke keer op dezelfde manier, zolang mensen een fout in het bestand nog kunnen laten verbeteren.",
            "fraud": "Of iemand wilde misleiden, is een oordeel over omstandigheden. Zie *Het idee* in [Automatisch over mensen beslissen](/course/open/rai-automated-decisions?lang=nl).",
            "child": "Dat weegt de omstandigheden van één gezin. Zie *Het idee* in [Automatisch over mensen beslissen](/course/open/rai-automated-decisions?lang=nl).",
        },
    )
    quick_fact
    return (quick_fact,)


@app.cell(hide_code=True)
def quick_check_r2():
    quick_r2 = L.choice(
        "quick-r2",
        question="De R² van een model op de test set is 0. Wat betekent dat?",
        options={
            "perfect": "De voorspellingen kloppen voor elk label precies",
            "half": "Het heeft ongeveer de helft van de labels goed",
            "right": "Niet beter dan altijd het gemiddelde gokken",
        },
        correct="right",
        why={
            "perfect": "Dat zou een R² van 1 zijn. Zie *Het idee* in [Hoe ver ernaast? MAE, RMSE en R²](/course/open/ml-regression-metrics?lang=nl).",
            "half": "R² is geen aandeel goede antwoorden, maar het deel van de spreiding van de labels dat het model verklaart. Zie *Het idee* in [Hoe ver ernaast? MAE, RMSE en R²](/course/open/ml-regression-metrics?lang=nl).",
            "right": "R² vergelijkt de MSE met de variantie, de MSE van altijd het gemiddelde voorspellen: bij 0 zijn ze gelijk.",
        },
    )
    quick_r2
    return (quick_r2,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Scholen, rechtbanken en jeugdzorg willen graag vooraf weten wie gaat uitvallen, opnieuw een misdrijf pleegt of hulp nodig heeft: dan kunnen ze vroeg ingrijpen. Systemen die dat beloven, worden verkocht en gebruikt. Voordat een voorspelling iets over één persoon beslist, komen er twee vragen eerst: hoe goed is deze uitkomst überhaupt te voorspellen, en is wat het model voorspelt wel waar het je om gaat?

    **Een voorbeeld uit het dagelijks leven.** Een weerbericht voor morgen klopt vaak. Een weerbericht voor één dag over drie maanden is nauwelijks beter dan het gewone weer voor die tijd van het jaar.
    """)
    return


@app.cell(hide_code=True)
def predict_challenge():
    guess_challenge = L.guess(
        "challenge",
        question="160 teams kregen zo'n 13.000 gegevens over elk van duizenden gezinnen, vanaf de geboorte van een kind, en probeerden de schoolcijfers van het kind op 15-jarige leeftijd te voorspellen. Welke R² denk je dat het beste team haalde?",
        options={
            "right": "Ongeveer 0,2",
            "half": "Ongeveer 0,5",
            "high": "Ongeveer 0,9",
        },
        reveal="**Ongeveer 0,2**, voor de schoolcijfers en voor geldnood in het gezin, en bijna 0 voor de vier andere uitkomsten. En de beste teams waren maar een beetje beter dan een eenvoudig model met vier variabelen die een deskundige koos (Salganik en collega's, 2020).",
    )
    guess_challenge
    return (guess_challenge,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De *Fragile Families Challenge* vroeg hoe ver je een leven kunt voorspellen. Een onderzoek had zo'n 4.000 gezinnen gevolgd sinds de geboorte van hun kind. In 2017 kregen 160 teams de gegevens tot het negende jaar van het kind, en probeerden ze zes uitkomsten op 15-jarige leeftijd te voorspellen, zoals de schoolcijfers van het kind en of het gezin uit huis werd gezet. De beste R² was ongeveer 0,2 voor de cijfers en voor geldnood, en bijna 0 voor de andere vier. De teams waren niet slecht: voorspellen uit het verleden werkte hier gewoon slecht. Waarom, is nog open; een mogelijke verklaring is dat een leven afhangt van veel wat in geen enkele data staat.

        De informaticus Arvind Narayanan deelt taken voor AI in drie soorten in. *Waarneming*, zoals spraak naar tekst of een scan lezen, heeft echt vooruitgang geboekt: het antwoord zit in de data. *Oordelen automatiseren*, zoals spamfilters, is verre van perfect maar wordt beter. *Sociale uitkomsten voorspellen*, zoals wie opnieuw een misdrijf pleegt of goed werk zal doen, noemt hij principieel twijfelachtig.

        Er is een tweede probleem. Wat je kunt meten, is vaak niet wat je bedoelt. Amerikaanse ziekenhuizen gebruikten een algoritme dat patiënten voor extra zorg koos door hun zorgkosten te voorspellen, als plaatsvervanger voor hoe ziek ze waren. Maar aan zwarte patiënten die even ziek waren, werd minder uitgegeven, dus bij dezelfde score waren ze zieker. Als patiënten met dezelfde score even ziek waren geweest, was het aandeel zwarte patiënten dat voor extra zorg gekozen werd, gestegen van 17,7 naar 46,5 procent (Obermeyer en collega's, 2019). Dat gat tussen het label waarop een model getraind wordt en de vraag die je beantwoord wilt hebben, heet een **target-goal mismatch**.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **target-goal mismatch** | het gat tussen wat een model leert voorspellen, omdat het te meten is, en wat je eigenlijk wilt weten, zoals zorgkosten in plaats van zorgbehoefte |
        """),
        L.watch_out("Een R² van 0,2 betekent niet dat elke voorspelling er een beetje naast zit. Voor sommige mensen zit de voorspelling er ver naast, en een besluit over één persoon rust op die ene voorspelling."),
        mo.Html(C.picture({"title": "Tien kinderen: echt cijfer op 15 en de voorspelling", "xlabel": "kind", "ylabel": "gemiddeld cijfer (1 tot 4)",
                           "mean": "gemiddelde 2,8",
                           "alt": "Tien verzonnen kinderen, met hun echte gemiddelde cijfer op 15 en de voorspelling van een model. De voorspellingen blijven tussen 2,4 en 3,1, dicht bij het gemiddelde van 2,8, terwijl de echte cijfers lopen van 1,75 tot 4,0. Bij kind 4, 5, 8 en 9 zit de voorspelling er 0,85 of 0,9 naast."})),
        mo.md(r"""
        Verzonnen door de course. Het cijfer is een gemiddeld cijfer van 1 tot 4. Een gevulde stip is het echte cijfer van een kind, een open stip de voorspelling van het model; de lijn ertussen is de misser.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Voordat een voorspelling iets over een persoon beslist:

    1. Schrijf het doel op dat je bedoelt, en het target dat de data kunnen meten.
    2. Vraag hoe die twee verschillen, en voor wie.
    3. Zoek uit hoe goed iemand een uitkomst als deze al heeft voorspeld, en vergelijk met een eenvoudig model.
    4. Kijk naar de missers voor afzonderlijke mensen, niet alleen naar het gemiddelde.
    5. Is de score laag, laat de voorspelling dan hooguit helpen plannen voor groepen, niet beslissen over één persoon.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een functie die de MAE van de voorspellingen van een model geeft, zodat je twee modellen met elk één regel kunt vergelijken. Vier verzonnen cijfers: het eerste model voorspelt bij elk iets wat er dichtbij ligt, het tweede voorspelt altijd 3. Voer de cel uit.
    """)
    return


@app.cell
def example_mae():
    import numpy as np

    def _mae(y, pred):
        return np.mean(abs(y - pred))

    _y = np.array([2.0, 3.0, 4.0, 3.0])
    print(_mae(_y, np.array([2.5, 3.0, 3.5, 3.0])))
    print(_mae(_y, np.array([3.0, 3.0, 3.0, 3.0])))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het eerste model zit er gemiddeld 0,25 naast, het tweede 0,5. Een functie doet hetzelfde werk voor elke twee arrays. Stap 3 van het recept vergelijkt een zorgvuldig model met een eenvoudig model, met R²: oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_r2():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Hoeveel verklaart het model?

        De kinderen van het plaatje. `actual` bevat hun echte cijfers, `best` de voorspellingen van een zorgvuldig model, `simple` die van een model met vier variabelen. Vervang de `...` door de body van `r2`, zoals in het uitgewerkte voorbeeld, maar met R² uit [Hoe ver ernaast? MAE, RMSE en R²](/course/open/ml-regression-metrics?lang=nl): zet de MSE in `mse` en de variantie van `y` in `variance`, en geef R² terug met `return`. De laatste regel geeft R² voor beide modellen, zodat de check het ziet. De check probeert ook een ander eenvoudig model.
        """),
        L.task("Laat `r2(y, pred)` de R² geven van de voorspellingen `pred` voor de echte waarden `y`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_r2():
    box_r2 = L.code_box("r2", lines=12)
    box_r2
    return (box_r2,)


@app.cell(hide_code=True)
def check_r2(box_r2):
    result_r2 = L.check("r2", box_r2.value)
    result_r2
    return (result_r2,)


@app.cell(hide_code=True)
def hints_r2():
    L.hints(
        "r2",
        [
            "In de functie zijn `y` en `pred` arrays, zoals in het uitgewerkte voorbeeld. Reken er R² mee uit, in drie regels.",
            "De MSE is het gemiddelde van de gekwadrateerde missers. De variantie is de MSE van altijd het gemiddelde van `y` voorspellen. R² is 1 min het een gedeeld door het ander.",
            "`variance = np.mean((y - ___) ** 2)`",
        ],
        why="`mse = np.mean((y - pred) ** 2)`, `variance = np.mean((y - y.mean()) ** 2)` en `return 1 - mse / variance` geven `[0.2, 0.15]`: het zorgvuldige model verklaart een vijfde van de spreiding, het eenvoudige model iets minder.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Oefening 2. Welke soort taak?

    Zes taken voor AI. Zet elke taak bij de soort waar ze bij hoort, in de drie soorten van Narayanan.
    """)
    return


@app.cell(hide_code=True)
def sort_kinds_cell():
    sort_kinds = L.sort("kinds")
    sort_kinds
    return (sort_kinds,)


@app.cell(hide_code=True)
def check_kinds(sort_kinds):
    result_kinds = L.check("kinds", sort_kinds)
    result_kinds
    return (result_kinds,)


@app.cell(hide_code=True)
def hints_kinds():
    L.hints(
        "kinds",
        [
            "Vraag bij elke taak: zit het antwoord al in de data, is het een oordeel dat mensen geven, of ligt het in iemands toekomst?",
            "Twee taken gaan over een opname en een foto. Twee andere vragen wat er over jaren met een persoon gebeurt.",
            "Waarneming: spraak naar tekst en \\_\\_\\_. Oordeel: spam en \\_\\_\\_. Sociale uitkomsten: \\_\\_\\_ en \\_\\_\\_.",
        ],
        why="**Waarneming:** spraak naar tekst en een scan. **Oordelen automatiseren:** spam en een opstel. **Sociale uitkomsten voorspellen:** opnieuw? en een goede keuze?",
    )
    return


@app.cell(hide_code=True)
def text_exercise_target():
    mo.md(r"""
    ### Oefening 3. Het target en het doel
    """)
    return


@app.cell(hide_code=True)
def choice_target_cell():
    choice_target = L.choice(
        "target",
        question="Een gemeente wil voorspellen welke kinderen extra hulp op school nodig hebben, en traint een model op welke kinderen vroeger extra hulp kregen. Wat is het grootste probleem?",
        options={
            "size": "Er staan te weinig kinderen in de data om van te leren",
            "right": "Het leert wie hulp kreeg, niet wie die nodig had",
            "fine": "Niets, zolang de data van vroeger kloppen",
        },
        correct="right",
        why={
            "size": "Meer kinderen lossen het niet op: het label zelf stelt de verkeerde vraag. Zie *Het idee*.",
            "right": "Wie hulp kreeg, hangt af van wie erom vroeg, wie opviel en wat er te geven was: een target-goal mismatch, zoals bij zorgkosten voor zorgbehoefte.",
            "fine": "Kloppende data over wie hulp kreeg, zeggen nog weinig over wie die nodig had. Zie het voorbeeld van de ziekenhuizen in *Het idee*.",
        },
    )
    choice_target
    return (choice_target,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Automatisch over mensen beslissen](/course/open/rai-automated-decisions?lang=nl) vroeg je wat geautomatiseerd mag worden. Hier zag je een grens die geen wet stelt: sommige uitkomsten in het leven van mensen zijn nauwelijks te voorspellen, en het label dat je kunt meten, is vaak niet de vraag die je bedoelt. R² uit [Hoe ver ernaast? MAE, RMSE en R²](/course/open/ml-regression-metrics?lang=nl) zegt hoeveel ruimte een voorspelling laat.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_target, result_kinds, result_r2):
    L.wrap_up(result_r2, result_kinds, choice_target)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
