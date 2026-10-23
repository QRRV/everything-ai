# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-bias-in-data. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-bias-in-data", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [De workflow van machine learning](/course/open/ml-workflow?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_again():
    quick_again = L.choice(
        "quick-again",
        question="In welke stap van de workflow kijk je naar de voorbeelden die een model fout heeft?",
        options={
            "baseline": "4\\. Baseline",
            "metric": "2\\. Metric",
            "right": "6\\. Nog een ronde",
        },
        correct="right",
        why={
            "baseline": "De baseline is de eenvoudigste voorspelling om mee te vergelijken; die komt vóór het model. Zie *Het idee* in [De workflow van machine learning](/course/open/ml-workflow?lang=nl).",
            "metric": "De metric wordt afgesproken voordat er überhaupt een model is. Zie *Het idee* in [De workflow van machine learning](/course/open/ml-workflow?lang=nl).",
            "right": "Na het model: je kijkt wat het fout doet, en gaat terug naar de stap die het tegenhoudt.",
        },
    )
    quick_again
    return (quick_again,)


@app.cell(hide_code=True)
def quick_check_goal():
    quick_goal = L.choice(
        "quick-goal",
        question="Waarom spreek je de metric en het doel af voordat je uitkomsten ziet?",
        options={
            "right": "Zodat ze niet kunnen meebuigen naar het model dat je toevallig hebt",
            "fast": "Omdat trainen met een doel sneller gaat",
            "train": "Omdat het model op de metric getraind wordt, dus die moet er zijn voordat het trainen begint",
        },
        correct="right",
        why={
            "fast": "Het doel verandert niets aan hoe het trainen loopt. Wat kan er misgaan als je het achteraf kiest? Zie het let-op in [De workflow van machine learning](/course/open/ml-workflow?lang=nl).",
            "train": "Een model traint op zijn loss, niet op de metric waarmee je het beoordeelt. Wat kan er misgaan als je de metric achteraf kiest? Zie het let-op in [De workflow van machine learning](/course/open/ml-workflow?lang=nl).",
            "right": "Achteraf gekozen buigen een metric en een doel mee naar de uitkomsten, net als een testscore als je met de test set kiest.",
        },
    )
    quick_goal
    return (quick_goal,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een score is een gemiddelde over een test set, en een test set bestaat uit mensen, foto's of opnames die iemand verzamelde. In 2018 testten Joy Buolamwini en Timnit Gebru drie commerciële systemen die uit een foto van iemands gezicht het geslacht raden. De systemen hadden het bij hooguit 0,8% van de mannen met een lichte huid fout, en bij tot wel 34,7% van de vrouwen met een donkere huid. Eén totaalscore zei daar weinig over. Een model leert van zijn data, dus leert het ook wie erin ontbreekt, en de kijk van de mensen die de data verzamelden en labelden. Deze les gaat over die soort bias, **schadelijke bias**: als een systeem groepen oneerlijk behandelt. Het heeft niets te maken met de bias term $b$ van een model.

    **Een voorbeeld uit het dagelijks leven.** Een schoenenwinkel met alleen maten 38 tot en met 44 helpt de meeste klanten. Iemand met maat 35 gaat met lege handen weg, en in de cijfers van de winkel zie je die klant nooit terug.
    """)
    return


@app.cell(hide_code=True)
def predict_hidden():
    guess_hidden = L.guess(
        "hidden",
        question="Een gezichtsdetector vindt 94% van de gezichten in zijn test set. 90% van die gezichten is van groep A, 10% van groep B. Wat is volgens jou het kleinste deel van de gezichten van groep B dat hij kan vinden?",
        options={
            "close": "Niet veel minder dan 94%: de totaalscore is een gemiddelde over alle gezichten",
            "eighty": "Ongeveer 80%",
            "right": "40%",
        },
        reveal="**40%.** Vindt hij elk gezicht van groep A, dan is het gemiddelde over alle gezichten 0,9 × 1 + 0,1 × 0,4 = 0,94. Een groep van 10% weegt maar voor een tiende mee in de totaalscore, dus zelfs een slechte uitkomst voor die groep verschuift het getal nauwelijks. De detector van deze les vindt ongeveer 98% van groep A en maar 60% van groep B, en toch 0,9 × 0,98 + 0,1 × 0,6 ≈ 0,94 van alle gezichten. Je ziet het pas als je per groep meet.",
    )
    guess_hidden
    return (guess_hidden,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Drie vragen voordat je een score vertrouwt:

        - **Wie kan er schade van hebben?** **Schade** is een slecht effect van een AI-systeem op mensen: een lening die onterecht geweigerd wordt, een gezicht dat een camera niet vindt, een groep die als stereotype wordt neergezet. Een **stakeholder** is iedereen die belang heeft bij het systeem of erdoor geraakt wordt: de mensen die het gebruiken, de mensen over wie het beslist, ook als ze het nooit zien, en de mensen die het bouwen of kopen.
        - **Wie staat er in de data, en wie ontbreekt?** **Data bias** is scheefheid in een dataset: sommige groepen ontbreken, zijn te zeldzaam, of worden vertekend beschreven, en een model dat erop traint, leert dezelfde scheefheid. Die kan komen van wie er gemeten werd, van de mensen die de labels gaven, en van het verleden dat de data vastlegt.
        - **Voor wie klopt de score?** Een accuracy over twee groepen, A en B, is een gemiddelde van hun accuracy's, gewogen naar hun grootte:
        """),
        L.formula(r"\text{accuracy} = \frac{n_A}{n} \cdot \text{accuracy}_A + \frac{n_B}{n} \cdot \text{accuracy}_B"),
        mo.md(r"""
        Een kleine groep weegt weinig mee, hoe slecht het model het voor die groep ook doet. Dus meet je ook per groep, zoals een confusion matrix elke klasse op een eigen rij laat zien.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **schade** | | een slecht effect van een AI-systeem op mensen, zoals een oneerlijke beslissing, niet herkend worden, of als stereotype neergezet worden | |
        | **stakeholder** | | iedereen die belang heeft bij een AI-systeem of erdoor geraakt wordt: gebruikers, de mensen over wie het beslist, makers en kopers | |
        | **data bias** | | scheefheid in een dataset: groepen die ontbreken, te zeldzaam zijn of vertekend beschreven worden, zodat een model dezelfde scheefheid leert | |
        | **schadelijke bias** | | als een systeem groepen oneerlijk behandelt; niet de bias term $b$ | |
        | $n_A$ | n A | het aantal voorbeelden van groep A in de test set | `len(results[results["group"] == "A"])` |
        """),
        L.watch_out("Een kolom zoals geslacht of afkomst uit de features laten, haalt scheefheid niet weg. Andere kolommen, zoals een postcode of een voornaam, kunnen ervoor in de plaats komen, en het model leert de scheefheid via die kolommen."),
        mo.Html(C.picture({"share": "Aandeel in de test set", "share_y": "aandeel", "acc": "Gevonden gezichten", "acc_y": "accuracy", "all": "alle",
                           "alt": "Links het aandeel van elke groep in de test set van een gezichtsdetector: groep A 0,9, groep B 0,1. Rechts het aandeel gezichten dat hij vindt: ongeveer 0,98 voor groep A, 0,6 voor groep B, en 0,94 voor alle gezichten samen."})),
        mo.md(r"""
        De gezichtsdetector van de gok, in getallen: 180 gezichten van groep A en 20 van groep B. De staaf voor alle gezichten, rechts, is bijna even hoog als die van groep A, omdat groep A negen tiende van de test set is.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Voordat je een score vertrouwt:

    1. Zet de stakeholders op een rij: wie gebruikt het systeem, over wie beslist het, en wie wordt erdoor geraakt zonder het te gebruiken?
    2. Vraag welke schade een fout kan doen, en bij wie.
    3. Tel wie er in de data staat, per groep, en vraag wie er ontbreekt.
    4. Vraag wie er mat en wie de labels gaf, en hoe.
    5. Meet de metric per groep, niet alleen in totaal.
    6. Schrijf op wat je vond, naast de score.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De gezichtsdetector van de figuur: zijn uitkomsten op 200 testgezichten, 1 voor een gezicht dat hij vond en 0 voor een dat hij miste. `pd.DataFrame({...})` maakt een tabel van een dict: elke key wordt de naam van een kolom, en zijn lijst de waarden in die kolom, één per rij. `value_counts()` gedeeld door het aantal rijen geeft het aandeel van elke waarde. Het commentaar met `#` zegt bij welke stap van het recept elk deel hoort. Voer de cel uit.
    """)
    return


@app.cell
def example_groups():
    import pandas as pd

    # 1. stakeholders: de mensen wier gezicht hij moet vinden, en iedereen die op hem vertrouwt
    # 2. schade: een gezicht dat hij mist, is een mens die het systeem niet ziet
    _groups = ["A"] * 180 + ["B"] * 20
    _found = [1] * 176 + [0] * 4 + [1] * 12 + [0] * 8
    _results = pd.DataFrame({"group": _groups, "found": _found})
    # 3. wie er in de data staat: het aandeel van elke groep
    print(_results["group"].value_counts() / len(_results))
    # 5. de score in totaal, en per groep
    print(_results["found"].mean())
    for _group in ["A", "B"]:
        print(_group, _results[_results["group"] == _group]["found"].mean())
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Groep A is 0,9 van de test set en groep B 0,1. In totaal vindt de detector 0,94 van de gezichten: 188 van de 200. Voor groep A is dat ongeveer 0,98, voor groep B maar 0,6. Stap 4 staat niet in de code: wie maakte deze foto's, en wie besliste welke gezichten als gevonden tellen? Stap 6 is een zin naast de score: "vindt 94% van alle gezichten; van groep B, 10% van de test set, maar 60%".
    """)
    return


@app.cell(hide_code=True)
def text_exercise_accents():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Per accent

        Een spraakapp is getest op 20 opnames: 12 van mensen met accent A, 5 met accent B en 3 met accent C. `results` heeft een rij per opname, met het accent en een 1 in `correct` als de app het begreep. Vervang de `...` door één regel: voeg de accuracy van dit accent toe aan `accuracies`, met een masker zoals in het uitgewerkte voorbeeld. De laatste regel is `accuracies`, zodat de check het ziet. De check probeert ook andere uitkomsten.
        """),
        L.task("Zet in `accuracies` de accuracy van de app voor accent A, B en C, in die volgorde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_accents():
    box_accents = L.code_box("per-accent", lines=10)
    box_accents
    return (box_accents,)


@app.cell(hide_code=True)
def check_accents(box_accents):
    result_accents = L.check("per-accent", box_accents.value)
    result_accents
    return (result_accents,)


@app.cell(hide_code=True)
def hints_accents():
    L.hints(
        "per-accent",
        [
            "Stap 5 van het recept, zoals in het uitgewerkte voorbeeld: per groep het gemiddelde van `correct` over de rijen van die groep.",
            "Welk masker houdt alleen de rijen van één accent over? En hoe komt een waarde achteraan een lijst?",
            "`accuracies.append(results[___][\"correct\"].___())`.",
        ],
        why="`results[results[\"accent\"] == accent]` houdt de opnames van één accent over, en het gemiddelde van hun `correct` is het deel dat de app begreep: 11 van 12 voor A, 3 van 5 voor B en 1 van 3 voor C. In totaal begrijpt hij 15 van de 20, 0,75, en dat verbergt dat hij de meeste opnames met accent C fout heeft.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_sources():
    mo.md(r"""
    ### Oefening 2. Waar komt de scheefheid vandaan?

    Zes gevallen van data bias. Zet elk geval bij de plek waar de scheefheid vandaan komt: wie er in de data staat, wie de labels gaf, of het verleden dat de data vastlegt.
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
            "Vraag bij elk geval: ontbreekt een groep of is hij zeldzaam, gaven mensen een oordeel als label, of legt de data getrouw een oneerlijk verleden vast?",
            "Twee gevallen gaan over wie er überhaupt in de data kon komen. Welke twee gaan over een oordeel dat mensen opschreven?",
            "Wie in de data staat: de foto's van bekende mensen en \\_\\_\\_. Wie de labels gaf: grof of niet en \\_\\_\\_. Het verleden: \\_\\_\\_ en de oude leningen.",
        ],
        why="**Wie in de data staat:** de foto's van bekende mensen en de vrijwilligers. **Wie de labels gaf:** grof of niet, en de pijnscores. **Het verleden:** tien jaar aannemen en de oude leningen; daar klopt de data, maar wat hij vastlegt, was niet eerlijk.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_stakeholders():
    mo.md(r"""
    ### Oefening 3. Wie heeft er belang bij?
    """)
    return


@app.cell(hide_code=True)
def choice_stakeholders_cell():
    choice_stakeholders = L.choice(
        "stakeholders",
        question="Een school gebruikt een model dat op grond van aanwezigheid en cijfers leerlingen aanwijst die misschien uitvallen, zodat een mentor met ze kan praten. Is een leerling die het model niet aanwijst een stakeholder?",
        options={
            "unseen": "Nee: die leerling ziet het model en wat het zegt nooit",
            "right": "Ja: een leerling die het mist, krijgt geen hulp",
            "silent": "Nee: het model beslist alleen over de leerlingen die het aanwijst",
        },
        correct="right",
        why={
            "unseen": "Een stakeholder hoeft het systeem niet te zien. Wat gebeurt er met een leerling die hulp nodig heeft en niet wordt aangewezen? Zie *Het idee*.",
            "silent": "Het model bekijkt elke leerling, en iemand niet aanwijzen is ook een beslissing. Wat gebeurt er met een leerling die het mist? Zie *Het idee*.",
            "right": "Het model beslist over elke leerling, ook als het niets zegt. De school en de mentoren, de leerlingen die het aanwijst, hun ouders en de makers hebben er ook belang bij.",
        },
    )
    choice_stakeholders
    return (choice_stakeholders,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl) schreef je de grenzen van een dataset op, wie of wat er ontbreekt; in [De workflow van machine learning](/course/open/ml-workflow?lang=nl) keek je in stap 6 naar wat een model fout heeft. Hier vraag je ook bij wie het misgaat, en wie er helemaal niet in de data staat. Deze vragen komen terug zodra een model over mensen beslist. In de volgende les, *Vraag na vraag: de beslisboom*, leer je een model kennen dat voorspelt met de ene vraag na de andere, zoals een stroomschema.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_stakeholders, result_accents, result_sources):
    L.wrap_up(result_accents, result_sources, choice_stakeholders)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
