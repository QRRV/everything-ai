# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-ethical-theories. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-ethical-theories", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_measure():
    quick_measure = L.choice(
        "quick-measure",
        question="Een model bepaalt welke sollicitanten worden uitgenodigd voor een gesprek. Hoe meet je allocatieve schade?",
        options={
            "outputs": "Het model veel vacatureteksten laten schrijven en de stereotypen erin tellen",
            "right": "Per groep het aandeel van de sollicitanten dat een uitnodiging krijgt",
            "overall": "Het aandeel van alle sollicitanten dat wordt uitgenodigd",
        },
        correct="right",
        why={
            "outputs": "Zo zoek je naar representatieve schade. Een uitnodiging wordt uitgedeeld: wat kun je per groep tellen? Zie *Het idee* in [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).",
            "right": "Een uitnodiging wordt uitgedeeld, dus je telt per groep wie hem krijgt: het aandeel ja van elke groep.",
            "overall": "Eén getal over iedereen verbergt hoe de groepen verschillen. Wat vergelijk je? Zie de formule in [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).",
        },
    )
    quick_measure
    return (quick_measure,)


@app.cell(hide_code=True)
def quick_check_look():
    quick_look = L.choice(
        "quick-look",
        question="Waarom moet je zoeken naar stereotypen in wat een systeem laat zien, in plaats van ze te meten aan zijn beslissingen?",
        options={
            "rare": "Omdat ze veel zeldzamer zijn dan allocatieve schade, dus een test set bevat ze zelden",
            "pictures": "Omdat ze alleen in afbeeldingen voorkomen",
            "right": "Omdat een stereotype niets beslist, dus er is geen ja om te tellen",
        },
        correct="right",
        why={
            "rare": "Het gaat niet om hoe vaak het gebeurt. Wordt er iemand afgewezen als een systeem een stereotype laat zien? Zie het blok *Let op* in [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).",
            "pictures": "Een vertaling en een zoekvak kunnen ook een stereotype laten zien, in woorden. Wordt er dan iemand afgewezen? Zie *Het idee* in [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).",
            "right": "Een stereotype wijst niemand af, dus je verzamelt veel uitvoer en telt wie hoe wordt getoond.",
        },
    )
    quick_look
    return (quick_look,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Twee mensen kunnen het over elk getal eens zijn en het toch oneens zijn over wat een systeem moet doen. Dat komt niet doordat een van de twee niet kan rekenen: ze oordelen op verschillende dingen. De ethiek denkt al heel lang over die dingen na, en haar belangrijkste theorieën stellen elk hun eigen vraag. Als je die vragen kent, zie je beter wat er bij een keuze echt op het spel staat, en kun je het zeggen in woorden die anderen kunnen volgen.

    **Een voorbeeld uit het dagelijks leven.** Er is nog één stuk taart. Geef het aan wie de meeste honger heeft, aan wie het beloofd was, of snijd het zo dat iedereen een stukje krijgt: elk antwoord volgt een ander idee van wat goed is.
    """)
    return


@app.cell(hide_code=True)
def predict_far():
    guess_far = L.guess(
        "far",
        question="Een ziekenhuis heeft een programma dat voorkomt dat patiënten na hun ontslag terugkomen. Een model kiest de patiënten: eerst wie dichtbij woont (nearby first), omdat die naar elke afspraak komt, of eerst wie het het hardst nodig heeft (need first). Nearby first voorkomt 32 heropnames, need first 28. Hoe groot is volgens jou het aandeel van de patiënten ver weg dat nearby first helpt, vergeleken met de patiënten dichtbij het ziekenhuis?",
        options={
            "same": "Ongeveer even groot als dichtbij het ziekenhuis",
            "less": "Iets kleiner dan dichtbij het ziekenhuis",
            "right": "Een vijfde zo groot aandeel",
        },
        reveal="**Een vijfde zo groot aandeel.** Nearby first helpt 2 van de 100 patiënten ver weg, 0,02, tegen 30 van de 300 dichtbij, 0,1. De aanpak die in totaal het meeste goed doet, doet het minst voor de groep die het slechtst af is. De getallen van dit ziekenhuis heeft de course verzonnen.",
    )
    guess_far
    return (guess_far,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **AI-ethiek** vraagt wat goed en fout is in hoe AI wordt gemaakt en gebruikt. Ze leent haar gereedschap van de ethiek, die verschillende theorieën heeft. Elke theorie stelt een andere vraag over de keuze van het ziekenhuis.

        - **Consequentialisme**: wat brengt elke keuze teweeg, opgeteld over iedereen? De beste keuze doet het meeste goed. Nearby first voorkomt 32 heropnames, need first 28.
        - **Deontologie**: welke plichten en rechten gelden, wat de uitkomst ook is? Immanuel Kant vond dat je een mens nooit alleen als middel mag gebruiken. Een deontoloog kan zeggen dat patiënten weglaten omdat ze het totaal omlaag halen, hen behandelt als getallen, niet als mensen.
        - *Een eerlijke afspraak* (contractualisme): welke regel zou iedereen kunnen aanvaarden? John Rawls liet je kiezen achter een *sluier van onwetendheid*, zonder te weten wie je zult zijn, en betoogde dat je dan zorgt voor wie het slechtst af is. Need first helpt 0,06 van de groep die het slechtst af is, nearby first 0,02.
        - **Deugdethiek**: wat zou iemand met een goed karakter doen? Ze zoekt eerder praktische wijsheid dan één regel: een rechtvaardige en zorgzame dokter.
        - **Zorgethiek**: wie hangt van wie af? Ze begint bij relaties en bij de mensen die anderen nodig hebben, en luistert naar wat zij zeggen nodig te hebben.

        Ethiek is niet alleen in het Westen gegroeid. In *Ubuntu*, uit zuidelijk Afrika, is een mens een mens door andere mensen: het vraagt wat een keuze doet met een gemeenschap en de banden daarin. Confucianistische ethiek kijkt naar rollen en relaties, en naar harmonie daartussen. De theorieën zijn het vaak eens; hier pleiten de meeste ertegen om de patiënten ver weg over te slaan. Waar ze het niet eens zijn, zoals bij het meeste goed in totaal tegenover het meeste voor wie het slechtst af is, zie je wat er op het spel staat.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **AI-ethiek** | het vakgebied dat onderzoekt wat goed en fout is in hoe AI wordt gemaakt en gebruikt, en wat je eraan doet |
        | **consequentialisme** | een keuze beoordelen op wat ze teweegbrengt: de beste keuze doet het meeste goed, opgeteld over iedereen |
        | **deontologie** | een keuze beoordelen op plichten en rechten die gelden wat de uitkomst ook is, zoals een mens nooit alleen als middel gebruiken |
        | **deugdethiek** | vragen wat iemand met een goed karakter, zoals een rechtvaardig en eerlijk iemand, zou doen |
        | **zorgethiek** | beginnen bij relaties en bij de mensen die van anderen afhangen, en luisteren naar wat zij nodig hebben |
        """),
        L.watch_out("Een theorie omzetten in een getal, zoals de oefening doet met de sluier van onwetendheid, maakt er één lezing van. Rawls schreef niet over ziekenhuisprogramma's, en iemand anders kan hem anders lezen: zeg welke lezing je gebruikt."),
        mo.Html(C.picture({"totals": "Voorkomen heropnames", "totals_y": "in totaal", "worst": "Slechtst af", "worst_y": "aandeel geholpen",
                           "nearby": "nearby", "need": "need",
                           "alt": "Links de heropnames die elke aanpak in totaal voorkomt: 32 voor nearby first, 28 voor need first. Rechts het aandeel dat geholpen wordt in de groep die het slechtst af is: 0,02 voor nearby first, 0,06 voor need first."})),
        mo.md(r"""
        Links wat het consequentialisme optelt: nearby first wint, 32 tegen 28. Rechts waar iemand achter de sluier naar kijkt: need first wint, 0,06 tegen 0,02. Dezelfde twee aanpakken, beoordeeld met twee vragen, komen andersom uit.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een beslissing door de theorieën te bekijken:

    1. Schrijf de beslissing en de keuzes op.
    2. Gevolgen: welk goed en welke schade doet elke keuze, en bij wie, bij elkaar opgeteld?
    3. Plichten en rechten: breekt een keuze een regel of een recht, wat ze ook teweegbrengt?
    4. Een eerlijke afspraak: welke regel zou je kiezen zonder te weten wie je zult zijn?
    5. Karakter en zorg: wat zou een goed, zorgzaam mens doen, en wat zeggen de betrokkenen dat ze nodig hebben?
    6. Waar de antwoorden verschillen, schrijf je op wat er op het spel staat.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het ziekenhuis van de gok. `people` zegt hoeveel patiënten er in elke groep wonen, `prevented` hoeveel heropnames elke aanpak in elke groep voorkomt. `prevented.items()` geeft elke aanpak met zijn eigen dictionary van groepen, zoals in [Dictionaries](/course/open/py-dicts?lang=nl). Voer de cel uit.
    """)
    return


@app.cell
def example_hospital():
    _people = {"near": 300, "far": 100}
    _prevented = {"nearby first": {"near": 30, "far": 2}, "need first": {"near": 18, "far": 10}}

    # 2. gevolgen: de heropnames die elke aanpak voorkomt, opgeteld over iedereen
    _totals = {}
    for _policy, _groups in _prevented.items():
        _totals[_policy] = _groups["near"] + _groups["far"]
    print(_totals)
    # 4. een eerlijke afspraak: het aandeel van elke groep dat elke aanpak helpt
    for _policy, _groups in _prevented.items():
        print(_policy, _groups["near"] / _people["near"], _groups["far"] / _people["far"])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Nearby first voorkomt 32 heropnames, need first 28: stap 2 kiest nearby first. Maar nearby first helpt 0,1 van de patiënten dichtbij het ziekenhuis en maar 0,02 van wie ver weg woont; need first helpt 0,06 en 0,1. Achter de sluier weet je niet in welke groep je terechtkomt, dus kijk je bij elke aanpak naar de groep die het slechtst af is. Oefening 1 pikt die groep eruit.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_worst():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Achter de sluier

        Vervang de `...` door één regel: voor deze aanpak het aandeel dat geholpen wordt in de groep die het slechtst af is, dus het kleinste van de twee aandelen. `min(a, b)` geeft het kleinste van twee getallen. De laatste regel is `worst`, zodat de check het ziet. De check probeert ook andere getallen, waarin de groep die het slechtst af is verandert.
        """),
        L.task("Zet in `worst` het aandeel dat geholpen wordt in de groep die het slechtst af is, onder de naam van elke aanpak."),
    ])
    return


@app.cell(hide_code=True)
def exercise_worst():
    box_worst = L.code_box("worst", lines=7)
    box_worst
    return (box_worst,)


@app.cell(hide_code=True)
def check_worst(box_worst):
    result_worst = L.check("worst", box_worst.value)
    result_worst
    return (result_worst,)


@app.cell(hide_code=True)
def hints_worst():
    L.hints(
        "worst",
        [
            "Stap 4 van het recept: achter de sluier kijk je bij elke aanpak naar de groep die het slechtst af is.",
            "Het uitgewerkte voorbeeld print bij elke aanpak twee aandelen. Welk van de twee hoort bij de groep die het slechtst af is?",
            "`worst[policy] = min(groups[\"near\"] / ___, groups[\"far\"] / ___)`.",
        ],
        why="`groups[\"near\"] / people[\"near\"]` is het aandeel dat geholpen wordt dichtbij het ziekenhuis, en `groups[\"far\"] / people[\"far\"]` ver weg; `min` houdt het kleinste over. Voor nearby first is dat 0,02, ver weg; voor need first 0,06, dichtbij het ziekenhuis.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_theories():
    mo.md(r"""
    ### Oefening 2. Welke theorie?

    Acht argumenten over de keuze van het ziekenhuis. Zet elk argument bij de theorie waar het uit komt.
    """)
    return


@app.cell(hide_code=True)
def sort_theories_cell():
    sort_theories = L.sort("theories")
    sort_theories
    return (sort_theories,)


@app.cell(hide_code=True)
def check_theories(sort_theories):
    result_theories = L.check("theories", sort_theories)
    result_theories
    return (result_theories,)


@app.cell(hide_code=True)
def hints_theories():
    L.hints(
        "theories",
        [
            "Vraag bij elk argument: telt het uitkomsten op, houdt het vast aan een regel of recht, vraagt het naar karakter, of begint het bij relaties en de mensen om wie het gaat?",
            "*Een gelijk recht* geldt wat het totaal ook is, en *Een goede dokter* gaat over karakter. Welk ander argument geldt wat de uitkomst ook is, en welk ander vraagt naar karakter?",
            "Gevolgen: het totaal en \\_\\_\\_. Plichten: een gelijk recht en \\_\\_\\_. Deugden: een goede dokter en \\_\\_\\_. Zorg: vraag het hun en \\_\\_\\_.",
        ],
        why="**Gevolgen:** het totaal en op de lange duur. **Plichten en rechten:** een gelijk recht en geen middel. **Deugden:** een goede dokter en wat voor ziekenhuis. **Zorg:** vraag het hun en wie van ons afhangt.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_ubuntu():
    mo.md(r"""
    ### Oefening 3. Wat Ubuntu toevoegt
    """)
    return


@app.cell(hide_code=True)
def choice_ubuntu_cell():
    choice_ubuntu = L.choice(
        "ubuntu",
        question="In Ubuntu is een mens een mens door andere mensen. Welke vraag voegt het toe aan de keuze van het ziekenhuis?",
        options={
            "right": "Wat de keuze doet met de gemeenschap en de banden daarin",
            "rule": "Welke regel een patiënt kan aanvaarden zonder te weten waar die zal wonen",
            "total": "Welke aanpak de meeste heropnames voorkomt",
        },
        correct="right",
        why={
            "right": "Ubuntu kijkt naar de banden tussen mensen: een ziekenhuis dat de dorpen ver weg weglaat, verzwakt ook zijn band met die gemeenschappen.",
            "rule": "Dat is de vraag van een eerlijke afspraak, de sluier van onwetendheid van Rawls. Waar kijkt Ubuntu naar, als een mens een mens is door anderen? Zie *Het idee*.",
            "total": "Dat is wat het consequentialisme optelt. Waar kijkt Ubuntu naar, als een mens een mens is door anderen? Zie *Het idee*.",
        },
    )
    choice_ubuntu
    return (choice_ubuntu,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl) leerde je schade vinden; hier leerde je manieren om haar te beoordelen. Het aandeel ja per groep komt terug als het aandeel dat geholpen wordt per groep, en de sluier van onwetendheid gaf een reden om naar de groep te kijken die het slechtst af is. In de volgende les, *Wat moet AI beter maken?*, vraag je wat een systeem eigenlijk beter moet maken voor mensen, en zie je dat klikken en welzijn niet hetzelfde zijn.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_ubuntu, result_theories, result_worst):
    L.wrap_up(result_worst, result_theories, choice_ubuntu)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
