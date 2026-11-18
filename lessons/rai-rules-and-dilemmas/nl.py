# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-rules-and-dilemmas. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-rules-and-dilemmas", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_consequences():
    quick_consequences = L.choice(
        "quick-consequences",
        question="Welke vraag stelt het consequentialisme over een keuze?",
        options={
            "duty": "Of ze een plicht of een recht nakomt, hoe de uitkomst ook uitvalt",
            "right": "Welk goed en welke schade ze teweegbrengt, opgeteld over iedereen",
            "character": "Wat iemand met een goed karakter op dezelfde plek zou doen",
        },
        correct="right",
        why={
            "duty": "Dat is de vraag van de deontologie, die vasthoudt aan plichten en rechten wat de uitkomst ook is. Zie *Het idee* in [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).",
            "right": "Het consequentialisme beoordeelt een keuze op wat ze teweegbrengt: de beste keuze doet het meeste goed, opgeteld over iedereen.",
            "character": "Dat is de vraag van de deugdethiek. Welke theorie telt uitkomsten op? Zie *Het idee* in [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).",
        },
    )
    quick_consequences
    return (quick_consequences,)


@app.cell(hide_code=True)
def quick_check_care():
    quick_care = L.choice(
        "quick-care",
        question="Eén theorie begint bij relaties en bij de mensen die van anderen afhangen. Welke, en wat vraagt ze?",
        options={
            "virtue": "Deugdethiek: wat zou iemand met een goed karakter doen?",
            "contract": "Een eerlijke afspraak: welke regel kan iedereen aanvaarden?",
            "right": "Zorgethiek: wie hangt van wie af, en wat hebben ze nodig?",
        },
        correct="right",
        why={
            "virtue": "Deugdethiek vraagt naar karakter, niet naar wie van wie afhangt. Zie *Het idee* in [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).",
            "contract": "Een eerlijke afspraak vraagt welke regel je zou kiezen zonder te weten wie je zult zijn. Zie *Het idee* in [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).",
            "right": "Zorgethiek begint bij relaties en bij wie anderen nodig heeft, en luistert naar wat zij zeggen nodig te hebben.",
        },
    )
    quick_care
    return (quick_care,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Mensen willen al lang regels die machines goed laten handelen. In 1942 schreef Isaac Asimov drie wetten voor de robots in zijn verhalen, en veel van die verhalen gaan over hoe de wetten misgaan. Vandaag is de vraag niet meer alleen fictie: een zelfrijdende auto die niet op tijd kan stoppen, moet iets doen, en iemand moet vooraf beslissen wat. Regels zijn makkelijk op te schrijven. Deze les schrijft er een paar als code, om te zien waar ze standhouden en waar ze breken.

    **Een voorbeeld uit het dagelijks leven.** Een gezin heeft twee huisregels: geen telefoon aan tafel, en altijd opnemen als oma belt. Allebei prima, tot oma belt tijdens het eten.
    """)
    return


@app.cell(hide_code=True)
def predict_decisions():
    guess_decisions = L.guess(
        "decisions",
        question="Een website die de Moral Machine heet, vroeg bezoekers wat een zelfrijdende auto moet doen bij een ongeluk dat hij niet kan vermijden, zoals drie mensen raken of uitwijken naar een muur. Hoeveel beslissingen verzamelde hij volgens jou?",
        options={
            "thousands": "Ongeveer 40.000",
            "millions": "Ongeveer 4 miljoen",
            "right": "Ongeveer 40 miljoen",
        },
        reveal="**Ongeveer 40 miljoen**, in tien talen, van miljoenen mensen in 233 landen en gebieden (Awad en anderen, 2018). Sommige voorkeuren golden bijna overal; andere verschilden per regio van de wereld, zoals *Het idee* laat zien.",
    )
    guess_decisions
    return (guess_decisions,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Machine-ethiek** vraagt hoe een machine zo kan handelen dat het moreel juist is: door regels te volgen die mensen schreven, door te leren van voorbeelden, of door een mengvorm. De **drie wetten van de robotica** van Asimov zijn de bekendste poging met regels. Kort gezegd:

        1. Een robot mag een mens geen kwaad doen, en een mens ook niet door niets te doen in gevaar laten komen.
        2. Een robot moet mensen gehoorzamen, behalve als dat de eerste wet breekt.
        3. Een robot moet zichzelf beschermen, behalve als dat de eerste of de tweede wet breekt.

        Als code is elke wet een filter op de keuzes die een robot heeft, in volgorde toegepast. Dat werkt zolang er een keuze door komt. Maar de wetten zeggen niet wat *schade* is, of wiens bevel telt als twee mensen het oneens zijn, en als elke keuze iemand schaadt, verbiedt de eerste wet ze allemaal.

        Dat laatste geval is het *trolleyprobleem*, een gedachte-experiment van de filosoof Philippa Foot uit 1967: een op hol geslagen tram rijdt vijf mensen dood, tenzij de bestuurder hem een andere baan op stuurt, waar hij er één doodrijdt. De **Moral Machine** legde zulke keuzes voor aan miljoenen mensen, voor zelfrijdende auto's. Over alle antwoorden samen waren drie voorkeuren het sterkst: mensen sparen boven dieren, meer levens boven minder, en jong boven oud. Maar in een groep landen die de onderzoekers het oostelijke cluster noemden, was de voorkeur om jongeren te sparen veel zwakker. Eén vaste regel gaat in tegen wat veel mensen zouden kiezen, en een meerderheid is nog geen ethisch argument.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **machine-ethiek** | het deel van AI-ethiek dat morele afwegingen in machines zelf probeert in te bouwen, vanuit regels, vanuit voorbeelden, of allebei |
        | **drie wetten van de robotica** | de verzonnen regels voor robots van Isaac Asimov: doe mensen geen kwaad, gehoorzaam mensen, bescherm jezelf, in die volgorde |
        | **Moral Machine** | een online experiment waarin miljoenen mensen kozen wat een zelfrijdende auto moet doen bij een onvermijdelijk ongeluk; de voorkeuren verschilden per regio |
        """),
        L.watch_out("Het trolleyprobleem is een gedachte-experiment, geen beeld van hoe auto's verongelukken: een echte auto remt en stuurt zonder zeker te weten wat er gebeurt. Het laat zien waar regels botsen, het is geen ontwerp voor een auto."),
        mo.Html(C.picture({"title": "Keuzes die na elke wet nog mogen", "count": "toegestane keuzes",
                           "stages": ["begin", "wet 1", "wet 2", "wet 3"], "kitchen": "robot die de post brengt", "car": "auto die niet kan stoppen",
                           "alt": "Voor een robot die de post brengt: 4 keuzes aan het begin, 3 na de eerste wet, 2 na de tweede en 1 na de derde. Voor een auto die niet kan stoppen: 2 keuzes aan het begin en 0 na de eerste wet."})),
        mo.md(r"""
        Twee situaties, verzonnen door de course. Gevulde staven: een robot die de post moet brengen heeft 4 keuzes, en elke wet haalt er een paar weg tot er 1 over is. Open staven: een auto die niet kan stoppen heeft er 2, en allebei schaden ze iemand, dus na de eerste wet blijft er niets over.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een regel voor een machine te testen:

    1. Schrijf de regel als code: welke keuzes houdt hij over?
    2. Probeer hem op een makkelijk geval, en dan op een geval waarin elke keuze iets kost.
    3. Zoek woorden die de regel niet definieert, zoals schade.
    4. Vraag wat de regel doet als twee mensen het tegenovergestelde willen.
    5. Waar de regel niets zegt, schrijf je op wie er dan beslist, en hoe.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De robot van de afbeelding moet de post brengen. Drie dictionaries zeggen voor elke keuze hoeveel mensen ze schaadt, of ze doet wat bevolen is, en of de robot heel blijft. Elke wet houdt de keuzes over die erdoor komen, uit wat de wet ervoor overhield. Voer de cel uit.
    """)
    return


@app.cell
def example_kitchen():
    _harmed = {"walk around the child": 0, "push past the child": 1, "stay in the corner": 0, "jump down the stairs": 0}
    _ordered = {"walk around the child": True, "push past the child": True, "stay in the corner": False, "jump down the stairs": True}
    _whole = {"walk around the child": True, "push past the child": True, "stay in the corner": True, "jump down the stairs": False}

    # 1. wet 1: schaad niemand
    _law1 = []
    for _action in _harmed:
        if _harmed[_action] == 0:
            _law1.append(_action)
    print("law 1:", _law1)
    # wet 2: gehoorzaam, binnen wat wet 1 toestaat
    _law2 = []
    for _action in _law1:
        if _ordered[_action]:
            _law2.append(_action)
    print("law 2:", _law2)
    # wet 3: blijf heel, binnen wat wet 2 toestaat
    _law3 = []
    for _action in _law2:
        if _whole[_action]:
            _law3.append(_action)
    print("law 3:", _law3)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste wet schrapt langs het kind dringen (push past the child), de tweede in de hoek blijven (stay in the corner), de derde van de trap springen (jump down the stairs): de robot loopt om het kind heen. Stap 2 van het recept vraagt om een moeilijker geval: oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_allowed():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De auto die niet kan stoppen

        Een zelfrijdende auto kan niet op tijd stoppen. Als hij doorrijdt, raakt hij drie mensen; als hij uitwijkt naar de muur, raakt zijn ene passagier gewond. `harmed` zegt hoeveel mensen elke keuze schaadt, en de loop gaat met `action` langs de keuzes. Vervang de `...` door de eerste wet, zoals in het uitgewerkte voorbeeld. De laatste regel is `allowed`, zodat de check het ziet. De check probeert ook een andere situatie.
        """),
        L.task("Zet in `allowed` de namen van de keuzes die niemand schaden."),
    ])
    return


@app.cell(hide_code=True)
def exercise_allowed():
    box_allowed = L.code_box("allowed", lines=7)
    box_allowed
    return (box_allowed,)


@app.cell(hide_code=True)
def check_allowed(box_allowed):
    result_allowed = L.check("allowed", box_allowed.value)
    result_allowed
    return (result_allowed,)


@app.cell(hide_code=True)
def hints_allowed():
    L.hints(
        "allowed",
        [
            "Stap 1 van het recept, zoals de eerste wet in het uitgewerkte voorbeeld: twee regels binnen de loop.",
            "Welke test zegt dat een keuze niemand schaadt? En wat doet de loop met een keuze die erdoor komt?",
            "`if harmed[action] == ___:` en daaronder `allowed.___(action)`.",
        ],
        why="`if harmed[action] == 0:` met `allowed.append(action)` eronder houdt de keuzes over die niemand schaden. Hier is dat geen enkele: `[]`. De eerste wet verbiedt elke keuze, en zegt niet welke schade kleiner is.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_gaps():
    mo.md(r"""
    ### Oefening 2. Beslecht of niet?

    Zes gevallen voor een robot met de drie wetten. Zet elk geval bij wat de wetten ermee doen.
    """)
    return


@app.cell(hide_code=True)
def sort_gaps_cell():
    sort_gaps = L.sort("gaps")
    sort_gaps
    return (sort_gaps,)


@app.cell(hide_code=True)
def check_gaps(sort_gaps):
    result_gaps = L.check("gaps", sort_gaps)
    result_gaps
    return (result_gaps,)


@app.cell(hide_code=True)
def hints_gaps():
    L.hints(
        "gaps",
        [
            "Vraag bij elk geval: gaat de ene wet duidelijk voor de andere, of laten de wetten het antwoord open?",
            "In drie gevallen trekken twee wetten een andere kant op en beslist hun volgorde. In welke drie helpt geen enkele volgorde?",
            "Beslecht: het bevel om te slaan, \\_\\_\\_ en \\_\\_\\_. Geen antwoord: elke keuze schaadt, \\_\\_\\_ en \\_\\_\\_.",
        ],
        why="**Beslecht:** het bevel om te slaan (de eerste wet gaat voor de tweede), het vuur in (de tweede gaat voor de derde) en de vallende muur (alleen de derde geldt). **Geen antwoord:** elke keuze schaadt (de eerste wet verbiedt alles), een harde waarheid (schade is niet gedefinieerd) en twee bevelen (de tweede wet zegt niet wiens).",
    )
    return


@app.cell(hide_code=True)
def text_exercise_moral_machine():
    mo.md(r"""
    ### Oefening 3. Eén regel voor iedereen?
    """)
    return


@app.cell(hide_code=True)
def choice_moral_machine_cell():
    choice_moral_machine = L.choice(
        "moral-machine",
        question="De Moral Machine vond dat sommige voorkeuren per regio van de wereld verschillen. Een autofabrikant wil één regel voor al zijn auto's. Wat volgt er uit de resultaten?",
        options={
            "majority": "Neem de keuze van de meerderheid van de wereld; dan is de regel goed voor iedereen",
            "right": "Elke vaste regel gaat in tegen wat veel mensen zouden kiezen",
            "small": "Niets: de verschillen waren te klein om ertoe te doen",
        },
        correct="right",
        why={
            "majority": "Een meerderheid laat zien wat de meeste mensen kozen, niet wat goed is, en de mensen in de minderheid moeten ook met de regel leven. Wat betekenen de verschillen per regio voor één regel? Zie *Het idee*.",
            "right": "Waar voorkeuren verschillen, gaat één regel altijd tegen sommige ervan in. Wie beslist, en hoe, is de ethische vraag, niet iets wat de data beslecht.",
            "small": "Sommige verschillen waren groot, zoals hoe graag mensen in verschillende regio's jongeren wilden sparen. Zie *Het idee*.",
        },
    )
    choice_moral_machine
    return (choice_moral_machine,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl) beoordeelde je één keuze door verschillende theorieën. De wetten van Asimov zijn plichten in een vaste volgorde, deontologie als code, en een dilemma waarin elke keuze iemand schaadt, is waar zulke plichten opraken. In de volgende les, *Principes en richtlijnen*, bekijk je de principes die overheden en bedrijven voor AI schreven in plaats van regels: korte woorden waar bijna iedereen het mee eens is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_moral_machine, result_allowed, result_gaps):
    L.wrap_up(result_allowed, result_gaps, choice_moral_machine)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
