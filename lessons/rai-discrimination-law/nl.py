# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-discrimination-law. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-discrimination-law", "nl")
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
def quick_check_kind():
    quick_kind = L.choice(
        "quick-kind",
        question="Een model geeft vrouwen minder leningen dan mannen met hetzelfde inkomen. Welke soort schade is dat?",
        options={
            "shown": "Representatieve schade, want het laat vrouwen armer zien",
            "right": "Allocatieve schade: vrouwen krijgen minder leningen",
            "none": "Geen schade, als het gemiddeld klopt",
        },
        correct="right",
        why={
            "shown": "Representatieve schade gaat over hoe een systeem groepen laat zien of beschrijft. Een lening is iets wat het uitdeelt. Zie *Het idee* in [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).",
            "right": "Een lening is iets wat het systeem uitdeelt, en een groep krijgt er minder van dan terecht is.",
            "none": "Een model kan gemiddeld goed zitten en toch een groep minder geven. Zie *Het idee* in [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).",
        },
    )
    quick_kind
    return (quick_kind,)


@app.cell(hide_code=True)
def quick_check_measure():
    quick_measure = L.choice(
        "quick-measure",
        question="Hoe meet je allocatieve schade?",
        options={
            "accuracy": "Met de accuracy van het model over alle mensen samen",
            "right": "Als het aandeel van elke groep dat een ja krijgt",
            "read": "Door de uitvoer een voor een te lezen en naar stereotypen te zoeken",
        },
        correct="right",
        why={
            "accuracy": "Eén getal voor iedereen verbergt hoe elke groep ervan afkomt. Zie *Het idee* in [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).",
            "right": "Je telt per groep hoeveel mensen een ja krijgen, en deelt door het aantal mensen in die groep.",
            "read": "Zo zoek je naar representatieve schade. Zie *Het idee* in [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl).",
        },
    )
    quick_measure
    return (quick_measure,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model leest bijna nooit een kolom die *ras* of *godsdienst* heet. Toch kan het groepen heel verschillend behandelen, via een postcode, een gat in een cv of een taaltoets. Is dat discriminatie? Volgens de Nederlandse wet kan dat, ook als niemand het zo bedoelde, en het maakt niet uit of een mens of een computer de keuze maakte.

    **Een voorbeeld uit het dagelijks leven.** Een club laat alleen mensen toe die langer zijn dan 1,80 meter. De regel noemt geen geslacht, maar veel meer mannen dan vrouwen zijn zo lang: zonder het te zeggen houdt de club vooral vrouwen buiten.
    """)
    return


@app.cell(hide_code=True)
def predict_vu():
    guess_vu = L.guess(
        "vu",
        question="In 2021 klaagde een student van de VU in Amsterdam dat tentamensoftware het gezicht van de student vaak niet vond, en legde een verband met huidskleur. Wat denk je dat het College voor de Rechten van de Mens in 2023 uiteindelijk besliste?",
        options={
            "software": "De tentamensoftware discrimineerde de student",
            "nothing": "De VU deed niets verkeerd",
            "right": "Niet de software, wel de klachtbehandeling",
        },
        reveal="**Niet de software, wel de klachtbehandeling.** In 2022 vond het College genoeg feiten om indirecte discriminatie te vermoeden: de problemen van de student zelf, samen met onderzoek dat laat zien dat gezichtsdetectie over het algemeen slechter werkt bij mensen met een donkere huidskleur. Toen moest de VU bewijzen dat ze niet had gediscrimineerd, en dat lukte: de student had niet meer problemen dan andere studenten, en die waren te verklaren door een slechte internetverbinding of een bril. Maar de klacht ging langs drie instanties, en pas ruim zeven maanden later nam er een hem in behandeling. Die behandeling was verboden onderscheid op grond van ras.",
    )
    guess_vu
    return (guess_vu,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Directe discriminatie.** Iemand wordt slechter behandeld dan een ander in een vergelijkbare situatie wordt, is of zou worden behandeld, vanwege een discriminatiegrond. Een voorbeeld van het College voor de Rechten van de Mens: een algoritme voor werving met variabelen die vrouwen een lagere aanbeveling geven.

        **Indirecte discriminatie.** Een regel, maatstaf of werkwijze die neutraal lijkt, treft mensen uit een beschermde groep in het bijzonder. De regel noemt de grond niet: je ziet het aan het effect. Nog een voorbeeld van het College: een vacature voor een programmeur die om perfect Nederlands vraagt. Dat houdt meer mensen met een niet-Nederlandse afkomst buiten, terwijl een programmeur het niet nodig heeft; voor een leraar kan dat anders liggen.

        **De wet.** In Nederland verbiedt de Algemene wet gelijke behandeling (AWGB) uit 1994 allebei, onder meer bij werk, bij het aanbieden van goederen en diensten en in het onderwijs. De wet noemt acht gronden: godsdienst, levensovertuiging, politieke gezindheid, ras, geslacht, nationaliteit, hetero- of homoseksuele gerichtheid en burgerlijke staat. Ras wordt ruim uitgelegd: het omvat ook huidskleur, afkomst en nationale of etnische afstamming. Voor handicap en chronische ziekte is er een eigen wet, net als voor leeftijd bij de arbeid; artikel 1 van de Grondwet verbiedt discriminatie op welke grond dan ook. De Europese richtlijnen uit 2000 beschrijven beide soorten in dezelfde woorden.

        **Wanneer indirect mag.** Een neutrale regel die een groep harder raakt, mag alleen als hij objectief gerechtvaardigd is. Daar horen drie vragen bij, en elk antwoord moet ja zijn. Dient de regel een *legitiem doel*? Is hij een *passend* middel om dat doel te bereiken? Is hij *noodzakelijk*: is er geen ander middel dat minder onderscheid maakt, en staat hij in verhouding tot het doel? Directe discriminatie heeft zo'n algemene uitweg niet, alleen uitzonderingen die de wet een voor een noemt, bijvoorbeeld werk waarvoor iemands geslacht bepalend is.

        **Het effect telt.** Het College schrijft dat het in het gelijkebehandelingsrecht niet uitmaakt of een computer of een mens het onderscheid maakt: het effect telt. Een werkgever kan niet wijzen naar de maker van de software. En als iemand feiten aanvoert die discriminatie aannemelijk maken, moet de andere kant bewijzen dat die niet discrimineerde.

        **Het College voor de Rechten van de Mens.** Wie denkt dat er gediscrimineerd is, kan het College gratis om een oordeel vragen. Zijn oordelen zijn juridisch niet bindend, je kunt er niet tegen in beroep, en het kan geen boete of schadevergoeding opleggen: daarvoor ga je naar de rechter. Maar in ruim 80 procent van de gevallen waarin het discriminatie vaststelt, worden maatregelen genomen, volgens zijn eigen cijfers. Een organisatie kan ook naar haar eigen handelen vragen. In 2023 oordeelde het College over een vraag van de datingapp Breeze over zijn eigen zelflerende algoritme, dat volgens het vermoeden van Breeze gebruikers met een donkere huidskleur of een niet-Nederlandse afkomst gemiddeld een kleinere kans op een match gaf. Het algoritme gebruikte geen gegevens over afkomst, maar versterkte hoe gebruikers elkaar leuk vonden. Het College noemde dat indirecte discriminatie, en zei dat Breeze maatregelen mag en zelfs moet nemen.

        **Een oordeel over een automatische regel.** In november 2024 oordeelde het College over de Dienst Toeslagen. De Basisregistratie Personen had code 98 gestuurd, *geen rechtmatig verblijf*, voor het pleegkind van een vrouw, en de Dienst stopte automatisch de huurtoeslag van de vrouw en vorderde die terug. De code was fout: het kind had een geldige verblijfsvergunning. Code 98 verwijst naar verblijf, niet naar ras, dus dit was geen directe discriminatie. Maar gezinnen zonder rechtmatig verblijf hebben bijna altijd een of meer gezinsleden met een niet-Nederlandse herkomst: indirect. Het doel was legitiem en het middel passend, maar niet noodzakelijk: een toeslag stopzetten heeft acute en verstrekkende gevolgen, dus de Dienst had de code eerst moeten controleren. Daarmee was het verboden onderscheid op grond van ras.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **directe discriminatie** | iemand slechter behandelen dan anderen in een vergelijkbare situatie vanwege een discriminatiegrond; de wet noemt het *direct onderscheid* |
        | **indirecte discriminatie** | een regel of werkwijze die neutraal lijkt maar mensen uit een beschermde groep in het bijzonder treft; de wet noemt het *indirect onderscheid* |
        | **Algemene wet gelijke behandeling** | de Nederlandse wet (AWGB) die beide soorten verbiedt op acht gronden, onder meer bij werk, diensten en onderwijs |
        | **College voor de Rechten van de Mens** | het Nederlandse instituut dat gratis oordeelt over klachten over ongelijke behandeling; zijn oordelen zijn niet bindend, maar hebben wel gezag |
        """),
        L.watch_out("Indirect is niet altijd verboden: een neutrale regel die een groep harder raakt, mag als alle drie de vragen ja opleveren. En een regel is niet in orde omdat hij geen grond noemt: indirecte discriminatie vraagt geen opzet, en geen kolom die ras heet."),
        mo.Html(C.picture({"title": "Hoe de wet naar een regel kijkt",
                           "questions": ["Behandelt de regel iemand slechter vanwege een discriminatiegrond?",
                                         "Treft een neutrale regel mensen uit een beschermde groep in het bijzonder?",
                                         "Indirecte discriminatie: is die objectief gerechtvaardigd?"],
                           "ways": [["Ja: directe discriminatie. Verboden, behalve waar de wet een uitzondering noemt.",
                                     "Nee: ga door naar vraag 2."],
                                    ["Ja: indirecte discriminatie. Ga door naar vraag 3.",
                                     "Nee: geen onderscheid."],
                                    ["Drie keer ja: toegestaan.", "Eén nee: verboden."]],
                           "steps": ["legitiem doel?", "passend?", "noodzakelijk?"],
                           "alt": "Drie stappen. 1: behandelt de regel iemand slechter vanwege een discriminatiegrond? Ja betekent directe discriminatie, verboden behalve waar de wet een uitzondering noemt; nee betekent doorgaan. 2: treft een neutrale regel mensen uit een beschermde groep in het bijzonder? Ja betekent indirecte discriminatie; nee betekent geen onderscheid. 3: is de indirecte discriminatie objectief gerechtvaardigd, met een legitiem doel, een passend middel en een noodzakelijk middel? Drie keer ja betekent toegestaan, één nee betekent verboden."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo toets je een regel, of een model, aan het gelijkebehandelingsrecht:

    1. Kijk of de regel zelf mensen anders behandelt naar een discriminatiegrond. Zo ja, dan is het directe discriminatie.
    2. Zo niet, tel per groep wie de regel raakt. Als een groep in het bijzonder wordt getroffen, is het indirecte discriminatie.
    3. Schrijf het doel van de regel op, en vraag of dat doel legitiem is.
    4. Vraag of de regel dat doel echt bereikt: is hij passend?
    5. Vraag of een ander middel minder onderscheid maakt, en of de schade in verhouding staat tot het doel: is hij noodzakelijk? Eén nee, en de regel is verboden.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De eerste twee vragen van het plaatje, vereenvoudigd, voor een regel die de course verzon. `_rule` zegt met `"yes"` of `"no"` of de regel een discriminatiegrond noemt en of hij een groep harder raakt. `_kind` begint als `"no distinction"` (geen onderscheid); elke `if` kan dat veranderen, en de laatste die waar is wint, dus een regel die een grond noemt is altijd direct. Voer de cel uit.
    """)
    return


@app.cell
def example_kind():
    # een verzonnen regel, beschreven door de course
    _rule = {"names_ground": "no", "hits_harder": "yes"}

    _kind = "no distinction"
    if _rule["hits_harder"] == "yes":
        _kind = "indirect"
    if _rule["names_ground"] == "yes":
        _kind = "direct"
    print(_kind)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De regel noemt geen grond maar raakt een groep harder: indirect. Verander `"names_ground"` in `"yes"` en voer de cel opnieuw uit: direct, wat het effect ook is. Bij een indirecte regel komen daarna de drie vragen onder vraag 3 van het plaatje: Oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_verdict():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Was de regel met code 98 gerechtvaardigd?

        Het College vond de regel met code 98 van de Dienst Toeslagen indirect. `test` bevat zijn antwoorden op de drie vragen: `"aim"` (legitiem doel), `"appropriate"` (passend) en `"necessary"` (noodzakelijk). Vervang de `...` door regels in de vorm van het uitgewerkte voorbeeld: begin `verdict` als `"allowed"` (toegestaan), en schrijf voor elke vraag een `if` die van `verdict` `"forbidden"` (verboden) maakt als het antwoord `"no"` is. De laatste regel is `verdict`, zodat de check het ziet. De check probeert ook andere antwoorden.
        """),
        L.task("Reken `verdict` uit: `\"allowed\"` of `\"forbidden\"`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_verdict():
    box_verdict = L.code_box("verdict", lines=11)
    box_verdict
    return (box_verdict,)


@app.cell(hide_code=True)
def check_verdict(box_verdict):
    result_verdict = L.check("verdict", box_verdict.value)
    result_verdict
    return (result_verdict,)


@app.cell(hide_code=True)
def hints_verdict():
    L.hints(
        "verdict",
        [
            "Indirecte discriminatie mag alleen als alle drie de antwoorden ja zijn. Begin dus bij toegestaan, en laat elk nee dat veranderen.",
            "Eerst `verdict = \"allowed\"`. Dan drie keer een `if` die een antwoord in `test` vergelijkt met `\"no\"`, met `verdict = \"forbidden\"` eronder.",
            "De eerste `if`: `if test[\"aim\"] == ___:` met `verdict = \"forbidden\"` eronder; de andere twee net zo, met `\"appropriate\"` en `\"necessary\"`.",
        ],
        why="`verdict = \"allowed\"`, dan `if test[\"aim\"] == \"no\":` met `verdict = \"forbidden\"` eronder, en hetzelfde voor `\"appropriate\"` en `\"necessary\"`. Dat geeft `'forbidden'`: het antwoord bij `\"necessary\"` is nee.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Oefening 2. Direct, indirect of geen van beide?

    Zes gevallen. Deel elk in met de eerste twee vragen van het plaatje.
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
            "Vraag bij elk geval eerst of de regel zelf een discriminatiegrond noemt, en pas daarna of hij een groep harder raakt.",
            "Twee gevallen noemen de grond zelf: geslacht en huidskleur. Van de andere vier raakt er een geen enkele groep harder.",
            "Direct: de lagere score en \\_\\_\\_. Geen onderscheid: \\_\\_\\_.",
        ],
        why="**Direct:** de lagere score en de grenscontrole. **Indirect:** perfect Nederlands, de match en alleen scholieren. **Geen onderscheid:** de programmeertoets.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_college():
    mo.md(r"""
    ### Oefening 3. Wat doet een oordeel?
    """)
    return


@app.cell(hide_code=True)
def choice_college_cell():
    choice_college = L.choice(
        "college",
        question="Het College oordeelt dat het wervingsprogramma van een werkgever discrimineerde. Wat doet dat oordeel op zichzelf?",
        options={
            "fine": "De werkgever moet een boete betalen",
            "right": "Het oordeel heeft gezag, maar geen boete of schadevergoeding",
            "ban": "Het programma mag vanaf dan nergens in Nederland meer worden gebruikt",
        },
        correct="right",
        why={
            "fine": "Het College kan geen boete of schadevergoeding opleggen; daarvoor ga je naar de rechter. Zie *Het idee*.",
            "right": "Zijn oordelen zijn juridisch niet bindend, en je kunt er niet tegen in beroep; voor schadevergoeding ga je naar de rechter. Maar in ruim 80 procent van de gevallen waarin het discriminatie vaststelt, worden maatregelen genomen.",
            "ban": "Het College oordeelt over één geval en kan aanbevelingen doen, maar het kan een programma niet verbieden. Zie *Het idee*.",
        },
    )
    choice_college
    return (choice_college,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl) mat je allocatieve schade als het aandeel van elke groep dat een ja krijgt. Aan dat aandeel zie je indirecte discriminatie: een neutrale regel waarvan het ja voor een groep veel minder vaak komt. De volgende les, *De 80%-regel*, maakt van die aandelen één getal om te vergelijken, op een echte Amerikaanse tabel uit 1994.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_college, result_kinds, result_verdict):
    L.wrap_up(result_verdict, result_kinds, choice_college)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
