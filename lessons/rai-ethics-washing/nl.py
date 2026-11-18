# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-ethics-washing. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-ethics-washing", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Principes en richtlijnen](/course/open/rai-principles?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_common():
    quick_common = L.choice(
        "quick-common",
        question="Welk principe noemden de meeste van de 84 documenten?",
        options={
            "sustainability": "Duurzaamheid",
            "right": "Transparantie",
            "solidarity": "Solidariteit",
        },
        correct="right",
        why={
            "sustainability": "Duurzaamheid hoorde bij de minst genoemde: 14 van de 84. Zie het plaatje in [Principes en richtlijnen](/course/open/rai-principles?lang=nl).",
            "right": "Transparantie, in 73 van de 84 documenten, meer dan elk ander principe.",
            "solidarity": "Solidariteit werd het minst genoemd van allemaal: 6 van de 84. Zie het plaatje in [Principes en richtlijnen](/course/open/rai-principles?lang=nl).",
        },
    )
    quick_common
    return (quick_common,)


@app.cell(hide_code=True)
def quick_check_meaning():
    quick_meaning = L.choice(
        "quick-meaning",
        question="Een richtlijn vraagt om transparantie en zegt verder niets. Wat zegt dat je?",
        options={
            "explain": "Dat elke beslissing uitgelegd moet worden aan wie het betreft",
            "code": "Dat de code van het systeem openbaar moet",
            "right": "Nog niet welke: AI melden, openleggen of uitleggen",
        },
        correct="right",
        why={
            "explain": "Dat is één betekenis, maar anderen noemen een melding dat er AI wordt gebruikt, of een gepubliceerde dataset, ook transparantie. Zie *Het idee* in [Principes en richtlijnen](/course/open/rai-principles?lang=nl).",
            "code": "Dat is één betekenis van meerdere. Zie *Het idee* in [Principes en richtlijnen](/course/open/rai-principles?lang=nl).",
            "right": "Transparantie kan vragen om een melding dat er AI aan het werk is, een blik in het systeem, of uitleg bij één beslissing. Het woord alleen zegt niet welke.",
        },
    )
    quick_meaning
    return (quick_meaning,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een principe opschrijven kost weinig. In maart 2019 kondigde Google een externe raad aan die het bedrijf zou adviseren over de ethiek van AI; na protest over wie erin zat, hief Google de raad nog geen twee weken later op. Een principe, een commissie of een verklaring kan een organisatie verantwoord laten lijken zonder dat er iets verandert aan wat ze bouwt. Deze les geeft je een korte toets om een belofte die iets kost te onderscheiden van een belofte die niets kost.

    **Een voorbeeld uit het dagelijks leven.** Een winkel belooft "tevreden of geld terug". De ene winkel neemt alles binnen 30 dagen terug, zonder vragen. De andere heeft geen retourbalie, geen formulier en geen adres. Allebei deden ze dezelfde belofte.
    """)
    return


@app.cell(hide_code=True)
def predict_code():
    guess_code = L.guess(
        "code",
        question="Onderzoekers lieten een groep softwareontwikkelaars de ethische code lezen van de ACM, de grootste vereniging voor informatica, en vroegen hun daarna wat ze zouden doen in een reeks lastige situaties. Een andere groep zag de code niet. Hoeveel veranderde de code aan de beslissingen, denk je?",
        options={
            "lot": "Veel",
            "some": "Een beetje",
            "right": "Geen verschil gevonden",
        },
        reveal="**Geen verschil gevonden.** Studenten en professionele ontwikkelaars die de code gelezen hadden, beslisten niet anders dan wie hem niet gelezen had (McNamara, Smith en Murphy-Hill, 2018). Een tekst alleen veranderde niet wat mensen deden.",
    )
    guess_code
    return (guess_code,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Ethics washing** is principes, commissies of verklaringen gebruiken om verantwoord over te komen, terwijl je echte veranderingen of bindende regels ontloopt. De naam is afgeleid van greenwashing. In 2018 waarschuwde de onderzoeker Ben Wagner dat ethiek een vlucht voor regelgeving kon worden: een bedrijf dat zijn eigen principes schrijft, hoopt misschien dat wetgevers het met rust laten. In 2019 waarschuwde de filosoof Thomas Metzinger, een van de leden van de Europese expertgroep uit de vorige les, voor ethics washing in het werk van die groep: hij schreef dat de *rode lijnen*, toepassingen van AI die nooit toegestaan zouden mogen zijn, onder druk van de industrie waren afgezwakt en uit de definitieve tekst gehaald.

        Een principe kost niets zolang er niets gebeurt als het gebroken wordt. Drie vragen geven het een prijs:

        - **Een controle**: hoe zou iemand ontdekken dat het gebroken werd? Een toets, een audit, een getal dat gepubliceerd wordt.
        - **Een eigenaar**: wie staat ervoor in? Een persoon of team met een naam, niet alleen "wij".
        - **Een gevolg**: wat gebeurt er als het gebroken wordt? Een model gaat offline, een lancering wacht, wie klaagt krijgt antwoord.

        In één zin: wat zou er veranderen, en wie zou het merken, als de belofte gebroken werd? Een belofte die geen van de drie haalt, is alleen woorden. Dat maakt niet elk principe tot washing: een principe kan een eerste stap zijn, en de vraag is wat erna komt.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **ethics washing** | principes, commissies of verklaringen gebruiken om verantwoord over te komen, terwijl je echte veranderingen of bindende regels ontloopt; naar het voorbeeld van greenwashing |
        """),
        L.watch_out("Een eigenaar alleen is niet genoeg. Een commissie waarvan het advies geheim is en niet gevolgd hoeft te worden, heeft leden die ervoor instaan, maar zonder controle en zonder gevolg verandert er niets als ze genegeerd wordt."),
        mo.Html(C.picture({"title": "Zes beloften van een bedrijf, langs drie vragen", "tests": ["controle", "eigenaar", "gevolg"],
                           "words": "alleen woorden",
                           "alt": "Zes beloften van een verzonnen bedrijf. Fair AI en planet hebben geen controle, geen eigenaar en geen gevolg: alleen woorden. De bias test heeft een controle en een eigenaar, de outside test alleen een controle, de ethics board alleen een eigenaar, en de belofte over complaints heeft alle drie."})),
        mo.md(r"""
        Een bedrijf dat de course verzon, met zes beloften. *Fair AI*: we zetten ons in voor eerlijke AI. *Bias test*: elk model wordt per groep op fouten getest, en het hoofd van het datateam tekent de test af. *Outside test*: deskundigen van buiten proberen elk model te breken voordat het uitkomt. *Ethics board*: een commissie met leden op naam adviseert ons. *Complaints*: iedereen kan klagen, een team op naam antwoordt binnen 30 dagen, en een model dat blijft falen gaat offline. *Planet*: we geven om de planeet. Een gevulde stip betekent dat de belofte het heeft, een open stip dat ze het niet heeft.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo toets je een belofte over ethiek:

    1. Schrijf de belofte op in één zin.
    2. Vraag hoe iemand zou ontdekken dat ze gebroken werd: is er een controle?
    3. Vraag wie ervoor instaat: is er een eigenaar met een naam?
    4. Vraag wat er gebeurt als ze gebroken wordt: is er een gevolg?
    5. Een belofte die geen van de drie haalt, is alleen woorden; zeg wat ze nodig zou hebben.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een school, verzonnen door de course, beoordeelt opstellen met AI en doet drie beloften. Een docent bekijkt elk cijfer, en het cijfer dat de docent geeft, is het cijfer dat telt. Leerlingen kunnen vragen waarom ze een cijfer kregen, en de directeur antwoordt. En de beoordeling is eerlijk. Drie dictionaries bevatten een 1 waar een belofte een controle, een eigenaar of een gevolg heeft, en een 0 waar dat niet zo is. De loop telt ze per belofte op. Voer de cel uit.
    """)
    return


@app.cell
def example_school():
    _check = {"teacher review": 1, "ask why": 0, "fair grading": 0}
    _owner = {"teacher review": 1, "ask why": 1, "fair grading": 0}
    _consequence = {"teacher review": 1, "ask why": 0, "fair grading": 0}

    for _promise in _check:
        print(_promise, _check[_promise] + _owner[_promise] + _consequence[_promise])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De controle door de docent haalt alle drie. Vragen waarom heeft een eigenaar, maar niemand controleert of de antwoorden komen, en er gebeurt niets als ze uitblijven. Eerlijk beoordelen haalt er geen: alleen woorden. Stap 5 van het recept vraagt welke beloften van een heel bedrijf alleen woorden zijn: oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_passed():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Drie toetsen per belofte

        Het bedrijf van het plaatje. `check`, `owner` en `consequence` bevatten de drie toetsen, 1 als een belofte slaagt en 0 als ze niet slaagt, en de loop gaat met `promise` langs de beloftes. Vervang de `...` door één regel die de drie toetsen voor de belofte optelt, zoals in het uitgewerkte voorbeeld, en het totaal in `passed` zet in plaats van het te printen. Een belofte met een totaal van 0 is alleen woorden. De laatste regel is `passed`, zodat de check het ziet. De check probeert ook andere gevolgen.
        """),
        L.task("Zet in `passed`, onder elke belofte, hoeveel van de drie toetsen ze haalt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_passed():
    box_passed = L.code_box("passed", lines=13)
    box_passed
    return (box_passed,)


@app.cell(hide_code=True)
def check_passed(box_passed):
    result_passed = L.check("passed", box_passed.value)
    result_passed
    return (result_passed,)


@app.cell(hide_code=True)
def hints_passed():
    L.hints(
        "passed",
        [
            "De som van het uitgewerkte voorbeeld, met het antwoord bewaard onder de belofte.",
            "Welke drie waarden horen bij deze belofte? En hoe zet je een waarde onder een key in een dictionary?",
            "`passed[promise] = check[promise] + ___ + ___`",
        ],
        why="`passed[promise] = check[promise] + owner[promise] + consequence[promise]` geeft `{'fair AI': 0, 'bias test': 2, 'outside test': 1, 'ethics board': 1, 'complaints': 3, 'planet': 0}`: `fair AI` en `planet` halen er geen, dus niemand controleert ze, niemand staat ervoor in, en er gebeurt niets als ze gebroken worden.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_notice():
    mo.md(r"""
    ### Oefening 2. Zou iemand het merken?

    Zes beloften van bedrijven die AI bouwen, verzonnen door de course. Zet elke belofte bij wat er gebeurt als ze gebroken wordt.
    """)
    return


@app.cell(hide_code=True)
def sort_notice_cell():
    sort_notice = L.sort("notice")
    sort_notice
    return (sort_notice,)


@app.cell(hide_code=True)
def check_notice(sort_notice):
    result_notice = L.check("notice", sort_notice)
    result_notice
    return (result_notice,)


@app.cell(hide_code=True)
def hints_notice():
    L.hints(
        "notice",
        [
            "Stel bij elke belofte de drie vragen van het recept: een controle, een eigenaar, een gevolg?",
            "Drie van de beloften noemen iets wat je kunt tellen of zien: een getal, een toets, een termijn. Welke drie?",
            "Kost niets: mensen eerst, \\_\\_\\_ en \\_\\_\\_. Iemand zou het merken: de foutpercentages, \\_\\_\\_ en \\_\\_\\_.",
        ],
        why="**Kost niets:** mensen eerst, we streven en de commissie. **Iemand zou het merken:** de foutpercentages (een gepubliceerd getal), geen lancering (een toets, een persoon en een lancering die wacht) en bezwaar (een mens en een termijn).",
    )
    return


@app.cell(hide_code=True)
def text_exercise_code_of_ethics():
    mo.md(r"""
    ### Oefening 3. Een code die niets veranderde
    """)
    return


@app.cell(hide_code=True)
def choice_code_of_ethics_cell():
    choice_code_of_ethics = L.choice(
        "code-of-ethics",
        question="Ontwikkelaars die een ethische code gelezen hadden, beslisten niet anders dan ontwikkelaars die dat niet hadden gedaan. Wat doet dat vermoeden?",
        options={
            "useless": "Dat ontwikkelaars helemaal niets over ethiek kunnen leren",
            "right": "Dat een tekst weinig verandert aan wat mensen doen",
            "bad": "Dat deze ethische code wel slecht geschreven moet zijn",
        },
        correct="right",
        why={
            "useless": "Het onderzoek testte het lezen van een code, niet andere manieren om ethiek te leren. Wat ontbrak er aan de code? Denk aan de drie vragen in *Het idee*.",
            "right": "Een code zonder controle, eigenaar of gevolg is een tekst die je kunt lezen en vergeten. Wat gedrag verandert, is wat erbij komt.",
            "bad": "Het onderzoek zegt niets over hoe goed de code geschreven was. Wat ontbrak er aan de code? Denk aan de drie vragen in *Het idee*.",
        },
    )
    choice_code_of_ethics
    return (choice_code_of_ethics,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Principes en richtlijnen](/course/open/rai-principles?lang=nl) zag je dat de meeste documenten een paar woorden delen en er verschillende dingen mee bedoelen. Hier toetste je of een principe überhaupt iets kost: een controle, een eigenaar en een gevolg maken van een woord een belofte. In de volgende les, *Ontwerpen met waarden*, ga je een stap verder: van een waarde, via regels, naar eisen die een engineer kan controleren.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_code_of_ethics, result_notice, result_passed):
    L.wrap_up(result_passed, result_notice, choice_code_of_ethics)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
