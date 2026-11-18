# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-dark-patterns. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-dark-patterns", "nl")
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
def quick_check_sum():
    quick_sum = L.choice(
        "quick-sum",
        question="Welke theorie beoordeelt een keuze op wat die oplevert, opgeteld over iedereen?",
        options={
            "right": "Consequentialisme",
            "deontology": "Deontologie, met plichten en rechten",
            "care": "Zorgethiek, vanuit relaties",
        },
        correct="right",
        why={
            "right": "Het consequentialisme telt het goede en het slechte op dat elke optie oplevert.",
            "deontology": "De deontologie vraagt welke plichten en rechten gelden, wat de uitkomst ook is. Zie *Het idee* in [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).",
            "care": "Zorgethiek begint bij wie van wie afhangt. Zie *Het idee* in [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).",
        },
    )
    quick_sum
    return (quick_sum,)


@app.cell(hide_code=True)
def quick_check_means():
    quick_means = L.choice(
        "quick-means",
        question="Immanuel Kant vond dat je een mens nooit alleen als middel mag gebruiken. Welke theorie is dat?",
        options={
            "virtue": "Deugdethiek",
            "right": "Deontologie",
            "veil": "Een eerlijke afspraak achter een sluier",
        },
        correct="right",
        why={
            "virtue": "Deugdethiek vraagt wat iemand met een goed karakter zou doen. Zie *Het idee* in [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).",
            "right": "Een plicht die geldt wat de uitkomst ook is: deontologie.",
            "veil": "De sluier van onwetendheid is het idee van John Rawls voor een regel die iedereen kan aanvaarden. Zie *Het idee* in [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).",
        },
    )
    quick_means
    return (quick_means,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Elke app en website is ontworpen om te sturen wat je doet: welke knop groot is, wat er voor je gekozen is, hoeveel stappen het kost om nee te zeggen. AI maakt dat sturen persoonlijk: een systeem kan leren wat bij jou werkt, en van moment tot moment veranderen wat het je laat zien. Deze les vraagt wanneer sturen in orde is, wanneer het tegen mensen in werkt, en wanneer het manipulatie wordt.

    **Een voorbeeld uit het dagelijks leven.** Een supermarkt legt snoep bij de kassa, op ooghoogte van een kind.
    """)
    return


@app.cell(hide_code=True)
def predict_share():
    guess_share = L.guess(
        "share",
        question="In 2019 doorzochten onderzoekers zo'n 11.000 webwinkels op trucs zoals nep-afteltimers. Op hoeveel daarvan denk je dat ze er minstens één vonden?",
        options={
            "few": "Ongeveer 1 op de 100",
            "right": "Ongeveer 1 op de 10",
            "half": "Ongeveer 1 op de 2",
        },
        reveal="**Ongeveer 1 op de 10**: 1.818 trucs op 1.254 websites, zoals timers die opnieuw beginnen, meldingen dat er nog maar een paar over zijn, en producten die zonder te vragen in het winkelmandje komen (Mathur en collega's, 2019). Hun crawler vond alleen trucs die in de tekst van een pagina staan, dus het echte aantal is waarschijnlijk hoger.",
    )
    guess_share
    return (guess_share,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **nudge**, een woord van Richard Thaler en Cass Sunstein (2008), is een verandering in hoe een keuze wordt voorgelegd die mensen voorspelbaar een kant op stuurt, zonder iets te verbieden of de kosten veel te veranderen: een volgorde, een herinnering, een standaardkeuze. Standaardkeuzes zijn sterk. In landen waar je orgaandonor bent tenzij je nee zegt, zijn veel meer mensen donor dan in landen waar je ja moet zeggen (Johnson en Goldstein, 2003). Sinds 2020 registreert Nederland iedereen van 18 jaar en ouder die geen keuze maakt als iemand zonder bezwaar.

        Een **dark pattern**, een naam van de ontwerper Harry Brignull (2010), is een ontwerp dat mensen iets laat doen wat ze niet van plan waren, tegen hun eigen belang in: een vakje dat al is aangevinkt, kosten die pas bij de laatste stap verschijnen, haast die niet echt is, een abonnement dat makkelijk begint en moeilijk stopt.

        Een **hypernudge**, een woord van de rechtsgeleerde Karen Yeung (2017), is een nudge die draait op data over jou, en die zich steeds en per persoon aanpast: een feed die verandert wat hij laat zien op basis van waar je eerder bij bleef hangen. Anderen zien misschien niet hetzelfde, en daardoor is het moeilijk op te merken en moeilijk te weerstaan.

        Sturen wordt *manipulatie* als het werkt door om iemands eigen redenen heen te gaan: als het op een verborgen manier inspeelt op een zwakke plek, zoals moeheid of de angst iets te missen (Susser, Roessler en Nissenbaum, 2019). Het schaadt **menselijke autonomie**, kiezen om je eigen redenen, en **menselijke waardigheid**, de waarde van ieder mens, die nooit alleen als middel gebruikt mag worden. De Digital Services Act van de EU verbiedt online platforms hun interface zo te ontwerpen dat die mensen misleidt of manipuleert, en de AI Act verbiedt AI-systemen die met manipulatieve technieken het gedrag van mensen zo verstoren dat het aanzienlijke schade veroorzaakt of waarschijnlijk zal veroorzaken.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **nudge** | een kleine verandering in hoe een keuze wordt aangeboden, zoals een standaardoptie, die mensen een kant op stuurt zonder iets te verbieden of de kosten te veranderen |
        | **dark pattern** | een ontwerptruc in een app of website die mensen keuzes laat maken die ze niet van plan waren, zoals een verstopte opzegknop of een vakje dat al is aangevinkt |
        | **hypernudge** | een nudge die draait op data over een persoon, en die een systeem steeds aan die persoon aanpast, zoals een feed die verandert op basis van waar iemand eerder bij bleef hangen |
        | **menselijke autonomie** | het vermogen van een mens om zelf keuzes te maken, om eigen redenen |
        | **menselijke waardigheid** | de waarde die ieder mens heeft, alleen al door mens te zijn, zodat niemand behandeld mag worden als alleen een object of een nummer |
        """),
        L.watch_out("Een nudge is niet fout omdat hij stuurt: elk ontwerp stuurt, want iets moet vooraan staan. Vraag in wiens belang hij stuurt, of mensen het kunnen zien, en hoe makkelijk ze iets anders kunnen kiezen."),
        mo.Html(C.picture({"title": "Stappen om lid te worden en op te zeggen, bij vijf abonnementen", "x": "stappen", "harder": "lastig op te zeggen",
                           "alt": "Vijf verzonnen abonnementen, met de stappen om lid te worden als open stip en de stappen om op te zeggen als gevulde stip. Music: 2 en 2. News: 3 en 9. Fitness: 2 en 7. Cloud: 4 en 3. Games: 1 en 6. News, fitness en games zijn moeilijker op te zeggen dan te nemen."})),
        mo.md(r"""
        Verzonnen door de course. Een open stip is hoeveel stappen het kost om lid te worden, een gevulde stip hoeveel om weer op te zeggen.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een ontwerp op dark patterns te controleren:

    1. Schrijf op welke keuze het ontwerp stuurt, en in wiens belang.
    2. Tel de stappen voor elke optie, zoals ja en nee, of lid worden en opzeggen.
    3. Zoek wat verstopt is of al is aangevinkt, en waar je wordt opgejaagd.
    4. Vraag of het ontwerp zich aan elke persoon aanpast, en met welke data.
    5. Maak de optie die goed is voor de persoon minstens zo makkelijk als die van het bedrijf.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Twee cookiebanners, verzonnen door de course. `_accept` bevat hoeveel klikken het bij elke banner kost om ja te zeggen, `_reject` hoeveel om nee te zeggen. De loop gaat langs de banners en print de banners waar nee zeggen meer klikken kost. Voer de cel uit.
    """)
    return


@app.cell
def example_banners():
    _accept = {"banner A": 1, "banner B": 1}
    _reject = {"banner A": 1, "banner B": 4}

    for _banner in _accept:
        if _reject[_banner] > _accept[_banner]:
            print(_banner, "maakt nee zeggen moeilijker")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Alleen banner B wordt geprint: nee zeggen kost daar vier klikken, en ja zeggen één. Een paar klikken extra lijken weinig, maar over miljoenen bezoekers bepalen ze waar veel mensen mee instemmen. Oefening 1 doet stap 2 van het recept voor de abonnementen in het plaatje.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_harder():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Welke zijn lastig op te zeggen?

        De abonnementen van het plaatje. `join` bevat hoeveel stappen het kost om lid te worden van elk abonnement, `leave` hoeveel om weer op te zeggen, en `harder` begint als een lege lijst. De loop gaat met `service` langs de abonnementen. Vervang de `...` door twee regels, zoals in het uitgewerkte voorbeeld, maar voeg de naam toe aan `harder` in plaats van hem te printen. De laatste regel is `harder`, zodat de check het ziet. De check probeert ook andere stappen.
        """),
        L.task("Laat `harder` de abonnementen geven die meer stappen kosten om op te zeggen dan om lid te worden."),
    ])
    return


@app.cell(hide_code=True)
def exercise_harder():
    box_harder = L.code_box("harder", lines=9)
    box_harder
    return (box_harder,)


@app.cell(hide_code=True)
def check_harder(box_harder):
    result_harder = L.check("harder", box_harder.value)
    result_harder
    return (result_harder,)


@app.cell(hide_code=True)
def hints_harder():
    L.hints(
        "harder",
        [
            "Stap 2 van het recept: vergelijk voor elk abonnement de stappen om op te zeggen met de stappen om lid te worden, zoals het uitgewerkte voorbeeld de klikken vergelijkt.",
            "Welke vergelijking zegt *meer stappen om op te zeggen dan om lid te worden*? En hoe voeg je een naam toe aan het eind van een lijst?",
            "`if leave[service] ___ join[service]:`",
        ],
        why="`if leave[service] > join[service]:` met `harder.append(service)` eronder geeft `['news', 'fitness', 'games']`: drie van de vijf zijn moeilijker op te zeggen, en `music`, met beide kanten op 2 stappen, is dat niet.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Oefening 2. Nudge, dark pattern of hypernudge?

    Zes ontwerpen, verzonnen door de course. Zet elk ontwerp bij de soort waar het bij hoort.
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
            "Vraag bij elk ontwerp: stuurt het openlijk, werkt het tegen de persoon in, of past het zich met data aan elke persoon aan?",
            "Twee ontwerpen verstoppen iets of jagen je op. Twee andere leren iets over elke persoon.",
            "Nudge: de salade en \\_\\_\\_. Dark pattern: \\_\\_\\_ en de timer. Hypernudge: \\_\\_\\_ en \\_\\_\\_.",
        ],
        why="**Nudge:** de salade en het pensioen. **Dark pattern:** het aangevinkte vakje en de timer. **Hypernudge:** de feed en de playlist.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_manipulation():
    mo.md(r"""
    ### Oefening 3. Wanneer wordt het manipulatie?
    """)
    return


@app.cell(hide_code=True)
def choice_manipulation_cell():
    choice_manipulation = L.choice(
        "manipulation",
        question="Een game toont een aanbod \"laatste kans\" precies als de data zeggen dat een speler moe is en net heeft verloren. Wat maakt dit manipulatie en geen nudge?",
        options={
            "price": "Het aanbod kost geld, en de speler heeft later misschien spijt van de aankoop",
            "game": "Het gebeurt in een game, waar mensen komen om te ontspannen en niet om te kopen",
            "right": "Het gebruikt een zwak moment in plaats van de eigen redenen van de speler",
        },
        correct="right",
        why={
            "price": "Veel eerlijke aanbiedingen kosten geld. Het gaat erom hoe het aanbod de speler ja laat zeggen. Zie *Het idee*.",
            "game": "De plek is niet het probleem: hetzelfde aanbod op een eerlijk moment kan in orde zijn. Zie *Het idee*.",
            "right": "Het is, zonder dat de speler het weet, getimed op een moment waarop die het het slechtst kan afwegen: het speelt in op een zwakke plek en gaat om de redenen van de speler heen, en dat schaadt de autonomie.",
        },
    )
    choice_manipulation
    return (choice_manipulation,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl) zag je de plicht om een mens nooit alleen als middel te gebruiken. Een dark pattern doet precies dat: het behandelt mensen als een bron van klikken, niet als mensen met hun eigen redenen. AI voegt de hypernudge toe: sturen dat zich aan elke persoon aanpast en moeilijk te zien is. In de volgende les, *Gedragscodes en waarden in onderzoek*, kijk je naar wat de mensen die zulke systemen bouwen het publiek verschuldigd zijn.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_manipulation, result_harder, result_kinds):
    L.wrap_up(result_harder, result_kinds, choice_manipulation)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
