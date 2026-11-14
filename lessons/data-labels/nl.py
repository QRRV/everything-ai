# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les data-labels. Route: course/route/lessen/klassiek-ml.yaml. Checks en plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("data-labels", "nl")
    C = L.checks   # een webwinkel die de course simuleert: 300 bestellingen, hun retouren, en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Classificatie: een klasse voorspellen](/course/open/ml-classification?lang=nl) en één over [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_classes():
    quick_classes = L.choice(
        "quick-classes",
        question="Het model van een webwinkel zegt bij elke bestelling: die komt terug, of die komt niet terug. Wat voor taak is dat?",
        options={
            "regression": "Regressie: het voorspelt een getal, 1 of 0",
            "right": "Classificatie: het label is een van twee klassen",
            "neither": "Geen van beide, want niemand weet het antwoord nog",
        },
        correct="right",
        why={
            "regression": "Als je de klassen 1 en 0 noemt, worden het nog geen getallen om te voorspellen: een waarde ertussenin, zoals 0,4 retour, betekent niets. Zie *Het idee* in [Classificatie: een klasse voorspellen](/course/open/ml-classification?lang=nl).",
            "neither": "Niemand weet het antwoord op het moment dat het model voorspelt; dat geldt voor elke voorspelling. Het antwoord komt later, als het label. Zie *Het idee* in [Classificatie: een klasse voorspellen](/course/open/ml-classification?lang=nl).",
            "right": "Twee klassen, teruggestuurd en gehouden. Vandaag gaat het erover waar zo'n label vandaan komt, en wanneer.",
        },
    )
    quick_classes
    return (quick_classes,)


@app.cell(hide_code=True)
def quick_check_license():
    quick_license = L.choice(
        "quick-license",
        question="Je wilt trainen op een tabel met bestellingen die je online vond, zonder licentie. Wat doe je?",
        options={
            "online": "Gebruik de tabel zoals je wilt: die staat online",
            "name": "Gebruik de tabel, en noem waar je die vond",
            "right": "Vraag het eerst de maker: zonder licentie mag je de tabel niet zomaar delen of erop voortbouwen",
        },
        correct="right",
        why={
            "online": "Gratis te downloaden is niet hetzelfde als vrij te gebruiken: de licentie zegt wat je mag. Zie *Het idee* in [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).",
            "name": "De maker noemen is wat CC BY vraagt, maar deze tabel heeft helemaal geen licentie, dus niets staat je nog toe de tabel te gebruiken. Zie *Het idee* in [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).",
            "right": "Goed, en bestellingen zijn gegevens over mensen, dus toestemming telt ook. Waar de data vandaan komt, hoort in de datasheet, en vandaag komt er één regel bij: waar de labels vandaan komen.",
        },
    )
    quick_license
    return (quick_license,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een webwinkel wil een model dat voorspelt welke bestellingen terugkomen, om klanten te helpen de goede maat te kiezen. Het model leert van eerdere bestellingen met een label: teruggestuurd of gehouden. Maar niemand tikt dat label in. Het verschijnt als er een pakket terugkomt, tot 30 dagen na de bestelling. Wat is tot die tijd het label van een bestelling die nog niet terug is?

    **Een voorbeeld uit het dagelijks leven.** Een restaurant vraagt elke gast om de maaltijd achteraf een cijfer te geven. Een gast die nog niet antwoordde, gaf geen slecht cijfer: het cijfer is onbekend. En de gasten die als eerste antwoorden, zijn vaak de gasten met een klacht. Tel je de stille gasten als tevreden, of tel je alleen de eerste antwoorden, dan zit je er beide keren naast.
    """)
    return


@app.cell(hide_code=True)
def predict_rate_cell():
    guess_rate = L.guess(
        "rate-guess",
        question="Op dag 60 heeft de winkel 95 bestellingen en tot nu toe 15 retouren. 49 van de bestellingen zijn jonger dan 30 dagen, dus een deel daarvan kan nog terugkomen. Uiteindelijk komt ongeveer één op de vijf bestellingen terug. Wat geeft 15 / 95?",
        options={
            "right": "Te weinig: 16%, want een deel van de jonge bestellingen komt nog terug",
            "fair": "Ongeveer goed: 16% ligt dicht bij één op de vijf",
            "much": "Te veel, want retouren komen het eerst binnen",
        },
        reveal="**Te weinig: 16%.** 15 / 95 telt de jonge bestellingen die nog niet terug zijn als gehouden, maar een deel daarvan komt nog terug. De 46 bestellingen van minstens 30 dagen oud geven 9 / 46 = 20%.",
    )
    guess_rate
    return (guess_rate,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Waar een label vandaan komt.** Elk label in een dataset is ergens gemaakt, op een van drie manieren.

        - Een mens oordeelt: iemand leest elk bericht en zet er spam of geen spam bij, of een arts geeft aan wat er op een scan te zien is. Dat is een **annotatielabel**. Het zegt wat die persoon denkt dat waar is, en het kost tijd en geld; de volgende les gaat erover hoe je dat goed doet.
        - Een meting: een sensor legt vast dat een machine uitviel, of een labtest geeft een uitslag. Het label is zo goed als het instrument.
        - Gedrag: wat mensen doen, geeft het label gratis: een bestelling komt terug of niet, een lezer klikt of niet, een lening wordt terugbetaald of niet. Dat is een **natural label**. Zulke labels zijn er volop, maar ze kunnen lang op zich laten wachten, en ze zeggen wat mensen deden, en dat is niet altijd wat je wilt weten: een klik zegt dat een kop iemand nieuwsgierig maakte, niet dat het verhaal goed is.

        **Het wachten.** Een klik komt binnen een paar seconden, een retour binnen 30 dagen, en of een lening terugbetaald wordt, weet je pas na jaren. Zolang het wachten niet voorbij is, is het label onbekend, niet 0. In Python zegt `None`, het woord voor niets, precies dat: nog geen label.

        **De feedback loop.** Het model voorspelt, de winkel doet er iets mee, klanten reageren, en hun reacties komen terug als labels voor het volgende model. Die weg terug is een **feedback loop**, en de lengte ervan is hoe lang een label erover doet om binnen te komen: hier 30 dagen. De loop voedt zichzelf ook: de winkel ziet alleen retouren van wat ze verkocht, en een nieuwsapp ziet alleen klikken op wat ze liet zien, dus een model leert alleen over wat eerdere keuzes doorlieten.

        **Hetzelfde wachten voor elk voorbeeld.** Geef een bestelling op een bepaalde dag pas een label als haar 30 dagen voorbij zijn. Beide manieren om het wachten over te slaan, geven een verkeerd percentage. Tel je de jonge bestellingen die nog niet terug zijn als gehouden, dan krijg je te weinig, want een deel ervan komt nog terug. Tel je de jonge bestellingen die al terugkwamen wel mee, maar de jonge die nog niet terug zijn niet, dan krijg je te veel, want van de nieuwste bestellingen ken je tot nu toe alleen de retouren.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **annotatielabel** | een label dat een mens geeft nadat die het voorbeeld beoordeelde |
        | **natural label** | een label dat gratis ontstaat uit wat mensen doen, zoals een retour of een klik |
        | **feedback loop** | de weg van de voorspellingen van een model, via wat mensen doen, terug naar de training data ervan; de lengte is hoe lang een label erover doet om binnen te komen |
        """),
        L.watch_out("Een bestelling die nog niet terug is, is geen gehouden bestelling. Zolang haar 30 dagen niet voorbij zijn, is haar label onbekend: `None`, niet 0."),
        mo.md(r"""
        De course simuleerde een webwinkel: 300 bestellingen op dag 0 tot en met 179, waarvan er 61, ongeveer één op de vijf, binnen 30 dagen terugkomen. Met de slider kies je welke dag het vandaag is. Links elke bestelling tot nu toe, op haar dag, in de rij van haar label: teruggestuurd (gevuld), gehouden (open) of nog onbekend (grijs); de grijze band is de laatste 30 dagen. Rechts het retourpercentage op drie manieren: *onbekend = 0* telt de bestellingen die nog niet bekend zijn als gehouden, *na 30 dagen* wacht voor elke bestelling 30 dagen, en *ook vroeg* telt de vroege retouren van jonge bestellingen mee, maar de jonge bestellingen die nog niet terug zijn niet. De gestreepte lijn is het deel van alle 300 bestellingen dat terugkomt, 20%, en dat weet alleen de simulatie. Beweeg de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_today_cell():
    def _frame(today):
        said = C.today_said(today, L.lang)
        alt = ("Links de bestellingen van een webwinkel naar de dag waarop ze geplaatst werden, in drie rijen: teruggestuurd, gehouden en nog onbekend, "
               "met de laatste 30 dagen grijs. Rechts drie staven met het retourpercentage als onbekende bestellingen tellen als gehouden, "
               "na 30 dagen, en met de vroege retouren van jonge bestellingen erbij, naast een gestreepte lijn op 20%. " + said)
        return (C.today_picture(today, C.words(L.lang) | {"alt": alt}, L.lang), said)

    explore_today = L.explore("today", "vandaag", _frame, start=60, values=C.TODAYS, role="",
                              task="Ga van dag 30 naar dag 180. Welk van de drie percentages blijft bij de gestreepte lijn, en wanneer liggen de andere twee er het verst vanaf?")
    explore_today
    return


@app.cell(hide_code=True)
def text_after_today():
    mo.md(r"""
    Op dag 30 is nog geen enkele bestelling 30 dagen oud: er is nog geen label, en de twee andere percentages liggen het verst van de lijn, op 13% en 100%. Vanaf dag 60 geeft 30 dagen wachten elke keer 20% of 21%, dicht bij de 20% die echt terugkomt. Onbekende bestellingen als gehouden tellen, geeft te weinig, 16% op dag 60, en de vroege retouren ook meetellen geeft te veel, 29% op dag 60. Hoe langer de winkel verkoopt, hoe kleiner het verschil, want de laatste 30 dagen worden een steeds kleiner deel van alle bestellingen: op dag 180 zijn ze 20% (onbekend = 0), 21% (na 30 dagen) en 22% (ook vroeg). Maar een model dat van de nieuwste bestellingen moet leren, ziet het grootste verschil. Met 300 bestellingen springen de percentages ook een of twee punten op en neer door toeval.

    ### Recept

    Zo geef je voorbeelden een label als dat label uit gedrag komt:

    1. Schrijf op waar elk label vandaan komt, een mens, een meting of gedrag, en wat het echt zegt.
    2. Zoek bij een label uit gedrag het wachten op: hoe lang duurt het tot het label vaststaat?
    3. Geef een voorbeeld pas een label als het wachten voorbij is; tot dan is het label `None`.
    4. Tel en train alleen met voorbeelden waarvan het label vaststaat.
    5. Schrijf de bron van de labels en het wachten in de datasheet, naast de data.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Acht bestellingen op dag 60: de dag waarop elke bestelling geplaatst werd, en of ze tot nu toe terugkwam. De cel print voor elke bestelling haar dag, haar leeftijd in dagen, of ze minstens 30 dagen oud is, en of ze terugkwam. Voer de cel uit.
    """)
    return


@app.cell
def example_ages():
    today = 60
    ordered = [5, 12, 20, 30, 38, 44, 52, 57]
    returned = [False, True, False, False, True, False, True, False]

    for i in range(len(ordered)):
        age = today - ordered[i]                           # hoeveel dagen geleden de bestelling geplaatst werd
        print(ordered[i], age, age >= 30, returned[i])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste vier bestellingen zijn 55, 48, 40 en 30 dagen oud: hun retouren zijn allemaal binnen, dus hun labels zijn 0, 1, 0 en 0. De andere vier zijn 22, 16, 8 en 3 dagen oud. Twee daarvan kwamen al terug, maar de twee andere kunnen nog terugkomen, dus alle vier wachten.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_label():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Pas na 30 dagen een label

        `label(ordered, returned, today)` krijgt de dag waarop een bestelling geplaatst werd, of ze tot nu toe terugkwam, en vandaag. De functie moet `None` geven zolang de bestelling jonger is dan 30 dagen, en daarna 1 als ze terugkwam en 0 als dat niet zo is. Vervang de `...`. De check roept haar aan voor de acht bestellingen van het uitgewerkte voorbeeld op dag 60, zoals `label(30, False, 60)`, en voor nog vijf bestellingen op andere dagen.
        """),
        L.task("Laat `label` `None`, 1 of 0 geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_label():
    box_label = L.code_box("label", lines=8)
    box_label
    return (box_label,)


@app.cell(hide_code=True)
def check_label(box_label):
    result_label = L.check("label", box_label.value)
    result_label
    return (result_label,)


@app.cell(hide_code=True)
def hints_label():
    L.hints(
        "label",
        [
            "Twee vragen, in deze volgorde: is de bestelling oud genoeg om haar label te kennen? Pas daarna: kwam ze terug?",
            "`age` is hoeveel dagen geleden de bestelling geplaatst werd. Zolang dat onder 30 ligt, kan de functie met `return` meteen `None` geven; daarna beslist `returned` tussen 1 en 0.",
            "`if age < ___:` met `return None` één stap ingesprongen, dan `if returned:` met `return ___` één stap ingesprongen, en aan het eind `return ___`.",
        ],
        why="Een bestelling die jonger is dan 30 dagen, kan nog terugkomen, dus haar label is nog onbekend, ook als ze al terugkwam: anders kregen de nieuwste bestellingen alleen een label als het retouren zijn. Na 30 dagen zijn alle retouren binnen en staat het label vast.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rate():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Het retourpercentage van de bekende labels

        `return_rate(labels)` krijgt een lijst labels, elk 1, 0 of `None`, en moet het deel teruggestuurde bestellingen geven van de bestellingen waarvan het label bekend is. Vervang de `...`. De check roept `return_rate([0, 1, 0, 0, None, None, None, None])` aan, de labels van de acht bestellingen op dag 60, wat 0,25 moet geven, en nog vier lijsten.
        """),
        L.task("Laat `return_rate` alleen de labels tellen die bekend zijn."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rate():
    box_rate = L.code_box("return-rate", lines=10)
    box_rate
    return (box_rate,)


@app.cell(hide_code=True)
def check_rate(box_rate):
    result_rate = L.check("return-rate", box_rate.value)
    result_rate
    return (result_rate,)


@app.cell(hide_code=True)
def hints_rate():
    L.hints(
        "return-rate",
        [
            "Een deel is een aantal gedeeld door een aantal. Welke labels tel je boven de streep, en welke eronder?",
            "Boven de streep: de labels die 1 zijn. Eronder: elk label dat bekend is, 1 of 0. Een `None` telt bij geen van beide. Vergelijk een label met `==`.",
            "`if value == 1:` telt 1 op bij `returned` en bij `known`, en `if value == ___:` telt alleen 1 op bij `known`.",
        ],
        why="Van de vier bekende labels is er één 1, dus het retourpercentage is 1 / 4 = 0,25. De vier `None` tellen als gehouden zou 1 / 8 = 0,125 geven: te weinig, want een deel van die bestellingen komt nog terug.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_clicks():
    mo.md(r"""
    ### Oefening 3. Wat een klik zegt
    """)
    return


@app.cell(hide_code=True)
def choice_clicks_cell():
    choice_clicks = L.choice(
        "what-clicks-say",
        question="Een nieuwsapp wil een model dat goede verhalen kiest. Ze geeft een verhaal label 1 als een lezer erop klikt. Wat leren deze labels het model?",
        options={
            "true": "Welke verhalen waar en goed geschreven zijn",
            "right": "Op welke koppen geklikt wordt, of het verhaal nu goed is of niet",
            "liked": "Welke verhalen lezers goed vonden nadat ze ze lazen",
        },
        correct="right",
        why={
            "true": "Een klik komt vóór het lezen: die zegt dat de kop iemand nieuwsgierig maakte, niet dat het verhaal waar is. Een natural label zegt wat mensen deden. Of een verhaal waar is, moet een mens beoordelen: een annotatielabel.",
            "liked": "Een klik komt vóór het lezen, dus die kan niet zeggen of de lezer het verhaal goed vond. Uitlezen of een cijfer zou dichterbij komen, en die labels komen later.",
            "right": "Goed: een natural label zegt wat mensen deden. En de loop voedt zichzelf: de app laat verhalen zien die klikken krijgen, dus ze leert alleen over klikken op wat ze liet zien.",
        },
    )
    choice_clicks
    return (choice_clicks,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Het label van een bestelling is een van twee klassen, zoals in [Classificatie: een klasse voorspellen](/course/open/ml-classification?lang=nl), en de datasheet uit [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl) krijgt er één regel bij: waar de labels vandaan komen en hoe lang ze erover doen om binnen te komen. De volgende les, *Goed labelen: richtlijnen en overeenstemming*, gaat over labels van mensen: hoe twee mensen dezelfde voorbeelden kunnen labelen en het eens worden.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_clicks, result_label, result_rate):
    L.wrap_up(result_label, result_rate, choice_clicks)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
