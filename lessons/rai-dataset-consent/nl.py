# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-dataset-consent. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-dataset-consent", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Gedragscodes en waarden in onderzoek](/course/open/rai-professional-ethics?lang=nl) en [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_paramount():
    quick_paramount = L.choice(
        "quick-paramount",
        question="De ACM-code noemt één overweging de belangrijkste. Welke?",
        options={
            "right": "Het algemeen belang",
            "employer": "Wat je werkgever van je vraagt",
            "science": "De vooruitgang van de wetenschap",
        },
        correct="right",
        why={
            "right": "De code noemt het algemeen belang de belangrijkste overweging, voor wie je ook werkt.",
            "employer": "Een gedragscode geldt voor wie je ook werkt. Zie *Het idee* in [Gedragscodes en waarden in onderzoek](/course/open/rai-professional-ethics?lang=nl).",
            "science": "Vooruitgang telt, maar de code zet iets anders voorop. Zie *Het idee* in [Gedragscodes en waarden in onderzoek](/course/open/rai-professional-ethics?lang=nl).",
        },
    )
    quick_paramount
    return (quick_paramount,)


@app.cell(hide_code=True)
def quick_check_no_license():
    quick_no_license = L.choice(
        "quick-no-license",
        question="Een dataset die je online vond, heeft geen licentie. Waar ga je van uit?",
        options={
            "free": "Dat je er alles mee mag, zoals bij CC0",
            "public": "Dat hij in het publieke domein valt, vrij voor iedereen",
            "right": "Dat kopiëren of delen niet mag: vraag eerst",
        },
        correct="right",
        why={
            "free": "CC0 is een licentie waarin de maker afziet van alle rechten. Zonder licentie is er niets gegeven. Zie *Het idee* in [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).",
            "public": "Iets valt in het publieke domein omdat de rechten verlopen of opgegeven zijn, niet omdat niemand een licentie schreef. Zie *Het idee* in [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).",
            "right": "Zonder licentie ga je ervan uit dat je hem niet mag kopiëren of delen, en vraag je het de maker voordat je erop voortbouwt.",
        },
    )
    quick_no_license
    return (quick_no_license,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Modellen worden getraind op enorme hoeveelheden tekst, foto's en opnames, waarvan veel gewoon gedownload is. Dat je een bestand kunt downloaden, zegt niet of je erop mag trainen, en ook niet of de mensen erin dat goed zouden vinden. Deze les houdt die twee vragen uit elkaar, en laat zien wanneer onderzoek met mensen een controle nodig heeft voordat het begint.

    **Een voorbeeld uit het dagelijks leven.** Een foto die jij van een vriend maakte, is voor het auteursrecht van jou. Toch vraag je het je vriend voordat je hem op een reclamebord zet.
    """)
    return


@app.cell(hide_code=True)
def predict_missing():
    guess_missing = L.guess(
        "missing",
        question="Onderzoekers zochten van meer dan 1.800 tekstdatasets uit waar ze vandaan kwamen; met zulke datasets worden modellen getraind. Hoe vaak denk je dat de licentie ontbrak op de grote websites waar zulke datasets staan?",
        options={
            "rare": "Ongeveer 1 op de 20",
            "third": "Ongeveer 1 op de 3",
            "right": "Meer dan 7 op de 10",
        },
        reveal="**Meer dan 7 op de 10.** En veel van de licenties die die websites wel lieten zien, klopten niet (Longpre en collega's, 2024).",
    )
    guess_missing
    return (guess_missing,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Twee vragen beslissen of je op een dataset mag trainen, en ze gaan over verschillende mensen.

        **De licentie** komt van wie de rechten op de data heeft, zoals de maker van de dataset of van de foto's erin. Die zegt wat je mag kopiëren, veranderen en delen, zoals in [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl). Maar een licentie geeft alleen rechten die de gever heeft. In 2019 publiceerde IBM *Diversity in Faces*, een lijst van ongeveer een miljoen foto's van Flickr die de fotografen onder een Creative Commons-licentie hadden gedeeld. Het auteursrecht was daar het probleem niet: niemand had de mensen op de foto's gevraagd of ze wilden helpen gezichtsherkenning te trainen. Creative Commons zei zelf dat auteursrecht geen goed middel is om privacy te beschermen.

        **Toestemming** komt van de mensen over wie de data gaan, of die ze maakten: ze stemmen vrij en bewust in met een genoemd gebruik. Openbaar is niet hetzelfde als akkoord: een bericht op een open forum is geschreven voor de lezers, niet om er een model op te trainen. Onder de AVG blijft informatie over een persoon een persoonsgegeven, ook als iedereen haar kan zien. Of de wet trainen zonder licentie toestaat, verschilt per land, en daarover lopen nog rechtszaken; deze les gaat over wat je kunt verantwoorden, niet alleen over wat mag.

        **Ethische toetsing.** Onderzoek met mensen begint met een controle door een ethische commissie: in de VS een institutional review board, in Nederland een medisch-ethische toetsingscommissie voor medisch onderzoek, en vaak een facultaire commissie voor ander onderzoek. Ook gebruikersonderzoek en sommig labelwerk hebben die nodig. In januari 2012 veranderde Facebook een week lang wat 689.003 gebruikers in hun nieuwsoverzicht zagen, om te zien of emoties zich verspreiden. De gebruikers werd niets gevraagd; ze hadden alleen ingestemd met de gebruiksvoorwaarden van Facebook. Het tijdschrift dat het onderzoek in 2014 publiceerde, voegde er een *editorial expression of concern* aan toe.

        **Data die je niet terugkrijgt.** In 2020 haalde MIT zijn dataset *80 Million Tiny Images* offline, nadat Vinay Prabhu en Abeba Birhane er beledigende labels en beelden in hadden gevonden, allemaal van het web gehaald zonder iemand te vragen. Kopieën, en modellen die erop getraind waren, verdwenen niet. De course houdt een lijst bij van elke dataset die ze gebruikt, in `data/README.md` en `course/route/bronnen.yaml`: de bron, de licentie, en wat de course ermee mag. CIFAR-10, een klein gelabeld deel van Tiny Images, heeft geen licentie, dus de course laat je hem alleen downloaden; de foto's van Caltech-101 komen van het web, dus de course deelt geen model dat erop getraind is.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **ethische toetsing** | een controle door een ethische commissie, voordat onderzoek met mensen begint, dat de risico's aanvaardbaar zijn en dat de deelnemers geïnformeerd toestemming geven |
        """),
        L.watch_out("Toestemming geldt voor een doel. Wie akkoord ging met één onderzoek, ging niet akkoord met elk later gebruik van de data: als het gebruik verandert, vraag je het opnieuw."),
        mo.Html(C.picture({"title": "Twee vragen, vier gevallen", "columns": ["heeft een licentie", "heeft geen licentie"],
                           "rows": ["de mensen gingen akkoord", "de mensen zijn niet gevraagd"],
                           "cells": ["Common Voice van Mozilla: mensen lezen zinnen voor om hun stem te doneren, en de opnames komen uit onder CC0.",
                                     "Berichten die vrijwilligers aan één onderzoeksteam gaven, dat de set nooit een licentie gaf.",
                                     "Diversity in Faces van IBM: foto's die onder een Creative Commons-licentie gedeeld waren, verzameld voor gezichtsherkenning zonder de mensen erop te vragen.",
                                     "Berichten gekopieerd van een forum waarvan de regels er niets over zeggen."],
                           "both": "beide vragen in orde",
                           "alt": "Een raster van vier gevallen. Met een licentie en toestemming van de mensen zijn beide vragen in orde: Common Voice van Mozilla. Zonder licentie maar met toestemming: berichten die aan één onderzoeksteam gegeven zijn, dat ze geen licentie gaf. Met een licentie maar zonder de mensen te vragen: Diversity in Faces van IBM. Met geen van beide: berichten gekopieerd van een forum waarvan de regels er niets over zeggen."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo beslis je of je op een dataset mag trainen:

    1. Zoek uit wie hem maakte, en onder welke licentie.
    2. Ga na of de licentie toestaat wat je van plan bent: trainen, een model delen, geld verdienen.
    3. Zoek uit over wie de data gaan of wie ze maakte, en of die mensen akkoord gingen met dit gebruik.
    4. Werk je direct met mensen, vraag dan na of er ethische toetsing nodig is.
    5. Schrijf de antwoorden in de datasheet, en laat de data weg als een vraag geen goed antwoord heeft.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een team, verzonnen door de course, wil een model trainen op tekst die mensen schreven. De regel van het team: gebruik een set alleen als die een licentie heeft, en alleen als de schrijvers akkoord gingen. `licenses` bevat de licentie van elke set, `"none"` als die er geen heeft. Voer de cel uit: die controleert de eerste vraag.
    """)
    return


@app.cell
def example_first_question():
    _licenses = {"forum": "none", "survey": "CC BY", "reviews": "CC BY-SA"}

    for _name in _licenses:
        if _licenses[_name] != "none":
            print(_name, "heeft een licentie")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `survey` en `reviews` slagen voor de eerste vraag. Maar de mensen die de reviews schreven, zijn nooit gevraagd: Oefening 1 voegt de tweede vraag toe.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_usable():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Beide vragen

        De vijf sets van het team. `licenses` bevat de licentie van elke set, `agreed` of de schrijvers akkoord gingen met trainen. De loop gaat met `name` langs de sets, en de eerste vraag, de licentie, staat er al in. Vervang de `...` zo dat een set alleen in `usable` komt als de schrijvers ook akkoord gingen. De laatste regel is `usable`, zodat de check het ziet. De check probeert ook andere antwoorden.
        """),
        L.task("Zet in `usable` de namen van de sets die een licentie hebben en waarvan de schrijvers akkoord gingen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_usable():
    box_usable = L.code_box("usable", lines=12)
    box_usable
    return (box_usable,)


@app.cell(hide_code=True)
def check_usable(box_usable):
    result_usable = L.check("usable", box_usable.value)
    result_usable
    return (result_usable,)


@app.cell(hide_code=True)
def hints_usable():
    L.hints(
        "usable",
        [
            "Binnen de eerste `if` een tweede: alleen een set die voor allebei slaagt, komt bij de `append`.",
            "Een waarde die al `True` of `False` is, kan in zijn eentje achter `if` staan. Welke dictionary bevat zulke waarden? En wat komt er in de lijst: een naam of een licentie?",
            "`if ___[name]:` en daaronder `usable.append(___)`",
        ],
        why="`if agreed[name]:` en daaronder `usable.append(name)` geeft `['survey', 'interviews']`: `reviews` heeft een licentie maar geen toestemming, `chats` toestemming maar geen licentie.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_questions():
    mo.md(r"""
    ### Oefening 2. Welke vraag is niet in orde?

    Zes gevallen. Zet elk geval bij de vraag waarvoor het niet slaagt, of bij *beide vragen in orde*.
    """)
    return


@app.cell(hide_code=True)
def sort_questions_cell():
    sort_questions = L.sort("questions")
    sort_questions
    return (sort_questions,)


@app.cell(hide_code=True)
def check_questions(sort_questions):
    result_questions = L.check("questions", sort_questions)
    result_questions
    return (result_questions,)


@app.cell(hide_code=True)
def hints_questions():
    L.hints(
        "questions",
        [
            "Vraag bij elk geval twee dingen: is er een licentie, en gingen de mensen akkoord met dit gebruik?",
            "In twee gevallen gingen de mensen akkoord, maar gaf wie de rechten heeft geen licentie. In twee andere heeft een bedrijf de rechten en geeft het een licentie, maar zijn de mensen van wie de woorden zijn nooit gevraagd.",
            "Beide in orde: de vertalingen en \\_\\_\\_. Licentie ontbreekt: \\_\\_\\_ en het spel. Toestemming ontbreekt: \\_\\_\\_ en \\_\\_\\_.",
        ],
        why="**Beide in orde:** de vertalingen en de recepten. **Licentie ontbreekt:** de schoolfoto's en het spel. **Toestemming ontbreekt:** de klantchats en de podcasts.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_review():
    mo.md(r"""
    ### Oefening 3. Een test met medestudenten
    """)
    return


@app.cell(hide_code=True)
def choice_review_cell():
    choice_review = L.choice(
        "review",
        question="Een student wil een chatbot testen met 30 medestudenten en alles bewaren wat ze typen. Wat komt eerst?",
        options={
            "publish": "Alle chats publiceren, zodat anderen de uitkomsten kunnen nagaan",
            "right": "Hun toestemming, en nagaan of ethische toetsing nodig is",
            "nothing": "Niets: een test met medestudenten telt niet als onderzoek",
        },
        correct="right",
        why={
            "publish": "Publiceren maakt het erger: daar gingen de medestudenten ook niet mee akkoord. Zie *Het idee*.",
            "right": "De medestudenten moeten weten wat er bewaard wordt en daarmee akkoord gaan, en veel universiteiten vragen ook bij studentenprojecten om ethische toetsing.",
            "nothing": "Een test met mensen is onderzoek met mensen, ook in een studieproject: ze typen dingen over zichzelf die bewaard worden. Zie *Het idee*.",
        },
    )
    choice_review
    return (choice_review,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl) schreef je op waar een dataset vandaan komt; in [Gedragscodes en waarden in onderzoek](/course/open/rai-professional-ethics?lang=nl) zag je wat een professional het publiek verschuldigd is. Hier komen die twee samen: een licentie zegt wat de maker toestaat, toestemming waar de mensen in de data mee akkoord gingen, en ethische toetsing controleert onderzoek met mensen voordat het begint. In de volgende les, *COMPAS en een sollicitatietool*, bekijk je twee echte systemen die misgingen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_review, result_questions, result_usable):
    L.wrap_up(result_usable, result_questions, choice_review)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
