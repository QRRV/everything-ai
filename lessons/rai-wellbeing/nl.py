# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-wellbeing. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-wellbeing", "nl")
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
def quick_check_rule():
    quick_rule = L.choice(
        "quick-rule",
        question="Een regel zegt dat een model iemands geloof nooit mag gebruiken, ook niet waar zijn voorspellingen daar beter van worden. Bij welke theorie past die regel het best?",
        options={
            "outcome": "Consequentialisme: de regel geeft de beste uitkomst in totaal",
            "right": "Deontologie: sommige regels gelden wat de uitkomst ook is",
            "care": "Zorgethiek: de regel komt uit luisteren naar wie het raakt",
        },
        correct="right",
        why={
            "outcome": "De regel geldt ook waar je betere voorspellingen krijgt als je hem breekt. Welke theorie houdt vast aan een regel, wat de uitkomst ook is? Zie *Het idee* in [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).",
            "right": "De regel geldt ook als hem breken beter zou uitpakken, en dat is wat een plicht of een recht doet.",
            "care": "Hier wordt niemand iets gevraagd: het is een regel die geldt wat er ook gebeurt. Welke theorie doet dat? Zie *Het idee* in [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).",
        },
    )
    quick_rule
    return (quick_rule,)


@app.cell(hide_code=True)
def quick_check_veil():
    quick_veil = L.choice(
        "quick-veil",
        question="Welke van twee aanpakken kies je achter de sluier van onwetendheid van Rawls, gelezen zoals in die les?",
        options={
            "right": "De aanpak waarbij wie het slechtst af is, het best af is",
            "total": "De aanpak die in totaal het meeste goed doet",
            "vote": "De aanpak waar de meeste mensen in de zaal voor zouden stemmen",
        },
        correct="right",
        why={
            "right": "Je weet niet in welke groep je terechtkomt, dus zorg je voor de groep die het slechtst af is.",
            "total": "Dat is wat het consequentialisme optelt. Achter de sluier weet je niet wie je zult zijn: naar welke groep kijk je? Zie oefening 1 in [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).",
            "vote": "Achter de sluier weet niemand wie die zal zijn, dus een stemming vertelt je niets nieuws. Naar welke groep kijk je? Zie oefening 1 in [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).",
        },
    )
    quick_veil
    return (quick_veil,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een app die kiest welke video's je ziet, leert van waar je op klikt, en laat daar meer van zien. Dat werkt: mensen klikken. Maar een systeem dat een getal optimaliseert, heeft een keuze gemaakt over wat goed is voor mensen, vaak zonder het te zeggen. In 2018 betaalden onderzoekers mensen in de VS om vier weken te stoppen met Facebook, en vergeleken hen met mensen die het bleven gebruiken.

    **Een voorbeeld uit het dagelijks leven.** Een winkel die snoep naast de kassa zet, verkoopt meer snoep. Dat is wat klanten op het moment zelf pakken, niet wat de meesten voor zichzelf zouden kiezen.
    """)
    return


@app.cell(hide_code=True)
def predict_facebook():
    guess_facebook = L.guess(
        "facebook",
        question="Hoe voelden de mensen die vier weken stopten met Facebook zich volgens jou gemiddeld, vergeleken met wie het bleef gebruiken?",
        options={
            "worse": "Slechter: ze misten hun vrienden en het nieuws",
            "same": "Niet anders",
            "right": "Iets beter",
        },
        reveal="**Iets beter.** Gemiddeld zeiden ze dat ze iets gelukkiger en tevredener met hun leven waren, en in de weken erna gebruikten ze Facebook minder dan wie het was blijven gebruiken. Ze wisten ook iets minder van het nieuws. Waar mensen op klikken, is niet altijd wat hun leven beter maakt.",
    )
    guess_facebook
    return (guess_facebook,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Welzijn** is wat een leven goed maakt voor wie het leeft. Filosofen geven verschillende antwoorden op wat dat is, en elk antwoord meet iets anders:

        - *Hedonisme*: welzijn is plezier en de afwezigheid van pijn. Je meet gevoelens.
        - *Preference satisfaction*: welzijn is krijgen wat je wilt. Een klik is een wens van het moment; wat je zou willen als je erover nadenkt, kan anders zijn.
        - *Een lijst van goederen*: sommige dingen zijn goed voor iedereen, of je ze nu wilt of niet, zoals gezondheid, vriendschap en kennis.
        - De **capability approach** van Amartya Sen en Martha Nussbaum: kijk naar wat mensen echt kunnen doen en zijn, zoals kunnen lezen, ergens kunnen komen en meedoen in hun gemeenschap. Niet wat ze hebben of voelen, maar hun echte vrijheid om een leven te kiezen. Een lijst telt de goederen zelf; de capability approach vraagt of mensen vrij zijn om ze te bereiken.

        Wat een systeem ook optimaliseert, het heeft een van deze gekozen. Klikken zijn de voorkeur van het moment. Het consequentialisme van de vorige les telt welzijn op, maar moet zeggen welk welzijn het optelt.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **welzijn** | wat een leven goed maakt voor wie het leeft; theorieën zeggen plezier, krijgen wat je wilt, of een lijst van goederen zoals gezondheid en vriendschap |
        | **capability approach** | welzijn beoordelen op wat mensen echt kunnen doen en zijn, zoals lezen, ergens kunnen komen en meedoen in hun gemeenschap (Sen en Nussbaum) |
        """),
        L.watch_out("De capability approach betekent niet dat de app het beter weet. Hij vraagt wat mensen kunnen doen, en laat de keuze wat ze daarmee doen aan hen over."),
        mo.Html(C.picture({"title": "Een feed gerangschikt op klikken", "share": "aandeel van de kijkers", "clicks": "klikt", "learned": "kon iets nieuws",
                           "titles": ["prank", "gossip", "unboxing", "fix a bike", "cook rice", "first aid"],
                           "alt": "Zes video's in de volgorde waarin een app ze laat zien. Het aandeel dat klikt, daalt langs de feed van 0,62 naar 0,12; het aandeel dat daarna iets nieuws kon, stijgt van 0 naar 0,7."})),
        mo.md(r"""
        De feed van deze les, verzonnen door de course: zes video's in de volgorde waarin de app ze laat zien, gerangschikt op het aandeel van de kijkers dat klikt. Open staven zijn het aandeel dat klikt, gevulde staven het aandeel dat daarna iets nieuws kon, zoals een fiets maken (fix a bike): die gaan de andere kant op.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om te zien wat een systeem beter maakt:

    1. Schrijf op wat het systeem nu optimaliseert, zoals klikken, minuten of sterren.
    2. Vraag bij welke theorie van welzijn die maat hoort.
    3. Kies ook een maat voor de andere theorieën: een gevoel, een weloverwogen wens, een goed op de lijst, iets wat mensen kunnen.
    4. Scoor wat mensen echt krijgen, zoals de top van een feed, op elke maat.
    5. Waar de scores verschillen, schrijf je op welke het systeem dient, en welke het weglaat.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De feed van de afbeelding, in de volgorde van de app. De meeste kijkers zien alleen de top van een feed, dus stap 4 scoort de eerste drie rijen: `feed.iloc[:3]` houdt ze over, zoals een slice van een lijst de eerste drie waarden overhoudt. Voer de cel uit.
    """)
    return


@app.cell
def example_feed():
    import pandas as pd

    _titles = ["prank", "gossip", "unboxing", "fix a bike", "cook rice", "first aid"]
    _clicks = [0.62, 0.55, 0.48, 0.21, 0.18, 0.12]
    _learned = [0.0, 0.0, 0.1, 0.6, 0.5, 0.7]
    _feed = pd.DataFrame({"title": _titles, "clicks": _clicks, "learned": _learned})
    # 1. wat de app optimaliseert: klikken, in de volgorde waarin hij de video's laat zien
    print(_feed)
    # 4. wat mensen krijgen: de top drie, en de hele feed om mee te vergelijken
    print(round(_feed.iloc[:3]["clicks"].mean(), 2))
    print(round(_feed["clicks"].mean(), 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De top drie krijgt gemiddeld 0,55 voor klikken, tegen 0,36 voor de hele feed: de app doet waarvoor hij gebouwd is. Stap 2: een klik is een voorkeur van het moment. Of kijkers er beter van worden, kunnen alleen de andere maten zeggen. Oefening 1 scoort dezelfde top drie daarop.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_scores():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Drie scores voor één feed

        Dezelfde feed, nu ook met `glad`: het aandeel van de kijkers dat een week later zegt blij te zijn dat ze keken. Dat is een weloverwogen wens. De getallen heeft de course verzonnen. Vervang de `...` door één regel, zoals in het uitgewerkte voorbeeld. De laatste regel is `scores`, zodat de check het ziet. De check probeert ook andere getallen voor `learned`.
        """),
        L.task("Zet in `scores` het gemiddelde van elke maat over de eerste drie video's van de feed, onder de naam van de maat."),
    ])
    return


@app.cell(hide_code=True)
def exercise_scores():
    box_scores = L.code_box("scores", lines=13)
    box_scores
    return (box_scores,)


@app.cell(hide_code=True)
def check_scores(box_scores):
    result_scores = L.check("scores", box_scores.value)
    result_scores
    return (result_scores,)


@app.cell(hide_code=True)
def hints_scores():
    L.hints(
        "scores",
        [
            "Stap 4 van het recept, zoals in het uitgewerkte voorbeeld, maar met de kolom waarvan de naam in `measure` staat.",
            "Welke rijen ziet een kijker als eerste, en hoe houd je alleen die over? Neem dan het gemiddelde van één kolom.",
            "`scores[measure] = feed.___[___][measure].mean()`.",
        ],
        why="`feed.iloc[:3]` houdt de eerste drie video's over, en `[measure]` de kolom van deze ronde: ongeveer 0,55 voor clicks, 0,3 voor glad en 0,03 voor learned. De feed die wint op klikken, verliest op waar kijkers later blij mee zijn en op wat ze daarna kunnen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_measures():
    mo.md(r"""
    ### Oefening 2. Welke theorie?

    Acht dingen die een app zou kunnen meten. Zet elk ding bij de theorie van welzijn waar het bij hoort.
    """)
    return


@app.cell(hide_code=True)
def sort_measures_cell():
    sort_measures = L.sort("measures")
    sort_measures
    return (sort_measures,)


@app.cell(hide_code=True)
def check_measures(sort_measures):
    result_measures = L.check("measures", sort_measures)
    result_measures
    return (result_measures,)


@app.cell(hide_code=True)
def hints_measures():
    L.hints(
        "measures",
        [
            "Vraag bij elke maat: is het een gevoel, een wens, een goed dat telt of je het nu wilt of niet, of iets wat iemand kan doen?",
            "Bij *Gezondheid en vrienden* staat *wat ze ook willen*. Welke twee maten vragen wat mensen kunnen doen?",
            "Plezier: glimlachen en \\_\\_\\_. Krijgen wat je wilt: klikken en \\_\\_\\_. Een lijst: gezondheid en vrienden, en \\_\\_\\_. Wat mensen kunnen: iets nieuws kunnen en \\_\\_\\_.",
        ],
        why="**Plezier:** glimlachen en stemming. **Krijgen wat je wilt:** klikken en gevraagd. **Een lijst van goederen:** gezondheid en vrienden, en kennis. **Wat mensen kunnen doen en zijn:** iets nieuws kunnen en meedoen.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_goal():
    mo.md(r"""
    ### Oefening 3. Een doel voor een app
    """)
    return


@app.cell(hide_code=True)
def choice_goal_cell():
    choice_goal = L.choice(
        "goal",
        question="Een taalapp wordt gebouwd om één getal omhoog te krijgen. Welk getal past het best bij de capability approach?",
        options={
            "minutes": "Minuten per dag die mensen in de app doorbrengen",
            "stars": "De sterren die mensen de app na een week in de app store geven",
            "right": "Hoeveel mensen na een maand een kort gesprek kunnen voeren",
        },
        correct="right",
        why={
            "minutes": "Minuten in de app zijn wat mensen op het moment zelf kiezen, en een app kan mensen bezighouden zonder ze iets te leren. Wat kunnen ze daarna? Zie *Het idee*.",
            "stars": "Sterren zeggen hoe leuk mensen de app vonden, een gevoel of een wens. Wat kunnen ze daarna? Zie *Het idee*.",
            "right": "Een gesprek voeren is iets wat mensen nu kunnen, een echte vrijheid die ze eerst niet hadden. Wat ze ermee doen, is aan hen.",
        },
    )
    choice_goal
    return (choice_goal,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl) beoordeelde je één keuze door verschillende theorieën; hier zag je dat zelfs "het meeste goed" een theorie nodig heeft van wat goed is voor mensen. Wat mensen echt krijgen op meer dan één maat scoren, is dezelfde gewoonte als per groep tellen in [Twee soorten schade](/course/open/rai-two-kinds-of-harm?lang=nl): één getal kan verbergen waar het om gaat. In de volgende les, *Regels en dilemma's voor machines*, probeer je regels voor een machine te schrijven, en zie je waar ze botsen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_goal, result_measures, result_scores):
    L.wrap_up(result_scores, result_measures, choice_goal)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
