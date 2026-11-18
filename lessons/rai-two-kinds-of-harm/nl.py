# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-two-kinds-of-harm. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-two-kinds-of-harm", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Wie staat er in de data, en wie ontbreekt?](/course/open/rai-bias-in-data?lang=nl)
    """)
    return


@app.cell(hide_code=True)
def quick_check_stakeholder():
    quick_stakeholder = L.choice(
        "quick-stakeholder",
        question="Een gemeente bepaalt welke straten eerst worden gerepareerd met een model dat meldingen uit een app leest. Wie zijn stakeholders van dat model?",
        options={
            "app": "Alleen de mensen die meldingen doen",
            "city": "Alleen de ambtenaren die de reparaties plannen, want zij gebruiken het model",
            "right": "Ook de bewoners die de app nooit gebruiken",
        },
        correct="right",
        why={
            "app": "Een stakeholder hoeft het systeem niet te gebruiken. Wat gebeurt er met een straat waar niemand de app gebruikt? Zie *Het idee* in [Wie staat er in de data, en wie ontbreekt?](/course/open/rai-bias-in-data?lang=nl)",
            "city": "De mensen die een systeem gebruiken zijn maar een deel van de stakeholders. Wie woont er in de straten die het achteraan zet? Zie *Het idee* in [Wie staat er in de data, en wie ontbreekt?](/course/open/rai-bias-in-data?lang=nl)",
            "right": "Ook hun straten worden beoordeeld, op meldingen die ze nooit doen, dus het model kan ze achteraan zetten. De mensen die de app gebruiken en de gemeente hebben ook belang.",
        },
    )
    quick_stakeholder
    return (quick_stakeholder,)


@app.cell(hide_code=True)
def quick_check_score():
    quick_score = L.choice(
        "quick-score",
        question="Een model heeft 92% van een test set goed, en 70% van groep B, die 8% van die test set is. Wat schrijf je bij de score?",
        options={
            "right": "92% in totaal, maar 70% voor groep B, 8% van de test set",
            "drop": "Niets meer: groep B telt al mee in de 92%, dus die zegt alles",
            "mean": "Het gemiddelde van de twee, 81%, als eerlijkere totaalscore",
        },
        correct="right",
        why={
            "right": "Stap 6 van het recept in die les: de score per groep naast de totale, en hoe groot de groep is, zodat een lezer ziet voor wie hij geldt.",
            "drop": "Groep B telt mee in de 92%, maar weegt maar 8%. Wat verbergt de totale score? Zie de formule in [Wie staat er in de data, en wie ontbreekt?](/course/open/rai-bias-in-data?lang=nl)",
            "mean": "Een gemiddelde van twee scores is een nieuw getal dat voor niemand geldt. Wat moet een lezer zien? Zie het recept in [Wie staat er in de data, en wie ontbreekt?](/course/open/rai-bias-in-data?lang=nl)",
        },
    )
    quick_score
    return (quick_score,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een model dat een lening afwijst, doet schade die je kunt aanwijzen: iemand kreeg iets niet. Maar in een onderzoek uit 2015 zochten Matthew Kay, Cynthia Matuszek en Sean Munson beroep voor beroep naar afbeeldingen van mensen aan het werk, en vergeleken wie er op de afbeeldingen stond met wie die beroepen echt doet. Bij die zoekopdracht werd niemand iets geweigerd, en toch klopte er iets niet. Deze les gaat over allebei die soorten schade, en over hoe je elk van de twee vindt.

    **Een voorbeeld uit het dagelijks leven.** Een schoolboek waarin elke dokter als man is getekend, weigert niemand een plek op de geneeskundeopleiding. Toch vertelt het elk kind dat het leest wie er dokter wordt.
    """)
    return


@app.cell(hide_code=True)
def predict_ceo():
    guess_ceo = L.guess(
        "ceo",
        question="Volgens de Amerikaanse arbeidsstatistiek in dat onderzoek was 27% van de CEO's in de VS vrouw. Welk deel van de afbeeldingen bij de zoekopdracht CEO toonde volgens jou een vrouw?",
        options={
            "same": "Ongeveer 27%: een zoekmachine laat de wereld zien zoals hij is",
            "more": "Meer dan 27%: zoekmachines proberen variatie te laten zien",
            "right": "11%",
        },
        reveal="**11%.** Dat is minder dan de helft van het aandeel onder echte CEO's. De zoekopdracht weigerde niemand iets, maar liet een wereld zien die eenzijdiger is dan de echte, en zo'n beeld kan een stereotype sterker maken.",
    )
    guess_ceo
    return (guess_ceo,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        In 2017 gaven Solon Barocas, Kate Crawford en anderen de twee soorten hun naam, en Crawfords lezing op NIPS dat jaar (nu NeurIPS) maakte ze bekend.

        - **Allocatieve schade** komt van wat een systeem uitdeelt: een lening, een baan, een plek op een school, maar ook een last zoals een extra controle. Het is schade als sommige groepen minder van het goede of meer van het slechte krijgen dan terecht is. Je kunt het meten: per groep het deel van de mensen dat een ja krijgt.
        - **Representatieve schade** komt van hoe een systeem mensen laat zien of beschrijft: als stereotype, met een belediging, of helemaal niet. Crawford telde ook een systeem mee dat sommige groepen slechter herkent, zoals de gezichtsdetector van de vorige les; latere lijsten noemen dat vaak apart: quality-of-service harm, schade doordat een dienst voor sommige groepen slechter werkt, en die meet je per groep, zoals de vorige les deed. Stereotypen, beledigingen en weggelaten groepen beslissen over niemand iets, dus er is geen ja om te tellen. Je moet ernaar zoeken: verzamel veel uitvoer, schrijf op wie er wordt getoond en hoe, en tel.

        Beide soorten gaan over **fairness**: het streven dat een systeem mensen of groepen niet onrechtvaardig behandelt. Voor wat een systeem uitdeelt, is het aandeel ja van een groep:
        """),
        L.formula(r"\text{aandeel ja van een groep} = \frac{\text{ja in de groep}}{\text{mensen in de groep}}"),
        mo.md(r"""
        Een verschil tussen het aandeel ja van twee groepen bewijst niet dat een systeem oneerlijk is: de groepen kunnen verschillen op punten die voor de beslissing tellen. Het vertelt je waar je moet kijken. Latere lessen vragen wanneer zo'n verschil onrechtvaardig is.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **allocatieve schade** | schade door wat een systeem uitdeelt, zoals leningen, banen of extra controles, als sommige groepen minder van het goede of meer van het slechte krijgen dan terecht is |
        | **representatieve schade** | schade door hoe een systeem groepen laat zien of beschrijft: stereotypen, beledigingen, of een groep die wordt weggelaten; het beslist over niemand iets |
        | **fairness** | het streven dat een systeem mensen of groepen niet onrechtvaardig behandelt |
        """),
        L.watch_out("Stereotypen, beledigingen en weggelaten groepen zie je niet in een score. Niemand wordt afgewezen, dus er is geen aandeel ja om te vergelijken: als je alleen beslissingen meet, zie je ze nooit."),
        mo.Html(C.picture({"loans": "Ja voor een lening", "loans_y": "aandeel ja", "ceo": "Vrouwen onder CEO's", "ceo_y": "aandeel",
                           "images": "beelden", "real": "echt",
                           "alt": "Links het aandeel ja van een leenmodel per groep: 0,75 voor groep A, 0,375 voor groep B. Rechts het aandeel vrouwen: 0,11 in de afbeeldingen bij de zoekopdracht CEO in een onderzoek uit 2015, 0,27 onder echte CEO's in de VS."})),
        mo.md(r"""
        Links allocatief: een leenmodel dat ja zegt tegen 0,75 van de aanvragen van groep A en tegen 0,375 van die van groep B. Die getallen heeft de course verzonnen. Rechts representatief: 0,11 van de afbeeldingen bij CEO toonde een vrouw, tegen 0,27 van de echte CEO's. Het verschil links gaat over wie iets krijgt, dat rechts over hoe een groep wordt getoond.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om beide soorten schade te vinden:

    1. Vraag wat het systeem uitdeelt (een ja, een lening, een controle), en wat het over mensen laat zien of schrijft.
    2. Voor wat het uitdeelt: deel per groep het aantal keer ja door het aantal mensen.
    3. Voor wat het laat zien: vraag om veel uitvoer, en schrijf bij elke uitvoer op wie er wordt getoond, en hoe.
    4. Tel per groep of rol, en vergelijk met iets om het naast te leggen, zoals het aandeel in de echte wereld.
    5. Schrijf op wat je vond, naast de score.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het leenmodel van de afbeelding: 20 aanvragen, verzonnen door de course, met 1 in `yes` voor een lening die het toekent en 0 voor een die het afwijst. De loop zet het aandeel ja van elke groep in een dictionary, onder de naam van de groep, zoals `rates[group] = ...` doet. Voer de cel uit.
    """)
    return


@app.cell
def example_loans():
    import pandas as pd

    # 1. wat het uitdeelt: een ja of een nee voor een lening
    _groups = ["A"] * 12 + ["B"] * 8
    _yes = [1] * 9 + [0] * 3 + [1] * 3 + [0] * 5
    _loans = pd.DataFrame({"group": _groups, "yes": _yes})
    # 2. per groep het aantal keer ja gedeeld door de mensen: het gemiddelde van `yes`
    _rates = {}
    for _group in ["A", "B"]:
        _rates[_group] = _loans[_loans["group"] == _group]["yes"].mean()
        print(_group, _rates[_group])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Groep A krijgt een ja voor 9 van haar 12 aanvragen, 0,75; groep B voor 3 van haar 8, 0,375: half zo vaak. Stap 5 is een zin naast de score: "zegt ja tegen 75% van de aanvragen van groep A en tegen 37,5% van die van groep B". Stap 3 en 4 staan niet in deze code, want een lening wordt uitgedeeld, niet getoond. Oefening 1 doet ze wel.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shares():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Wie staat er op de afbeeldingen?

        Een beeldgenerator kreeg per beroep tien keer de vraag om een afbeelding: van een verpleegkundige (nurse), een ingenieur (engineer), een leraar (teacher) en een CEO. Iemand bekeek de 40 afbeeldingen en schreef bij elke afbeelding op of er een vrouw op staat: 1 als dat zo is, 0 als dat niet zo is. Of er een vrouw op staat, zie je niet altijd aan een afbeelding: dat oordeel is zelf al een grof label, zoals de vraag uit de vorige les over wie de labels gaf al waarschuwt. De afbeeldingen, en de aandelen vrouwen in elk beroep waar de check ze mee vergelijkt, zijn verzonnen voor deze oefening. Vervang de `...` door één regel, zoals in het uitgewerkte voorbeeld. De laatste regel is `shares`, zodat de check het ziet. De check probeert ook andere afbeeldingen.
        """),
        L.task("Zet in `shares` het aandeel vrouwen op de afbeeldingen van elk beroep, onder de naam van het beroep."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shares():
    box_shares = L.code_box("shares", lines=11)
    box_shares
    return (box_shares,)


@app.cell(hide_code=True)
def check_shares(box_shares):
    result_shares = L.check("shares", box_shares.value)
    result_shares
    return (result_shares,)


@app.cell(hide_code=True)
def hints_shares():
    L.hints(
        "shares",
        [
            "Stap 3 en 4 van het recept, zoals stap 2 in het uitgewerkte voorbeeld: per beroep het aandeel van zijn afbeeldingen dat een vrouw toont.",
            "Welk mask houdt alleen de rijen van één beroep over? En hoe zet je een waarde in een dictionary onder een key?",
            "`shares[job] = images[___][\"woman\"].___()`.",
        ],
        why="`images[images[\"job\"] == job]` houdt de tien afbeeldingen van één beroep over, en het gemiddelde van hun `woman` is het deel dat een vrouw toont: 10 van de 10 voor nurse, 1 voor engineer, 9 voor teacher en 1 voor CEO. In elk beroep zijn de afbeeldingen eenzijdiger dan de verzonnen beroepsbevolking, in de richting van het stereotype.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Oefening 2. Welke soort schade?

    Zes gevallen. Zet elk geval bij de soort schade die het doet.
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
            "Vraag bij elk geval: krijgt iemand hier iets, of juist niet? Of gaat het om hoe een groep wordt getoond of beschreven?",
            "Drie gevallen beslissen iets over een persoon. Welke drie beslissen over niemand iets?",
            "Allocatief: de lening, \\_\\_\\_ en de vacature. Representatief: \\_\\_\\_, het aanvullen en \\_\\_\\_.",
        ],
        why="**Allocatief:** de lening, de extra controles en de vacature. Elk wordt aan sommige mensen gegeven en aan anderen niet, dus je kunt per groep tellen wie het krijgt. **Representatief:** de vertaling, het aanvullen en de CEO's. Die beslissen over niemand iets, maar laten een groep zien als minder, of als stereotype.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_look_for():
    mo.md(r"""
    ### Oefening 3. Ernaar zoeken
    """)
    return


@app.cell(hide_code=True)
def choice_look_for_cell():
    choice_look_for = L.choice(
        "look-for",
        question="Een bedrijf bouwt een chatbot die korte verhalen voor kinderen schrijft. Hoe zoek je naar representatieve schade voordat hij live gaat?",
        options={
            "complaints": "Wachten op klachten van ouders als hij live is, en herstellen wat zij vinden",
            "right": "Hem om veel verhalen vragen en tellen wie hoe wordt beschreven",
            "score": "Zijn accuracy meten per groep lezers",
        },
        correct="right",
        why={
            "complaints": "Dan hebben de verhalen de kinderen al bereikt, en schade waar niemand naar zoekt, wordt zelden gemeld. Wat kun je doen voordat hij live gaat? Zie het recept.",
            "right": "Stap 3 en 4 van het recept: vraag om veel verhalen, schrijf op wie in welke rol voorkomt, zoals de held, de dokter of de schurk, en tel per groep, zoals in oefening 1.",
            "score": "Een verhaal heeft geen goed antwoord om het mee te vergelijken, en een stereotype is geen fout antwoord. Wat zou je tellen? Zie *Het idee*.",
        },
    )
    choice_look_for
    return (choice_look_for,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Wie staat er in de data, en wie ontbreekt?](/course/open/rai-bias-in-data?lang=nl) mat je een score per groep. Hier meet dezelfde telling per groep allocatieve schade, en is tellen wat een model laat zien de manier om naar representatieve schade te zoeken. In de volgende les, *Ethische theorieën toegepast op AI*, leer je manieren om te beoordelen of zo'n schade verkeerd is, en waarom.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_look_for, result_kinds, result_shares):
    L.wrap_up(result_shares, result_kinds, choice_look_for)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
