# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-human-oversight. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-human-oversight", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Wie is verantwoordelijk?](/course/open/rai-accountability?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_crumple():
    quick_crumple = L.choice(
        "quick-crumple",
        question="De veiligheidsbestuurder van een testauto krijgt de schuld van een ongeluk dat het ontwerp van de auto waarschijnlijk maakte. Hoe heet dat?",
        options={
            "hands": "Het probleem van de vele handen",
            "right": "Een morele kreukelzone",
            "gap": "Een responsibility gap: niemand kon het voorzien",
        },
        correct="right",
        why={
            "hands": "Bij vele handen verspreidt de schuld zich tot niemand hem draagt. Hier komt hij bij één persoon terecht. Zie *Het idee* in [Wie is verantwoordelijk?](/course/open/rai-accountability?lang=nl).",
            "right": "De persoon die het dichtst bij de machine staat, vangt de schuld op, zoals de kreukelzone van een auto een botsing opvangt.",
            "gap": "Hier krijgt wel iemand de schuld: de bestuurder. Zie *Het idee* in [Wie is verantwoordelijk?](/course/open/rai-accountability?lang=nl).",
        },
    )
    quick_crumple
    return (quick_crumple,)


@app.cell(hide_code=True)
def quick_check_overlap():
    quick_overlap = L.choice(
        "quick-overlap",
        question="Twee teams zeggen allebei dat ze instaan voor het testen van het model. Waarom is dat een probleem?",
        options={
            "right": "Elk team rekent er misschien op dat het andere het doet",
            "slow": "Twee keer testen duurt twee keer zo lang, en dat vertraagt de lancering",
            "none": "Dat is het niet: twee namen zijn dubbel zo veilig",
        },
        correct="right",
        why={
            "right": "Overlap kan een gat laten dat net zo groot is als geen naam: geef de stap één eigenaar.",
            "slow": "Het probleem is niet de tijd, maar dat niemand test. Zie het blok *Let op* in [Wie is verantwoordelijk?](/course/open/rai-accountability?lang=nl).",
            "none": "Twee namen zijn niet dubbel zo veilig. Zie het blok *Let op* in [Wie is verantwoordelijk?](/course/open/rai-accountability?lang=nl).",
        },
    )
    quick_overlap
    return (quick_overlap,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een veelgehoord antwoord op *wie is verantwoordelijk?* is: een mens controleert wat het systeem beslist. De Europese wet vraagt erom, en een bedrijf dat AI gebruikt, zegt het vaak om mensen gerust te stellen. Maar wie elke dag honderden voorstellen van een systeem nakijkt en ze bijna altijd goed vindt, kijkt al snel niet meer echt. Dan is de controle een handtekening, en is die persoon de kreukelzone uit de vorige les. Deze les laat zien wanneer een mens in de loop helpt, en hoe je dat nagaat.

    **Een voorbeeld uit het dagelijks leven.** Een bestuurder volgt de navigatie een afgesloten weg in, langs een bord dat dat zegt: het scherm had het al honderd keer goed.
    """)
    return


@app.cell(hide_code=True)
def predict_missed():
    guess_missed = L.guess(
        "missed",
        question="Studenten vlogen in een simulator met een hulpsysteem dat meestal gelijk had. Als het systeem zweeg over iets wat hun eigen instrumenten lieten zien, hoe vaak denk je dat ze het misten?",
        options={
            "few": "Ongeveer 5 procent van de keren",
            "right": "Ongeveer 40 procent",
            "most": "Ongeveer 80 procent",
        },
        reveal="**Ongeveer 40 procent**: 41 procent, tegen 3 procent bij studenten die zonder het hulpsysteem vlogen. En als het systeem een verkeerde instructie gaf die hun instrumenten tegenspraken, lag het foutpercentage rond de 65 procent (Skitka, Mosier en Burdick, 1999).",
    )
    guess_missed
    return (guess_missed,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Menselijk toezicht** betekent dat een mens in de gaten houdt wat een AI-systeem doet en kan ingrijpen. De Europese privacywet, de AVG, geeft mensen het recht om niet onderworpen te worden aan een besluit over hen dat volledig automatisch genomen is en rechtsgevolgen of andere ingrijpende gevolgen heeft. Waar zo'n besluit toch mag, hebben mensen ten minste het recht dat een mens ernaar kijkt, om hun standpunt te geven en om het aan te vechten. De AI Act vraagt dat systemen met een hoog risico zo gebouwd worden dat mensen er toezicht op kunnen houden: begrijpen wat het systeem wel en niet kan, zich ervan bewust blijven dat ze er te veel op kunnen vertrouwen, de output goed lezen, besluiten het niet te gebruiken of de output te negeren, en het stoppen.

        Het tweede punt heeft een naam. **Automation bias** is meer op het voorstel van een systeem vertrouwen dan het bewijs toelaat. Het komt op twee manieren voor: een probleem missen omdat het systeem er niet op wees, en een verkeerd voorstel volgen terwijl andere informatie zegt dat het niet klopt.

        Toezicht werkt alleen als de persoon drie dingen heeft: *tijd* om naar elk geval te kijken, *informatie* over het geval zelf en niet alleen de score van het systeem, en de *macht* om nee te zeggen zonder daarvoor te boeten. En het werkt alleen als je het controleert. Hoe vaak iemand het met het systeem eens is, zegt weinig; wat telt is hoeveel fouten van het systeem die persoon vangt. In 2022 las de onderzoeker Ben Green 41 regelingen die menselijk toezicht op algoritmes van de overheid eisen. Hij vond dat mensen vaak niet kunnen wat die regelingen van hen verwachten, terwijl het toezicht het systeem wel veilig laat lijken. Hij pleit ervoor het hele systeem en het gebruik ervan te toetsen, niet alleen een mens toe te voegen.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **menselijk toezicht** | een mens die een AI-systeem in de gaten houdt en kan ingrijpen: de output begrijpen, een besluit terugdraaien of het systeem stoppen |
        | **automation bias** | meer op het voorstel van een geautomatiseerd systeem vertrouwen dan het bewijs toelaat: missen wat het niet aanwees, en de fouten ervan volgen |
        """),
        L.watch_out("Een beoordelaar die het 9 van de 10 keer met het systeem eens is, bewijst niet dat het systeem goed is, of dat de controle werkt. Kijk naar de gevallen waarin het systeem het mis had: wat deed de beoordelaar daar?"),
        mo.Html(C.picture({"title": "Fouten van het model, wijzigingen van de beoordelaar",
                           "wrong": "model fout", "changed": "beoordelaar wijzigde", "xlabel": "aanvraag",
                           "alt": "Twintig leningaanvragen op een rij. Het model had het mis bij aanvraag 4, 7, 11 en 16. De beoordelaar veranderde de beslissing van het model bij aanvraag 11 en 14: maar één van de vier fouten werd rechtgezet, en de wijziging bij 14 maakte een goede beslissing fout."})),
        mo.md(r"""
        Een bank die de course verzon. Een gevulde stip bovenaan betekent dat het voorstel van het model fout was, zoals een audit later vond; een gevulde stip onderaan dat de beoordelaar het veranderde.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo zet je een mens in de loop:

    1. Schrijf op welke beslissingen een mens controleert, en wat die persoon kan doen: bevestigen, veranderen of stoppen.
    2. Geef de persoon tijd, het geval zelf en niet alleen een score, en de macht om nee te zeggen zonder daarvoor te boeten.
    3. Tel hoe vaak de persoon het met het systeem eens is.
    4. Tel op een steekproef waarvan je het juiste antwoord weet de fouten van het systeem, en hoeveel daarvan de persoon rechtzette.
    5. Vangt de persoon weinig fouten, verander dan de opzet, niet alleen de persoon.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Vijf aanvragen, verzonnen door de course: 1 is ja, 0 is nee. Stap 3 van het recept: tel hoe vaak de beoordelaar het met het model eens was. Voer de cel uit.
    """)
    return


@app.cell
def example_agreed():
    _model = [1, 0, 1, 1, 0]
    _person = [1, 0, 1, 1, 1]

    _agreed = 0
    for _i in range(len(_model)):
        if _person[_i] == _model[_i]:
            _agreed = _agreed + 1
    print(_agreed, "van de", len(_model))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De beoordelaar was het 4 van de 5 keer eens. Dat zegt hoe vaak de beoordelaar ja zei tegen het model, niet of een van beiden gelijk had. Daarvoor heb je de juiste antwoorden nodig, uit een audit, en stap 4 van het recept: oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_caught():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Welke fouten werden gevangen?

        De bank van het plaatje. `model` bevat de voorstellen van het model, `person` de beslissingen van de beoordelaar en `right` wat de audit vond, met 1 voor ja en 0 voor nee. De loop telt de fouten van het model al in `mistakes`. Vervang de `...` door twee regels: een `if` over de beslissing van de beoordelaar, en daaronder een regel die 1 optelt bij `caught`. De laatste regel is `caught`, zodat de check het ziet. De check probeert ook een andere beoordelaar.
        """),
        L.task("Laat `caught` de fouten van het model tellen die de beoordelaar rechtzette."),
    ])
    return


@app.cell(hide_code=True)
def exercise_caught():
    box_caught = L.code_box("caught", lines=12)
    box_caught
    return (box_caught,)


@app.cell(hide_code=True)
def check_caught(box_caught):
    result_caught = L.check("caught", box_caught.value)
    result_caught
    return (result_caught,)


@app.cell(hide_code=True)
def hints_caught():
    L.hints(
        "caught",
        [
            "Je zit al in de `if` voor de fouten van het model. Daar vraagt nog een `if` iets over de beoordelaar.",
            "Welke twee lijsten vergelijk je om te weten dat de beoordelaar het goed had? En wat gebeurt er dan met `caught`?",
            "`if person[i] == ___:` en daaronder `caught = caught + 1`",
        ],
        why="`if person[i] == right[i]:` met `caught = caught + 1` eronder geeft `1`: van de 4 fouten van het model zette de beoordelaar alleen die bij aanvraag 11 recht.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_missing():
    mo.md(r"""
    ### Oefening 2. Wat mist de beoordelaar?

    Zes opzetten, verzonnen door de course, waarin een mens een systeem controleert. Zet elke opzet bij wat die persoon mist.
    """)
    return


@app.cell(hide_code=True)
def sort_missing_cell():
    sort_missing = L.sort("missing")
    sort_missing
    return (sort_missing,)


@app.cell(hide_code=True)
def check_missing(sort_missing):
    result_missing = L.check("missing", sort_missing)
    result_missing
    return (result_missing,)


@app.cell(hide_code=True)
def hints_missing():
    L.hints(
        "missing",
        [
            "Vraag bij elke opzet: kan de beoordelaar goed kijken, kan de beoordelaar een fout zien, en verandert het iets als die er een vindt?",
            "Twee opzetten gaan over snelheid. In twee andere kan de beoordelaar iets fout zien, maar er niets mee doen, of verliest die erbij als die het wel doet.",
            "Tijd: een minuut per stuk en \\_\\_\\_. Informatie: alleen een score en \\_\\_\\_. Macht: \\_\\_\\_ en \\_\\_\\_.",
        ],
        why="**Te weinig tijd:** een minuut per stuk en om de paar seconden. **Te weinig informatie:** alleen een score en geen reden. **Geen macht om nee te zeggen:** een bonus voor instemmen en al verstuurd.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_audit():
    mo.md(r"""
    ### Oefening 3. Wat je eerst vraagt
    """)
    return


@app.cell(hide_code=True)
def choice_audit_cell():
    choice_audit = L.choice(
        "audit",
        question="Een gemeente zegt dat haar fraudemodel veilig is, omdat een mens elk geval controleert dat het aanwijst. Wat vraag je eerst?",
        options={
            "count": "Hoeveel mensen de signalen controleren, en hoeveel signalen elk van hen per dag afhandelt",
            "law": "Of elke beoordelaar een cursus over de wet heeft gehad",
            "right": "Hoe vaak een beoordelaar een signaal terugdraait, en of terecht",
        },
        correct="right",
        why={
            "count": "Dat zegt iets over tijd, een van de drie voorwaarden, maar niet of de controle werkt. Zie *Het idee*.",
            "law": "Een cursus kan helpen, maar laat niet zien dat beoordelaars de fouten van het model vangen. Zie *Het idee*.",
            "right": "Alleen wat de beoordelaars doen als het model het mis heeft, laat zien of de controle werkt; een controle die nooit iets terugdraait, is een handtekening.",
        },
    )
    choice_audit
    return (choice_audit,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Wie is verantwoordelijk?](/course/open/rai-accountability?lang=nl) zag je dat de schuld vaak terechtkomt bij de persoon die het dichtst bij de machine staat. Hier zag je waarom die persoon vaak weinig kan doen: automation bias, en een opzet zonder tijd, informatie of macht. In de volgende les, *Automatisch over mensen beslissen*, vraag je wanneer een besluit over mensen überhaupt geautomatiseerd mag worden, en wat iemand nodig heeft om bezwaar te kunnen maken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_audit, result_caught, result_missing):
    L.wrap_up(result_caught, result_missing, choice_audit)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
