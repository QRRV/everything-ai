# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les data-annotation. Route: course/route/lessen/klassiek-ml.yaml. Checks en plaatjes: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("data-annotation", "nl")
    C = L.checks   # 20 berichten, de labels van de annotatoren A, B en C die de course verzon, en de plaatjes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Eén vraag over [Waar labels vandaan komen](/course/open/data-labels?lang=nl) en één over [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_labels():
    quick_labels = L.choice(
        "quick-labels",
        question="Welke hiervan geeft een annotatielabel?",
        options={
            "click": "Of een lezer op een kop klikte",
            "right": "Iemand die elk bericht leest en aangeeft of het spam is",
            "sensor": "Een sensor die vastlegt wanneer een machine uitvalt",
        },
        correct="right",
        why={
            "click": "Een klik ontstaat gratis uit wat mensen doen: een natural label. Zie *Het idee* in [Waar labels vandaan komen](/course/open/data-labels?lang=nl).",
            "sensor": "Een sensor geeft een label uit een meting. Zie *Het idee* in [Waar labels vandaan komen](/course/open/data-labels?lang=nl).",
            "right": "Goed: een mens beoordeelt elk voorbeeld. Vandaag gaat het erover hoe je zulke oordelen eensgezind krijgt.",
        },
    )
    quick_labels
    return (quick_labels,)


@app.cell(hide_code=True)
def quick_check_prob():
    quick_prob = L.choice(
        "quick-prob",
        question="Annotator A noemt 13 van de 20 berichten spam. Je kiest willekeurig een van de 20. Hoe groot is de kans dat A het spam noemde?",
        options={
            "count": "13",
            "ok": "0,35",
            "right": "0,65",
        },
        correct="right",
        why={
            "count": "13 is hoeveel, niet hoe waarschijnlijk: een kans is een getal van 0 tot 1. Zie *Het idee* in [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl).",
            "ok": "0,35 is 7 / 20, de kans dat A het ok noemde. Zie *Het idee* in [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl).",
            "right": "Goed: 13 / 20 = 0,65, hoe vaak A spam zegt. Dat getal heb je vandaag nodig.",
        },
    )
    quick_prob
    return (quick_prob,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een team labelt berichten voor een spamfilter. Een model kan niet meer leren dan de labels zeggen: als twee zorgvuldige mensen een bericht verschillende labels zouden geven, zegt het label meer over wie het toevallig labelde dan over het bericht. Voordat je labels van mensen vertrouwt, wil je weten hoe goed ze het eens zijn, en hoe je ze het vaker eens krijgt.

    **Een voorbeeld uit het dagelijks leven.** Twee docenten beoordelen dezelfde opstellen. Zonder beoordelingsschema trekt de een punten af voor spelling en de ander niet, en hun cijfers verschillen. Met een schema en een paar beoordeelde voorbeeldopstellen zijn ze het veel vaker eens. En het erover eens zijn dat een leeg blad het laagste cijfer krijgt, zegt weinig: dat zou iedereen doen.
    """)
    return


@app.cell(hide_code=True)
def predict_chance_cell():
    guess_chance = L.guess(
        "chance-guess",
        question="Annotatoren A en B zijn het eens over 14 van de 20 berichten, 70%. A noemt 13 berichten spam, B 9. Als ze allebei hun labels willekeurig kozen, A nog steeds 13 keer spam en B 9 keer, hoe vaak zouden ze het dan eens zijn?",
        options={
            "low": "Bijna nooit, ongeveer 10%",
            "right": "Ongeveer de helft van de keren",
            "same": "Ongeveer 70%, even vaak als nu",
        },
        reveal="**Ongeveer de helft van de keren: 48,5%.** Allebei zeggen ze door toeval spam bij $0{,}65 \\cdot 0{,}45 = 0{,}2925$ van de berichten, en allebei ok bij $0{,}35 \\cdot 0{,}55 = 0{,}1925$: samen 0,485. Een groot deel van de 70% zou ook door toeval gebeuren.",
    )
    guess_chance
    return (guess_chance,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Richtlijnen.** Een **annotatierichtlijn** vertelt annotatoren wat elk label betekent, hoe ze in lastige gevallen kiezen, met voorbeelden van elk label, en wat ze doen als ze twijfelen. Hier zijn er twee voor dezelfde taak.

        > **Richtlijnen 1.** Spam is een bericht dat je niet wilt.
        >
        > **Richtlijnen 2.** Spam is een bericht van een afzender die jou niet kent, dat een prijs, geld of iets gratis aanbiedt, of je vraagt te betalen, te klikken, te bellen of te sms'en. Al het andere is ok, ook een advertentie die je alleen iets vertelt, een bericht van een school of een winkel waar je mee te maken hebt, en een vriend die vertelt dat die iets won. Voorbeelden: "word lid van onze club en krijg je eerste maand gratis" is spam; "er is een nieuw café in je straat geopend" is ok; "we hebben gratis concertkaartjes gewonnen sms me nu" van een vriend is ok.

        De course nam 20 berichten: 12 duidelijke uit haar eigen berichtenset, 6 spam en 6 ok, en 8 lastige, een uit de set en 7 die voor deze les geschreven zijn. De berichten zijn Engels; hieronder staan ze in het Nederlands. Daarna verzon de course twee annotatoren, A en B, die alle 20 onder elke richtlijn labelen. Over de 12 duidelijke berichten zijn ze het altijd eens. Over de lastige:

        | Bericht | Richtlijnen 1: A | B | Richtlijnen 2: A | B |
        |---|---|---|---|---|
        | we hebben gratis concertkaartjes gewonnen sms me nu | spam | ok | ok | ok |
        | de schoenwinkel heeft dit weekend uitverkoop alle laarzen voor de halve prijs | spam | ok | ok | ok |
        | word vandaag lid van onze club en krijg je eerste maand gratis | spam | spam | spam | spam |
        | het geld voor het schoolreisje moet vrijdag binnen zijn betaal op het kantoor | ok | spam | ok | ok |
        | beoordeel je laatste bestelling en krijg tien procent korting op de volgende | spam | ok | spam | ok |
        | stem op onze band in de wedstrijd sms ja en win een gesigneerd album | spam | spam | spam | spam |
        | er is een nieuw café in je straat geopend kom het eens proberen | spam | ok | ok | ok |
        | stuur dit door naar tien vrienden voor geluk | spam | ok | ok | ok |

        **Overeenstemming.** Geef dezelfde voorbeelden aan twee annotatoren, elk voor zich, en tel hoe vaak ze hetzelfde label geven: de eenvoudigste maat voor hun **inter-annotator agreement**, geschreven als $p_o$. Waar ze verschillen, is de richtlijn vaak onduidelijk, en dat zegt meer over de richtlijn dan over wie gelijk heeft.

        **Het eens zijn door toeval.** Twee mensen die geen enkel bericht lezen, zijn het toch af en toe eens. Stel dat A 65% van de berichten spam noemt en B 45%, en dat elk kiest zonder naar de ander te kijken. Van de 65% die A spam noemt, noemt B er ook 45% spam: allebei zeggen ze spam bij $0{,}65 \cdot 0{,}45 = 0{,}2925$ van de berichten. In aantallen: van de 13 berichten die A spam noemt, noemt B er 45% spam, $13 \cdot 0{,}45 = 5{,}85$, en $5{,}85 / 20 = 0{,}2925$. Allebei ok bij $0{,}35 \cdot 0{,}55 = 0{,}1925$. Samen zijn ze het door toeval eens bij $p_e = 0{,}485$ van de berichten.

        **Kappa.** **Cohen's kappa**, $\kappa$, genoemd naar Jacob Cohen (1960), zegt welk deel van de ruimte boven toeval twee annotatoren gebruiken:

        $$\kappa = \frac{p_o - p_e}{1 - p_e}$$

        Kappa is 1 als ze het altijd eens zijn, 0 als ze het niet vaker eens zijn dan door toeval, en onder 0 als ze het minder vaak eens zijn. Er is geen vaste grens voor goed genoeg: vergelijk richtlijnen op dezelfde berichten, en lees de berichten waarover mensen verschillen.

        **Gold questions.** Een **gold question** is een voorbeeld waarvan het label vaststaat, tussen de andere gemengd zonder te zeggen welk. Een annotator die de gold questions fout heeft, werkt slordig of begreep de richtlijn anders; een gold question die velen fout hebben, wijst op de richtlijn, of op het gold label zelf.

        **Crowdsourcing en de mensen die het doen.** Grote datasets worden vaak gelabeld via **crowdsourcing**: een groot karwei opgeknipt in veel kleine taken, zoals één bericht labelen, die veel mensen via internet doen, vaak betaald per taak. Per uur kan dat op minder dan een minimumloon uitkomen, en sommige taken laten gewelddadige of haatdragende teksten en beelden zien. Betalen voor de echte tijd die een taak kost, het recht om over te slaan wat iemand niet wil zien, en steun horen bij het proces, net als toestemming bij data in [Waar komt deze data vandaan?](/course/open/data-provenance?lang=nl).

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **annotatierichtlijn** | wat elk label betekent, hoe je in lastige gevallen kiest, met voorbeelden, en wat je doet bij twijfel | |
        | **inter-annotator agreement**, $p_o$ | hoe vaak annotatoren dezelfde voorbeelden hetzelfde label geven | `same / len(a)` |
        | $p_e$ | hoe vaak ze het door toeval eens zouden zijn, gegeven hoe vaak elk elk label geeft | `chance` |
        | **Cohen's kappa**, $\kappa$ | welk deel van de ruimte boven toeval twee annotatoren gebruiken: 1 altijd hetzelfde, 0 niet meer dan toeval | `(agree - chance) / (1 - chance)` |
        | **gold question** | een voorbeeld waarvan het label vaststaat, ertussen gemengd om annotatoren te controleren | |
        | **crowdsourcing** | een groot karwei opgeknipt in kleine taken die veel mensen via internet doen, vaak betaald per taak | |
        """),
        L.watch_out("Een hoge overeenstemming is op zich nog geen goede overeenstemming. Als bijna alle berichten ok zijn, is iemand die altijd ok zegt het met een zorgvuldige annotator over bijna alle berichten eens. Kappa haalt het toeval eruit."),
        mo.md(r"""
        De slider gaat door drie vergelijkingen: A en B onder richtlijnen 1, A en B onder richtlijnen 2, en A met een derde annotator, C, die overal ok zegt. Links de labels van allebei, een kolom per bericht, gevuld voor spam en open voor ok, in een grijze band waar ze verschillen; de duidelijke berichten links van de gestreepte lijn, de lastige rechts ervan. Rechts hoe vaak ze het eens zijn, hoe vaak dat door toeval zou zijn, en kappa. Beweeg de slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_case_cell():
    def _frame(case):
        said = C.case_said(case, L.lang)
        alt = ("Links twee rijen van 20 stippen, de labels van twee annotatoren, gevuld voor spam en open voor ok, met een grijze band "
               "waar ze verschillen. Rechts drie staven: hoe vaak ze het eens zijn, hoe vaak dat door toeval zou zijn, en kappa. " + said)
        return (C.case_picture(case, C.words(L.lang) | {"alt": alt}, L.lang), said)

    explore_case = L.explore("case", "vergelijking", _frame, start=1, values=[1, 2, 3], role="",
                             task="Ga door de drie vergelijkingen. Over welke berichten verschillen A en B onder richtlijnen 1, en wat is de kappa van C?")
    explore_case
    return


@app.cell(hide_code=True)
def text_after_case():
    mo.md(r"""
    Onder richtlijnen 1 verschillen A en B over 6 van de 8 lastige berichten: A noemt alles wat iets verkoopt of opdringt spam, zelfs het bericht van een vriend en een kettingbrief; B vooral wat om geld vraagt, zelfs dat van de school. Ze zijn het over 70% eens, maar 48,5% zou door toeval gebeuren, en kappa is 0,42. Richtlijnen 2 lost 5 van de 6 op met een regel en voorbeelden; alleen "tien procent korting" blijft een twijfelgeval: biedt een winkel waar je bestelde met korting geld aan, of is het een bericht van een winkel waar je mee te maken hebt? Een plek om de regel de volgende keer scherper te maken. De overeenstemming stijgt naar 95% en kappa naar 0,90. C is het met A over 55% van de berichten eens, over elk bericht dat A ok noemt, en toch is de kappa van C 0: overal ok zeggen voegt niets toe boven toeval. Als maar 1 op de 10 berichten spam was, zou C het over 90% eens zijn, en zou kappa nog steeds 0 zijn.

    ### Recept

    Zo krijg je labels van mensen die je kunt vertrouwen:

    1. Schrijf richtlijnen: wat elk label betekent, regels voor de lastige gevallen, voorbeelden van elk label, en wat je doet bij twijfel.
    2. Laat twee of meer annotatoren dezelfde steekproef labelen, elk voor zich.
    3. Bereken hun overeenstemming en kappa, en lees de voorbeelden waarover ze verschillen.
    4. Maak de richtlijnen scherper op die voorbeelden, en meet opnieuw op nieuwe voorbeelden.
    5. Meng gold questions erdoor, en controleer elke annotator erop.
    6. Betaal voor de echte tijd die het werk kost, laat mensen overslaan wat ze niet willen zien, en schrijf in de datasheet wie labelde, onder welke richtlijnen, met welke kappa.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De labels van A en B onder richtlijnen 1, in de volgorde van het plaatje: 6 duidelijke spam, 6 duidelijke ok, dan de 8 lastige berichten. De cel telt hoe vaak elk spam zegt, maakt van die aantallen delen, en print hoe vaak ze door toeval allebei spam en allebei ok zouden zeggen. Voer de cel uit.
    """)
    return


@app.cell
def example_chance():
    a = ["spam", "spam", "spam", "spam", "spam", "spam", "ok", "ok", "ok", "ok", "ok", "ok",
         "spam", "spam", "spam", "ok", "spam", "spam", "spam", "spam"]
    b = ["spam", "spam", "spam", "spam", "spam", "spam", "ok", "ok", "ok", "ok", "ok", "ok",
         "ok", "ok", "spam", "spam", "ok", "spam", "ok", "ok"]

    a_count = 0
    b_count = 0
    for i in range(len(a)):
        if a[i] == "spam":
            a_count = a_count + 1
        if b[i] == "spam":
            b_count = b_count + 1
    a_spam = a_count / len(a)                              # hoe vaak A spam zegt
    b_spam = b_count / len(b)
    print(a_count, b_count, a_spam, b_spam)
    print(round(a_spam * b_spam, 4), round((1 - a_spam) * (1 - b_spam), 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    A noemt 13 van de 20 berichten spam en B 9: delen van 0,65 en 0,45. Door toeval zeggen ze allebei spam bij 0,2925 van de berichten en allebei ok bij 0,1925, samen $p_e = 0{,}485$. Met de overeenstemming van 0,7 is kappa $(0{,}7 - 0{,}485) / (1 - 0{,}485) = 0{,}42$. Python print getallen met een punt: `0.65` is 0,65.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_agreement():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Hoe vaak twee annotatoren het eens zijn

        `agreement(a, b)` krijgt de labels van twee annotatoren voor dezelfde berichten, elk `"spam"` of `"ok"`, en moet het deel van de berichten geven waaraan ze hetzelfde label geven. Vervang de `...`. De check roept `agreement(["spam", "ok", "ok", "spam", "ok"], ["spam", "ok", "spam", "spam", "ok"])` aan, wat 0,8 moet geven, en nog vier paren lijsten, waaronder A en B onder richtlijnen 1.
        """),
        L.task("Laat `agreement` de berichten tellen waarover de twee het eens zijn."),
    ])
    return


@app.cell(hide_code=True)
def exercise_agreement():
    box_agreement = L.code_box("agreement", lines=7)
    box_agreement
    return (box_agreement,)


@app.cell(hide_code=True)
def check_agreement(box_agreement):
    result_agreement = L.check("agreement", box_agreement.value)
    result_agreement
    return (result_agreement,)


@app.cell(hide_code=True)
def hints_agreement():
    L.hints(
        "agreement",
        [
            "Voor elk bericht één vraag: gaven de twee het hetzelfde label?",
            "Kijk naar één bericht tegelijk: `a[i]` is het label dat A aan bericht `i` gaf, en `b[i]` het label dat B aan hetzelfde bericht gaf. Met een `if` laat je `same` alleen bij sommige berichten groeien.",
            "`if a[i] == ___:` met `same = same + 1` één stap ingesprongen.",
        ],
        why="Allebei spam telt als het eens zijn, en allebei ok ook. Onder richtlijnen 1 geven A en B 14 van de 20 berichten hetzelfde label: 14 / 20 = 0,7.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_kappa():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Cohen's kappa

        `kappa(agree, a_spam, b_spam)` krijgt de overeenstemming van twee annotatoren en hoe vaak elk van hen spam zegt, en moet Cohen's kappa geven. Vervang de twee `...`: eerst hoe vaak ze het door toeval eens zouden zijn, dan kappa. De check roept `kappa(0.7, 0.65, 0.45)` aan, de getallen van richtlijnen 1, en nog vijf.
        """),
        L.task("Laat `kappa` het toeval uit de overeenstemming halen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_kappa():
    box_kappa = L.code_box("kappa", lines=5)
    box_kappa
    return (box_kappa,)


@app.cell(hide_code=True)
def check_kappa(box_kappa):
    result_kappa = L.check("kappa", box_kappa.value)
    result_kappa
    return (result_kappa,)


@app.cell(hide_code=True)
def hints_kappa():
    L.hints(
        "kappa",
        [
            "Eerst toeval: op welke twee manieren kunnen twee annotatoren het door toeval eens zijn? Dan kappa: welk deel van de ruimte boven toeval gebruiken ze?",
            "Door toeval zeggen ze allebei spam, `a_spam` keer `b_spam`, of allebei ok, met voor elk het deel dat geen spam is. De ruimte boven toeval loopt van `chance` tot 1.",
            "`chance = a_spam * b_spam + (1 - ___) * (1 - ___)` en `return (agree - ___) / (1 - ___)`.",
        ],
        why="Met 0,7, 0,65 en 0,45 is toeval $0{,}65 \\cdot 0{,}45 + 0{,}35 \\cdot 0{,}55 = 0{,}485$, en kappa $(0{,}7 - 0{,}485) / (1 - 0{,}485) = 0{,}42$: A en B gebruiken 42% van de ruimte boven toeval.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_gold():
    mo.md(r"""
    ### Oefening 3. Wat een gold question vertelt
    """)
    return


@app.cell(hide_code=True)
def choice_gold_cell():
    choice_gold = L.choice(
        "gold",
        question="Annotator D labelt 200 berichten in tien minuten. Daartussen zitten 12 gold questions, 6 spam en 6 ok, en D heeft er 7 goed. Wat maak je daaruit op?",
        options={
            "fine": "D doet het prima: meer dan de helft is goed",
            "right": "D doet het op duidelijke berichten nauwelijks beter dan kop of munt: controleer de labels van D en vraag waarom",
            "drop": "Laat de gold questions de volgende keer weg: ze kosten D tijd",
        },
        correct="right",
        why={
            "fine": "Een munt opgooien geeft gemiddeld 6 van de 12 goed, en dit zijn duidelijke berichten. 7 van de 12 zegt dat de labels van D de berichten nauwelijks volgen.",
            "drop": "Met gold questions merk je slordig werk überhaupt op. Ze laten ook zien waar richtlijnen onduidelijk zijn: een gold question die veel mensen fout hebben, wijst op de richtlijn, of op het gold label zelf.",
            "right": "Goed. 200 berichten in tien minuten is 3 seconden per bericht. Kijk ook naar hoe het werk is opgezet: betaling per taak beloont snelheid, geen zorgvuldigheid.",
        },
    )
    choice_gold
    return (choice_gold,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Annotatielabels uit [Waar labels vandaan komen](/course/open/data-labels?lang=nl) zijn zo goed als de richtlijnen erachter, en het eens zijn door toeval bouw je uit hoe vaak elk spam zegt, uit [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl): allebei spam vermenigvuldigd, plus allebei ok vermenigvuldigd. De volgende les, *Verbeter de data, niet het model*, vindt de verkeerde labels die er toch doorheen glippen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_gold, result_agreement, result_kappa):
    L.wrap_up(result_agreement, result_kappa, choice_gold)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
