# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-bias-faces-images. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-bias-faces-images", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Bias in elke stap](/course/open/rai-bias-lifecycle?lang=nl) en [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_test():
    quick_test = L.choice(
        "quick-test",
        question="Een bedrijf traint een spraakassistent op opnames uit elke regio, maar test hem alleen op het eigen personeel, dat bijna helemaal in één regio is opgegroeid. Welke soort bias kan dat geven?",
        options={
            "representation": "Representation bias",
            "right": "Evaluation bias",
            "measurement": "Measurement bias",
        },
        correct="right",
        why={
            "representation": "De data om mee te trainen dekt elke regio, dus daar zit het probleem niet: het zit in de testdata. Zie *Het model bouwen en gebruiken* in [Bias in elke stap](/course/open/rai-bias-lifecycle?lang=nl).",
            "right": "Testdata die niet lijkt op de mensen voor wie de assistent gebruikt wordt, zodat een goede score een groep kan verbergen: evaluation bias.",
            "measurement": "Geen label is hier een vervanger die voor één groep slechter meet; het probleem is wie er in de test zit. Zie *Het model bouwen en gebruiken* in [Bias in elke stap](/course/open/rai-bias-lifecycle?lang=nl).",
        },
    )
    quick_test
    return (quick_test,)


@app.cell(hide_code=True)
def quick_check_recall():
    quick_recall = L.choice(
        "quick-recall",
        question="Een systeem krijgt foto's van 200 vrouwen. Het noemt er 180 een vrouw en 20 een man. Wat is de recall voor de klasse vrouw?",
        options={
            "right": "0,9",
            "wrong": "0,1",
            "upside": "1,11",
        },
        correct="right",
        why={
            "right": "180 / (180 + 20) = 0,9: het vindt 9 op de 10 vrouwen. In deze les is 1 min de recall van een groep het foutpercentage van die groep: hier 0,1.",
            "wrong": "Dat is het deel dat het fout had, 20 van de 200. Recall is het deel van de vrouwen dat het vindt: TP / (TP + FN). Zie [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
            "upside": "Dat is 200 / 180, omgekeerd: recall is nooit groter dan 1. Recall deelt de true positives door alle vrouwen: TP / (TP + FN). Zie [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl).",
        },
    )
    quick_recall
    return (quick_recall,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Vijf zwarte vrouwen spanden een rechtszaak aan tegen General Motors. Het bedrijf had vóór 1964 geen zwarte vrouwen aangenomen, en alle zwarte vrouwen die het na 1970 aannam, raakten hun baan kwijt toen er in een latere recessie op anciënniteit werd ontslagen. De rechter oordeelde dat ze de wet tegen discriminatie op ras en de wet tegen discriminatie op geslacht niet mochten samenvoegen tot een nieuw 'superrechtsmiddel'. In 1989 gebruikte de rechtsgeleerde Kimberlé Crenshaw de zaak om te laten zien hoe een kader dat langs één as kijkt, zwarte vrouwen onzichtbaar maakt: wie ras en geslacht los van elkaar bekijkt, mist wat er gebeurt waar die twee elkaar raken. Crenshaws woord daarvoor, intersectionaliteit, komt met een beeld van verkeer op een kruispunt: een ongeluk daar kan veroorzaakt worden door auto's uit elke richting, en soms door auto's uit alle richtingen tegelijk. Machine learning heeft dezelfde blinde vlek. Een systeem dat je per eigenschap controleert, kan er goed uitzien voor vrouwen en goed voor mensen met een donkerdere huid, en toch falen bij vrouwen met een donkerdere huid.

    **Een voorbeeld uit het dagelijks leven.** Een school meldt dat 90 procent van de meisjes slaagt, en 90 procent van de leerlingen uit het oosten van de stad. Daaruit weet je niet hoe het de meisjes uit het oosten vergaat: zij kunnen net degenen zijn die zakken.
    """)
    return


@app.cell(hide_code=True)
def predict_gerrymander():
    guess_gerrymander = L.guess(
        "gerrymander",
        question="Een model zegt ja tegen de helft van de mannen en de helft van de vrouwen, en tegen de helft van de zwarte en de helft van de witte mensen. De vier combinaties zijn even groot. Tegen welk deel van de zwarte vrouwen kan het ja zeggen?",
        options={
            "half": "Precies de helft",
            "between": "Tussen een kwart en driekwart",
            "right": "Van niemand tot iedereen",
        },
        reveal="**Van niemand tot iedereen.** Michael Kearns en collega's (2018) geven een voorbeeld: een model dat precies ja zegt tegen de zwarte mannen en de witte vrouwen. Het zegt ja tegen 50 procent van de mannen en van de vrouwen, en tegen 50 procent van de zwarte en van de witte mensen, dus per eigenschap lijkt het fair. Maar het zegt ja tegen geen enkele zwarte vrouw en geen enkele witte man. Ze noemen dat *fairness gerrymandering*: een model kan fair lijken voor een paar groepen die je vooraf koos, en unfair zijn voor hun combinaties.",
    )
    guess_gerrymander
    return (guess_gerrymander,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Intersectional fairness.** *Intersectional fairness* is fairness voor groepen die combinaties van eigenschappen zijn, zoals vrouwen met een donkerdere huid, en niet alleen voor elke eigenschap afzonderlijk. Je controleert het op een eenvoudige manier: tel per combinatie, en reken de maat per combinatie uit. Crenshaws paper gaat over recht, niet over statistiek; *Gender Shades* leende Crenshaws woord voor de analyse van vier combinaties, en de course doet dat ook. Hoe ver moet je gaan? Kearns en collega's waarschuwen dat je geen fairness kunt eisen voor elke subgroep die je kunt bedenken: elk model dat fouten maakt, zou je unfair kunnen noemen tegenover de groep die het verkeerd indeelde. Ze noemen dat 'overfitting' van een fairnesseis. Je kiest dus de combinaties die ertoe doen.

        **Gender Shades.** Joy Buolamwini en Timnit Gebru (2018) maakten een benchmark van 1.270 parlementsleden uit drie Afrikaanse landen (Rwanda, Senegal, Zuid-Afrika) en drie Europese (IJsland, Finland, Zweden), landen die gekozen waren omdat er in hun parlement ongeveer evenveel vrouwen als mannen zitten: de Pilot Parliaments Benchmark. Een dermatoloog gaf elke persoon een huidtype op de zespuntsschaal van Fitzpatrick; type I tot en met III telt als lichter, IV tot en met VI als donkerder. De labels voor gender zijn female en male, voor mensen die als vrouw of als man worden gezien. Het paper noemt die indeling zelf te simpel: ze doet geen recht aan hoe ingewikkeld gender is, en niet aan transgenderidentiteiten. In april en mei 2017 testten ze de genderclassifiers van Microsoft, Face++ en IBM: systemen die een gezicht het label female of male geven. Het foutpercentage van een groep is 1 min de true positive rate: voor vrouwen het deel van de vrouwen dat het systeem een man noemde, dus 1 min de recall voor vrouwen. Gezichten die het systeem helemaal niet vond, tellen niet mee: alleen dan kloppen de getallen van het paper met de aantallen, en het vervolgonderzoek noemt die regel ook.

        | Systeem | vrouwen | mannen | donkerder | lichter | vrouwen, donkerder | mannen, lichter |
        |---|---|---|---|---|---|---|
        | Microsoft | 10,7 | 2,6 | 12,9 | 0,7 | 20,8 | 0,0 |
        | Face++ | 21,3 | 0,7 | 16,5 | 4,7 | 34,5 | 0,8 |
        | IBM | 20,3 | 5,6 | 22,4 | 3,2 | 34,7 | 0,3 |

        *Foutpercentage op de Pilot Parliaments Benchmark, uit tabel 4 van* Gender Shades.

        Alle drie de systemen deden het slechtst bij vrouwen met een donkerdere huid, van 20,8 tot 34,7 procent, terwijl het hoogste foutpercentage bij mannen met een lichtere huid 0,8 procent was. Vrouwen met een donkerdere huid waren 21,3 procent van de benchmark, maar tussen 61,0 en 72,4 procent van de fouten. Kijk naar de kolommen: het slechtste getal per eigenschap, 22,4 voor mensen met een donkerdere huid bij IBM, ligt ver onder de 34,7 van de combinatie. Over de oorzaak zijn de auteurs voorzichtig: een donkerdere huid hangt misschien sterk samen met gezichtsvormen of met manieren om gender te tonen die weinig voorkwamen in de data waarop de systemen getraind waren. En ze kozen met opzet makkelijke foto's, een 'optimistische steekproef', dus op moeilijkere foto's verwachten ze hogere foutpercentages.

        **Na de audit.** Inioluwa Deborah Raji en Buolamwini testten in augustus 2018 opnieuw. Binnen zeven maanden hadden alle drie de bedrijven een nieuwe versie uitgebracht, en het foutpercentage bij vrouwen met een donkerdere huid was 17,7 tot 30,4 procentpunt gedaald; bij IBM van 34,7 naar 16,97 procent. Twee bedrijven die in de eerste audit niet genoemd werden, Amazon en Kairos, zaten op 31,4 en 22,5 procent bij vrouwen met een donkerdere huid.

        **Geen gezichtsherkenning.** *Gender Shades* testte genderclassificatie: een label raden uit één gezicht. Gezichtsherkenning zoekt foto's van dezelfde persoon bij elkaar. In 2019 testte NIST, het Amerikaanse instituut voor standaarden, 189 algoritmes voor gezichtsherkenning, de meeste commercieel, en waarschuwt dat de studies naar genderclassificatie geen gezichtsherkenning hebben getest, maar wel vaak worden aangehaald om gezichtsherkenning af te kraken. De eigen uitkomst van NIST: tussen groepen mensen verschilt het aandeel false positives, twee verschillende mensen die voor dezelfde persoon worden aangezien, vaak een factor 10 tot meer dan 100. Op pasfoto's voor aanvragen lag het het hoogst bij mensen uit West- en Oost-Afrika en Oost-Azië; bij een aantal algoritmes uit China was het voor Oost-Aziatische gezichten juist andersom.

        **Beeldgeneratoren.** Een beeldgenerator maakt een afbeelding bij een tekst. Federico Bianchi en collega's (2023) vroegen Stable Diffusion om *A photo of the face of* gevolgd door een beroep, 100 afbeeldingen per beroep, en gebruikten CLIP, een model dat afbeeldingen omzet in vectoren, om te vinden bij de gemiddelde gezichtsfoto van welke groep elke afbeelding het dichtst lag. 99 procent van de softwareontwikkelaars werd als wit weergegeven, terwijl 56 procent van de softwareontwikkelaars in de VS zichzelf wit noemde; huishoudelijk medewerkers, verpleegkundigen en cabinemedewerkers werden voor 100 procent als vrouw weergegeven. Op de afbeeldingen staat niemand: 'weergegeven als' is een label van CLIP, niet iemands identiteit. Alexandra Sasha Luccioni en collega's (2023) vergeleken drie generatoren met de cijfers van het Amerikaanse statistiekbureau voor werk, het Bureau of Labor Statistics (BLS), voor 146 beroepen. Ze verdeelden de beroepen in vijf groepen op het aandeel vrouwen, van het vijfde deel met de minste vrouwen tot het vijfde deel met de meeste, en maten het aandeel afbeeldingen in clusters die bij het woord vrouw horen. Dall·E 2 zette in elke groep veel minder afbeeldingen in clusters die bij vrouwen horen dan de statistiek aan vrouwen telde; Stable Diffusion 1.4 kwam het dichtst in de buurt, en zette er in de groep met de minste vrouwen zelfs iets meer. In deze les is 'minder vrouwen tekenen' kort voor deze maat.

        **De wet.** Sinds 2 februari 2025 verbiedt de Europese AI Act onder meer twee toepassingen waar gezichtsanalyse voor kan dienen: emoties afleiden van mensen op het werk en in het onderwijs, behalve om medische redenen of voor de veiligheid, en biometrische categorisering, uit een gezicht of andere biometrische gegevens, die iemands ras, politieke opvattingen, lidmaatschap van een vakbond, religieuze of levensbeschouwelijke overtuiging, seksleven of seksuele gerichtheid afleidt. Geslacht en gender staan niet in dat rijtje. Mensen indelen naar gevoelige of beschermde eigenschappen op basis van hun biometrische gegevens, en emotieherkenning elders, gelden in plaats daarvan als hoog risico; na een wijziging van de wet in 2026 gelden die verplichtingen vanaf 2 december 2027.

        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **intersectional fairness** | fairness voor groepen die combinaties van eigenschappen zijn, zoals vrouwen met een donkerdere huid, gecontroleerd per combinatie |
        | **fairness gerrymandering** | een model lijkt fair voor elke eigenschap afzonderlijk, en is unfair voor hun combinaties (Kearns en collega's) |
        | **foutpercentage van een groep** | in *Gender Shades*: 1 min de recall van een groep, het deel van die groep dat het systeem een verkeerd label geeft |
        """),
        L.watch_out("Een tabel per combinatie is niet alles wat intersectionaliteit betekent. Crenshaw wijst erop dat de ervaring groter is dan de som van racisme en seksisme; foutpercentages per vakje laten zien waar een systeem faalt, niet waarom, en niet hoe het is om erdoor in de steek gelaten te worden."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo controleer je een systeem op intersectionele bias:

    1. Kies de eigenschappen die ertoe doen voor de mensen bij wie het systeem gebruikt wordt, en de maat, zoals het foutpercentage.
    2. Tel per combinatie: hoeveel mensen, hoeveel er een uitkomst kregen, hoeveel fouten. Zeg wat je wegliet, zoals gezichten die het systeem niet vond.
    3. Reken de maat per combinatie uit. Voor één eigenschap tel je de aantallen van de vakjes op en deel je één keer; neem niet het gemiddelde van foutpercentages van vakjes die niet even groot zijn.
    4. Vergelijk de slechtste combinatie met de slechtste losse eigenschap, en kijk hoe groot elk vakje is: bij weinig mensen springt een percentage alle kanten op.
    5. Kies de combinaties voordat je kijkt; je kunt niet elke groep controleren die je kunt bedenken.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De getallen van Microsoft uit *Gender Shades*, per combinatie, uit de tabellen in het aanvullende materiaal bij het paper. `classified` is het aantal gezichten dat het systeem vond en een label gaf, `errors` het aantal dat het een verkeerd label gaf. De loop rekent het foutpercentage van elk vakje uit, als een deel van 0 tot 1. Voer de cel uit.
    """)
    return


@app.cell
def example_cells():
    # Microsoft in Gender Shades (2018): per groep de gezichten die het indeelde, en de fouten
    _cells = [
        {"skin": "darker", "gender": "female", "classified": 265, "errors": 55},
        {"skin": "darker", "gender": "male", "classified": 300, "errors": 18},
        {"skin": "lighter", "gender": "female", "classified": 295, "errors": 5},
        {"skin": "lighter", "gender": "male", "classified": 385, "errors": 0},
    ]

    for _cell in _cells:
        _rate = _cell["errors"] / _cell["classified"]
        print(_cell["skin"], _cell["gender"], round(_rate, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Vier foutpercentages, van 0,208 voor vrouwen met een donkerdere huid tot 0,0 voor mannen met een lichtere huid, net als in de tabel bij *Het idee*. Een rapport per eigenschap geeft één getal voor alle vrouwen: het mengt het vakje van 0,208 met het vakje van 0,017. Hoeveel verbergt dat mengen? Oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rate():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Eén getal per eigenschap

        Reken het foutpercentage uit van alle vakjes samen waarvan `key` de waarde `value` heeft, als deel tussen 0 en 1: bij `key = "gender"` en `value = "female"` zijn dat alle vrouwen. Eerst optellen, dan pas delen: begin met `classified = 0` en `errors = 0`, loop met `cell` over `cells`, en tel bij `cell[key] == value` de `"classified"` en de `"errors"` van het vakje op bij de totalen. Na de loop is `rate` gelijk aan `errors` gedeeld door `classified`; dat is de laatste regel, zodat de check het ziet. De check probeert ook de vakjes met een donkerdere huid, en de mannen.
        """),
        L.task("Reken `rate` uit: de fouten van de vakjes waarvan `key` gelijk is aan `value`, gedeeld door hun ingedeelde gezichten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rate():
    box_rate = L.code_box("rate", lines=22)
    box_rate
    return (box_rate,)


@app.cell(hide_code=True)
def check_rate(box_rate):
    result_rate = L.check("rate", box_rate.value)
    result_rate
    return (result_rate,)


@app.cell(hide_code=True)
def hints_rate():
    L.hints(
        "rate",
        [
            "Het foutpercentage van de vrouwen is alle fouten bij vrouwen gedeeld door alle ingedeelde vrouwen: tel de twee vakjes met `\"female\"` op, en deel één keer, na de loop.",
            "In de loop kiest een `if` de vakjes: `if cell[key] == value:`. Daaronder tellen twee regels de aantallen van het vakje op bij de totalen, zoals `classified = classified + cell[\"classified\"]`.",
            "Na de loop, niet erin: `rate = ___ / classified`.",
        ],
        why="`classified = 0` en `errors = 0`, dan `for cell in cells:` met daaronder `if cell[key] == value:`, en daarbinnen `classified = classified + cell[\"classified\"]` en `errors = errors + cell[\"errors\"]`. Na de loop `rate = errors / classified`. Dat geeft 60 / 560 = `0.107`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_subgroups():
    mo.md(r"""
    ### Oefening 2. Elke groep?
    """)
    return


@app.cell(hide_code=True)
def choice_subgroups_cell():
    choice_subgroups = L.choice(
        "subgroups",
        question="Een team wil zijn gezichtsclassifier meten op elke groep die het kan bedenken, tot en met 'de gezichten die hij een verkeerd label geeft'. Wat is het probleem met die laatste groep?",
        options={
            "small": "Die is te klein om iets te meten",
            "right": "Elk model dat fouten maakt, faalt erbij",
            "more": "Geen: meer groepen maken een model altijd fairder",
        },
        correct="right",
        why={
            "small": "De grootte is het probleem niet: de groep kan groot zijn. Bij die groep heeft het model per definitie ongelijk, dus elk model dat fouten maakt, faalt erbij. Zie *Intersectional fairness* bij *Het idee*.",
            "right": "Kearns en collega's: elke classifier die fouten maakt, kun je ervan beschuldigen dat hij unfair is tegenover de groep die hij verkeerd indeelde, en ze noemen dat 'overfitting' van een fairnesseis. Kies de combinaties die ertoe doen, zoals huidtype en gender, voordat je kijkt.",
            "more": "Meer groepen meten maakt een model niet fairder, en een groep die uit de eigen fouten van het model bestaat, laat elk model dat fouten maakt falen. Zie *Intersectional fairness* bij *Het idee*.",
        },
    )
    choice_subgroups
    return (choice_subgroups,)


@app.cell(hide_code=True)
def text_exercise_gap():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Minder vrouwen dan in de statistiek

        `quintiles` bevat tabel 3 van Luccioni en collega's: per groep beroepen is `"bls"` het percentage vrouwen in de arbeidsstatistiek, en zijn `"sd14"`, `"sd2"` en `"dalle2"` het percentage afbeeldingen dat Stable Diffusion 1.4, Stable Diffusion 2 en Dall·E 2 tekenden in clusters die bij vrouwen horen. Reken `gap` uit: het gemiddelde over de vijf groepen van het aandeel van het systeem min het aandeel in de statistiek. Begin met `total = 0`, loop met `quintile` over `quintiles`, tel `quintile[system] - quintile["bls"]` op bij `total`, en deel door het aantal groepen, `len(quintiles)`. Een negatief verschil betekent dat het systeem minder vrouwen tekent dan er in die beroepen werken. De check probeert ook de twee andere systemen.
        """),
        L.task("Reken `gap` uit: het gemiddelde van `quintile[system] - quintile[\"bls\"]` over de groepen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_gap():
    box_gap = L.code_box("gap", lines=18)
    box_gap
    return (box_gap,)


@app.cell(hide_code=True)
def check_gap(box_gap):
    result_gap = L.check("gap", box_gap.value)
    result_gap
    return (result_gap,)


@app.cell(hide_code=True)
def hints_gap():
    L.hints(
        "gap",
        [
            "Net als in oefening 1: optellen in de loop, delen erna. Deze keer tel je per groep één verschil op.",
            "Het verschil van één groep is `quintile[system] - quintile[\"bls\"]`: het aandeel van het systeem min dat van de statistiek. Deel na de loop het totaal door `len(quintiles)`.",
            "In de loop: `total = total + ___`.",
        ],
        why="`total = 0`, dan `for quintile in quintiles:` met daaronder `total = total + quintile[system] - quintile[\"bls\"]`. Na de loop `gap = total / len(quintiles)`. Dat geeft `-29.64` voor Dall·E 2.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_law():
    mo.md(r"""
    ### Oefening 4. Wat verbiedt de wet?
    """)
    return


@app.cell(hide_code=True)
def choice_law_cell():
    choice_law = L.choice(
        "law",
        question="Welke van deze verbiedt de AI Act in de EU?",
        options={
            "gender": "Een classifier die uit een gezichtsfoto het gender raadt",
            "right": "Emotieherkenning bij leerlingen in de klas",
            "nurses": "Een beeldgenerator die elke verpleegkundige als vrouw tekent",
        },
        correct="right",
        why={
            "gender": "Het verbod op biometrische categorisering noemt ras, politieke opvattingen, lidmaatschap van een vakbond, religieuze of levensbeschouwelijke overtuiging, seksleven en seksuele gerichtheid, maar niet geslacht of gender. Zie *De wet* bij *Het idee*.",
            "right": "Sinds 2 februari 2025 verbiedt de AI Act systemen die de emoties afleiden van mensen op het werk en in het onderwijs, behalve om medische redenen of voor de veiligheid.",
            "nurses": "Geen van de verboden in de AI Act gaat over wie een beeldgenerator tekent bij een beroep. Een scheef beeld van wie een beroep uitoefent, is schade die je meet, zoals in oefening 3. Zie *De wet* bij *Het idee*.",
        },
    )
    choice_law
    return (choice_law,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Bias in elke stap](/course/open/rai-bias-lifecycle?lang=nl) verborg één totaalscore een groep: evaluation bias. Hier verborg een score per eigenschap een combinatie: tel per vakje, en reken het foutpercentage van een vakje, en van een eigenschap, uit de aantallen uit. Het foutpercentage van *Gender Shades* is 1 min de recall uit [Meer dan accuracy: precision en recall](/course/open/ml-confusion-matrix?lang=nl), per groep. Bij een beeldgenerator was de vraag niet hoe vaak hij het mis heeft, maar of wie hij tekent overeenkomt met de wereld, of met een stereotype ervan. De volgende les, *Bias in spraak en in advertenties*, stelt dezelfde vraag over stemmen, en over wie een advertentie te zien krijgt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_law, choice_subgroups, result_gap, result_rate):
    L.wrap_up(result_rate, choice_subgroups, result_gap, choice_law)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
