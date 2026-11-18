# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-bias-lifecycle. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-bias-lifecycle", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Wie staat er in de data, en wie ontbreekt?](/course/open/rai-bias-in-data?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_overall():
    quick_overall = L.choice(
        "quick-overall",
        question="Een spraakapp begrijpt 75 procent van al zijn testopnames. Waarom meet je het ook per accent?",
        options={
            "higher": "Per accent is de score altijd hoger",
            "right": "De totale score kan een groep verbergen",
            "law": "De wet vraagt bij elke app een score per accent",
        },
        correct="right",
        why={
            "higher": "Per accent zijn sommige scores hoger en andere lager; de totale score mengt ze. Zie het *Uitgewerkt voorbeeld* in [Wie staat er in de data, en wie ontbreekt?](/course/open/rai-bias-in-data?lang=nl).",
            "right": "In die les begreep de app 11 van de 12 opnames met accent A, maar slechts 1 van de 3 met accent C.",
            "law": "Geen wet vraagt dat van elke app; de reden is dat één score een groep kan verbergen. Zie het *Uitgewerkt voorbeeld* in [Wie staat er in de data, en wie ontbreekt?](/course/open/rai-bias-in-data?lang=nl).",
        },
    )
    quick_overall
    return (quick_overall,)


@app.cell(hide_code=True)
def quick_check_past():
    quick_past = L.choice(
        "quick-past",
        question="Een model voor leningen leert van de leningen uit de afgelopen tien jaar, zonder fouten vastgelegd, maar de beslissingen van toen waren niet rechtvaardig. Waar komt de scheefheid vandaan?",
        options={
            "who": "Sommige groepen ontbreken in de data",
            "labels": "De mensen die de labels gaven, waren slordig",
            "right": "Het verleden dat de data vastlegt",
        },
        correct="right",
        why={
            "who": "Niets zegt dat er een groep ontbreekt: de data is compleet. Zie *Oefening 2* in [Wie staat er in de data, en wie ontbreekt?](/course/open/rai-bias-in-data?lang=nl).",
            "labels": "De leningen zijn zonder fouten vastgelegd. Zie *Oefening 2* in [Wie staat er in de data, en wie ontbreekt?](/course/open/rai-bias-in-data?lang=nl).",
            "right": "De data klopt, maar wat ze vastlegt was niet rechtvaardig, dus een model dat erop traint, leert de oude beslissingen.",
        },
    )
    quick_past
    return (quick_past,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Harini Suresh en John Guttag beginnen hun raamwerk (2021) met twee vereenvoudigde scenario's. In het eerste herkent een model hartaanvallen, en het mist ze vaker bij vrouwen. Meer data van vrouwen die een hartaanval hadden, maakt het model beter voor hen. In het tweede leert een model, met beoordelingen van eerdere sollicitanten door mensen als label, of iemand geschikt is als laborant, en het beoordeelt vrouwen lager. Hier helpt meer gelabelde data van dezelfde soort niet. De twee problemen leken hetzelfde, maar hadden een andere bron: het eerste model had te weinig voorbeelden van vrouwen, het tweede leerde van beoordelingen door mensen die scheef waren. Bias kan in elke stap binnenkomen, van de wereld tot een model in gebruik, en pas als je de stap benoemt, weet je waar de oplossing moet zitten.

    **Een voorbeeld uit het dagelijks leven.** Een thermometer hangt in de zon en geeft te veel aan. Vaker aflezen helpt niet: elke nieuwe meting heeft dezelfde fout. Je moet de thermometer verhangen. Maar als hij de koude nachten mist omdat je hem alleen om twaalf uur 's middags afleest, dan helpt vaker aflezen wel.
    """)
    return


@app.cell(hide_code=True)
def predict_shades():
    guess_shades = L.guess(
        "shades",
        question="Drie commerciële systemen die uit een foto het geslacht raden, zijn getest op 1.270 parlementsleden uit zes landen. Bij mannen met een lichtere huid was het hoogste foutpercentage 0,8 procent. Hoe hoog was het volgens jou bij vrouwen met een donkerdere huid?",
        options={
            "same": "Zo'n 1 procent, net als bij mannen met een lichtere huid",
            "few": "Tussen de 5 en 10 procent",
            "right": "Tot zo'n 35 procent",
        },
        reveal="**Tot 34,7 procent.** In *Gender Shades* (2018) vonden Joy Buolamwini en Timnit Gebru dat alle drie de systemen het slechtst deden bij vrouwen met een donkerdere huid, met foutpercentages van 20,8 tot 34,7 procent. De gangbare benchmarks hadden dat niet laten zien: vrouwen met een donkerdere huid waren maar 7,4 procent van de beelden in Adience en 4,4 procent in IJB-A. Dat is evaluation bias, een van de zeven soorten hieronder.",
    )
    guess_shades
    return (guess_shades,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Zeven plekken.** Suresh en Guttag volgen de weg van de wereld naar een model in gebruik, en noemen zeven plekken waar schade kan binnenkomen. Drie zitten in het maken van de data: historical, representation en measurement bias. Vier zitten in het bouwen en gebruiken van het model: aggregation, learning, evaluation en deployment bias. De soorten kunnen overlappen, maar als je ze elk apart benoemt, schrijven ze, zijn ze minder verwarrend en makkelijker aan te pakken. En de weg is een rondje: beslissingen die een model helpt nemen, veranderen de wereld waar de volgende data vandaan komt.
        """),
        mo.Html(C.picture({"title": "Bias in elke stap", "heads": ["de data maken", "het model bouwen en gebruiken"],
                           "steps": [("1. De wereld", ["historical bias"]), ("2. De steekproef", ["representation bias"]),
                                     ("3. Features en labels", ["measurement bias"]),
                                     ("4. Trainen", ["aggregation bias", "learning bias"]),
                                     ("5. Testen", ["evaluation bias"]), ("6. Gebruik", ["deployment bias"])],
                           "back": "beslissingen veranderen de wereld, en zo de volgende data",
                           "alt": "Zes vakken in een rondje. De data maken, aan de linkerkant omlaag: 1, de wereld, met historical bias; 2, de steekproef, met representation bias; 3, features en labels, met measurement bias. Dan opzij naar het model bouwen en gebruiken, aan de rechterkant omhoog: 4, trainen, met aggregation bias en learning bias; 5, testen, met evaluation bias; 6, gebruik, met deployment bias. Een gestippelde pijl loopt van 6 terug naar 1: beslissingen veranderen de wereld, en zo de volgende data."})),
        mo.md(r"""
        **De data maken.** *Historical bias* ontstaat zelfs als de data perfect gemeten en getrokken is, als de wereld zoals die is of was tot een model leidt dat schade doet. Woordvectoren die geleerd zijn uit Google News, drie miljoen woorden en woordgroepen, vulden *man staat tot programmeur als vrouw staat tot x* aan met x = huisvrouw (Bolukbasi en anderen, 2016). Het nieuws legde de wereld getrouw vast; de wereld was het probleem. *Representation bias* betekent dat de data te weinig voorbeelden heeft van een deel van de mensen voor wie het model gebruikt wordt. In een steekproef van de foto's met een locatie in de beeldverzameling ImageNet kwam zo'n 45 procent uit de VS, 1 procent uit China en 2,1 procent uit India (Shankar en anderen, 2017). Het gebeurt als de data van andere mensen komt dan die voor wie het model gebruikt wordt, als de steekproef ongelijk getrokken is, en zelfs bij een perfecte steekproef, als een groep klein is. *Measurement bias* betekent dat een feature of label een vervanger is voor waar het je om gaat, en dat voor sommige groepen slechter meet. Arrestaties staan voor criminaliteit, en in wijken waar meer politie is, worden meer mensen opgepakt. Een Amerikaans zorgalgoritme nam kosten als vervanger voor zorgbehoefte: bij dezelfde score van het algoritme waren zwarte patiënten duidelijk zieker dan witte patiënten, omdat er minder geld aan hun zorg werd uitgegeven. Dat verschil wegnemen zou het aandeel zwarte patiënten dat extra hulp kreeg, verhogen van 17,7 naar 46,5 procent (Obermeyer en anderen, 2019).

        **Het model bouwen en gebruiken.** *Aggregation bias* komt van één model voor groepen die je anders zou moeten behandelen: het past dan bij geen enkele groep goed, of alleen bij de grootste. De bloedwaarde HbA1c, veel gebruikt om diabetes vast te stellen en te volgen, verschilt op ingewikkelde manieren tussen etniciteiten en tussen mannen en vrouwen, dus kan één model of één grens voor iedereen bij geen enkele groep goed passen. *Learning bias* komt van keuzes in hoe een model leert die het verschil tussen groepen groter maken. Een getraind model kleiner maken met pruning kan bijvoorbeeld de verschillen in prestaties voor zeldzame groepen vergroten: met weinig ruimte houdt het model vast wat het vaakst voorkomt. *Evaluation bias* komt van testdata die niet lijkt op de mensen voor wie het model gebruikt wordt, en van één totale score die een groep verbergt, zoals bij *Gender Shades*. *Deployment bias* komt van een model anders gebruiken dan waarvoor het gebouwd is. Programma's die moeten voorspellen of iemand opnieuw een misdrijf zal plegen, zijn ook gebruikt voor iets waar ze niet voor bedoeld waren: om mee te bepalen hoe lang een straf wordt.

        **Waar de oplossing zit.** Elke stap heeft zijn eigen soort oplossing. Tegen representation bias helpt meer data, als het data is van de mensen die ontbraken. Tegen historical en measurement bias helpt meer data uit hetzelfde proces waarschijnlijk niet, schrijven Suresh en Guttag, omdat die door dezelfde stappen gaat. Measurement bias vraagt om beter meten of labelen; bij historical bias wordt de vraag wat een model überhaupt moet leren van een ongelijke wereld. Aggregation bias vraagt om een model dat rekening houdt met de verschillen tussen groepen, learning bias om een ander doel of een andere manier van leren, evaluation bias om een metric per groep of een betere benchmark, en deployment bias om aandacht voor hoe mensen het model echt gebruiken. Suresh en Guttag geven met opzet geen lijst met oplossingen, om niet te suggereren dat er een complete set oplossingen bestaat die overal werkt.

        **Bias in mensen.** Drie woorden uit de psychologie en de statistiek. *Implicit bias* zijn houdingen en stereotypen die mensen hebben zonder dat ze zich daarvan bewust zijn: eerdere ervaring beïnvloedt een oordeel op een manier die iemand bij zichzelf niet kan zien, schreven Anthony Greenwald en Mahzarin Banaji in 1995, en de term implicit bias kwam later uit dat onderzoek voort. *In-group bias* is de neiging om leden van je eigen groep gunstiger te beoordelen dan vergelijkbare mensen uit een andere groep. Het komt zelfs voor als mensen zien dat ze bij toeval in een groep worden ingedeeld: het minimal group effect, voor het eerst laten zien door Tajfel in 1970. Als mensen met deze biases de labels geven, of de eerdere beslissingen in de data namen, komen die biases in een model terecht als measurement bias of historical bias. *Selection bias* betekent dat de voorbeelden in de data systematisch verschillen van de mensen over wie je iets wilt zeggen, door hoe ze gekozen zijn. Een model voor werving leert alleen hoe het de aangenomen mensen verging, en ziet nooit hoe de anderen het gedaan zouden hebben. Selection bias is één manier om representation bias te krijgen, maar niet de enige: ook een perfecte steekproef kan te weinig voorbeelden van een kleine groep hebben.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent |
        |---|---|
        | **historical bias** | bias die al in de wereld zit waar de data vandaan komt, zodat zelfs perfect gemeten en getrokken data tot een model leidt dat schade doet |
        | **representation bias** | te weinig voorbeelden van een deel van de mensen voor wie een model gebruikt wordt, zodat het voor hen slechter werkt |
        | **measurement bias** | een feature of label is een vervanger voor waar het je om gaat, en meet dat voor sommige groepen slechter |
        | **aggregation bias** | één model voor groepen waarbij de features op een andere manier met de uitkomst samenhangen, zodat het bij geen van hen goed past |
        | **learning bias** | keuzes in hoe een model leert maken het verschil tussen groepen groter |
        | **evaluation bias** | testdata die niet lijkt op de mensen voor wie een model gebruikt wordt, zodat een goede score een groep verbergt |
        | **deployment bias** | een model wordt anders gebruikt dan waarvoor het gebouwd is |
        | **implicit bias** | houdingen en stereotypen die mensen hebben zonder dat ze zich daarvan bewust zijn |
        | **in-group bias** | leden van je eigen groep gunstiger beoordelen dan vergelijkbare anderen |
        | **selection bias** | de voorbeelden in de data verschillen systematisch van de mensen over wie je iets wilt zeggen, door hoe ze gekozen zijn |
        """),
        L.watch_out("Een test van implicit bias kan niet zeggen of één bepaalde persoon bevooroordeeld is: de uitkomsten wisselen daarvoor te veel van keer tot keer, schreven Greenwald en Calvin Lai in 2020. En omdat onderzoekers implicit bias niet betrouwbaar kunnen verminderen, schreven ze, is er geen reden om te verwachten dat een training dat wel kan."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om te vinden waar een bias binnenkwam:

    1. Zeg wat er misgaat, en voor wie.
    2. Loop de stappen af van de wereld tot het gebruik, en vraag je bij elke stap iets af. Was de wereld zelf ongelijk? Wie ontbreekt er in de data, of komt weinig voor? Meet elke feature en elk label waar het je om gaat voor iedereen even goed? Past één model bij elke groep? Maken de keuzes bij het trainen een verschil groter? Lijkt de testdata op de mensen voor wie het model gebruikt wordt, en meet je per groep? Wordt het model gebruikt zoals het bedoeld is?
    3. Tel waar het kan: het aandeel van elke groep in de data, de metric per groep, wie een regel kiest.
    4. Leg de oplossing bij de stap die je benoemde. Meer data van dezelfde soort helpt alleen tegen representation bias.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een zorgprogramma, met patiënten die de course verzon, naar het Amerikaanse zorgalgoritme hierboven. Elke patiënt in `_patients` heeft een groep, een aantal chronische ziektes (hoe ziek iemand is) en de zorgkosten van vorig jaar, in duizenden. Het programma neemt kosten als teken van zorgbehoefte, en helpt iedereen met kosten van `_limit` of meer. `_helped` houdt per groep een aantal bij: `_helped[_patient["group"]]` is het aantal van de eigen groep van de patiënt, en de regel onder de `if` telt er 1 bij op. Voer de cel uit.
    """)
    return


@app.cell
def example_helped():
    # verzonnen patiënten: groep, aantal chronische ziektes, zorgkosten van vorig jaar in duizenden
    _patients = [
        {"group": "A", "illnesses": 3, "cost": 7},
        {"group": "B", "illnesses": 5, "cost": 5},
        {"group": "A", "illnesses": 1, "cost": 2},
        {"group": "B", "illnesses": 2, "cost": 3},
        {"group": "A", "illnesses": 6, "cost": 11},
        {"group": "B", "illnesses": 6, "cost": 6},
        {"group": "A", "illnesses": 2, "cost": 5},
        {"group": "B", "illnesses": 1, "cost": 1},
        {"group": "A", "illnesses": 5, "cost": 8},
        {"group": "B", "illnesses": 3, "cost": 4},
    ]
    # het programma: help iedereen met zorgkosten van minstens zoveel
    _limit = 6

    _helped = {"A": 0, "B": 0}
    for _patient in _patients:
        if _patient["cost"] >= _limit:
            _helped[_patient["group"]] = _helped[_patient["group"]] + 1
    print(_helped)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Groep A krijgt drie van de vier plekken, groep B één. Maar kijk naar de ziektes: in beide groepen hebben de patiënten er 1, 2, 3, 5 en 6, dus de groepen zijn even ziek. Bij dezelfde ziekte werd minder aan groep B uitgegeven, dus zijn de kosten lager. In deze verzonnen lijst krijgt de patiënt van groep B met 5 ziektes geen hulp, en de patiënt van groep A met 3 wel. Wat als het programma op zorgbehoefte kiest? Oefening 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_helped():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Op zorgbehoefte

        Laat het programma op zorgbehoefte kiezen in plaats van op kosten: het helpt iedereen met minstens `need` chronische ziektes. Vervang de `...` door code zoals in het uitgewerkte voorbeeld, met de namen zonder liggend streepje: begin met `helped = {"A": 0, "B": 0}`, loop met `patient` over `patients`, en tel 1 op bij het aantal van de groep van de patiënt als `patient["illnesses"]` `need` of meer is. De laatste regel is `helped`, zodat de check het ziet. De check probeert ook andere waarden van `need`.
        """),
        L.task("Reken `helped` uit: per groep, hoeveel patiënten minstens `need` chronische ziektes hebben."),
    ])
    return


@app.cell(hide_code=True)
def exercise_helped():
    box_helped = L.code_box("helped", lines=22)
    box_helped
    return (box_helped,)


@app.cell(hide_code=True)
def check_helped(box_helped):
    result_helped = L.check("helped", box_helped.value)
    result_helped
    return (result_helped,)


@app.cell(hide_code=True)
def hints_helped():
    L.hints(
        "helped",
        [
            "Begin bij het uitgewerkte voorbeeld: de loop en de regel die telt blijven hetzelfde. Alleen de `if` verandert: die kijkt naar de ziektes, niet naar de kosten.",
            "De `if` vergelijkt `patient[\"illnesses\"]` met `need`. Welke vergelijking betekent *of meer*?",
            "`if patient[\"illnesses\"] ___ need:`, en daaronder de regel die telt uit het uitgewerkte voorbeeld, zonder liggende streepjes.",
        ],
        why="`helped = {\"A\": 0, \"B\": 0}`, dan `for patient in patients:` met daaronder `if patient[\"illnesses\"] >= need:`, en daarbinnen `helped[patient[\"group\"]] = helped[patient[\"group\"]] + 1`. Dat geeft `{'A': 2, 'B': 2}`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_steps():
    mo.md(r"""
    ### Oefening 2. Welke soort bias?

    Zes gevallen uit de eerste helft van de weg, waar de data gemaakt wordt. Deel elk geval in bij de soort bias die erin zit.
    """)
    return


@app.cell(hide_code=True)
def sort_steps_cell():
    sort_steps = L.sort("steps")
    sort_steps
    return (sort_steps,)


@app.cell(hide_code=True)
def check_steps(sort_steps):
    result_steps = L.check("steps", sort_steps)
    result_steps
    return (result_steps,)


@app.cell(hide_code=True)
def hints_steps():
    L.hints(
        "steps",
        [
            "Vraag je bij elk geval af: klopt de data terwijl de wereld zelf ongelijk was, ontbreekt er een groep of komt die weinig voor, of meet een label iets slechter voor één groep?",
            "Een andere stad gaat over van wie de data komt. Welke twee gevallen gaan over een label dat voor iets anders staat?",
            "Historical bias: loon en \\_\\_\\_. Representation bias: een andere stad en \\_\\_\\_. Measurement bias: \\_\\_\\_ en inspecties.",
        ],
        why="**Historical bias:** loon en CEO's; de data klopt, maar de wereld was ongelijk. **Representation bias:** een andere stad en vijf procent. **Measurement bias:** diagnoses en inspecties; het label meet waar het om gaat slechter voor één groep of één plek.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fix():
    mo.md(r"""
    ### Oefening 3. Meer van hetzelfde?
    """)
    return


@app.cell(hide_code=True)
def choice_fix_cell():
    choice_fix = L.choice(
        "fix",
        question="Een collega wil het zorgprogramma uit het uitgewerkte voorbeeld verbeteren door twee keer zoveel patiënten op dezelfde manier te verzamelen, met dezelfde regel op kosten. Wat vind jij?",
        options={
            "more": "Ja, meer data betekent minder bias",
            "right": "Nee, kosten meten de zorgbehoefte van B slecht",
            "share": "Alleen als de helft van de nieuwe patiënten in groep B zit",
        },
        correct="right",
        why={
            "more": "Nieuwe patiënten die op dezelfde manier verzameld zijn, gaan door dezelfde regel, en bij dezelfde ziekte kost groep B nog steeds minder. Meer data helpt alleen tegen representation bias. Zie *Waar de oplossing zit* in *Het idee*.",
            "right": "Dit is measurement bias: het label, kosten, is een vervanger die de zorgbehoefte van groep B slechter meet. De oplossing zit in het label, zoals in Oefening 1, niet in meer data.",
            "share": "Groep B is al de helft van de patiënten: die ontbreekt niet. De kosten van groep B zijn lager bij dezelfde ziekte, en meer van dezelfde data houdt dat zo. Zie *Waar de oplossing zit* in *Het idee*.",
        },
    )
    choice_fix
    return (choice_fix,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Wie staat er in de data, en wie ontbreekt?](/course/open/rai-bias-in-data?lang=nl) zocht je op drie plekken naar scheefheid: wie er in de data staat, wie de labels gaf, en het verleden dat de data vastlegt. Hier werden dat historical, representation en measurement bias, met nog vier plekken bij het bouwen en gebruiken van het model, en een regel: benoem de stap, en leg de oplossing daar. In [De workflow van machine learning](/course/open/ml-workflow?lang=nl) ging elke ronde terug naar de stap die het model tegenhoudt; bij bias ga je terug naar de stap waar die binnenkwam. De volgende les, *Feedback loops*, volgt de gestippelde pijl in het plaatje: hoe de beslissingen van een model de data veranderen waar het daarna van leert.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fix, result_helped, result_steps):
    L.wrap_up(result_helped, result_steps, choice_fix)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
