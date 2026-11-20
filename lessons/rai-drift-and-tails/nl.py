# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-drift-and-tails. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow AI Safety, Accident, Black Swan

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-drift-and-tails", "nl")
    C = L.checks   # een grens die oprekt, een gemiddelde na elke waarde, en een dunne staart naast een lange


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Een vraag over [Safeguards achter elkaar](/course/open/rai-swiss-cheese?lang=nl), en een over [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_floor():
    quick_floor = L.choice(
        "quick-floor",
        question="Drie safeguards missen elk 10% van de gewone schadelijke verzoeken, onafhankelijk van elkaar, en een truc in 1% van de verzoeken leidt alle drie om de tuin. Wat verlaagt het deel dat erdoor komt het meest?",
        options={
            "fourth": "Een vierde safeguard die ook 10% mist",
            "right": "Het gat van de truc dichten in één safeguard",
            "order": "De drie safeguards in een andere volgorde zetten",
        },
        correct="right",
        why={
            "right": "De truc legt een bodem: de 1% met de truc komt erdoor, hoeveel safeguards er ook achter elkaar staan. Zodra één safeguard ook de truc vangt, moeten de verzoeken met de truc daar langs zoals elk ander verzoek.",
            "fourth": "Een vierde safeguard verlaagt het gewone deel van 0,1% naar 0,01%, maar de 1% met de truc komt nog steeds door elke safeguard. Zie [Safeguards achter elkaar](/course/open/rai-swiss-cheese?lang=nl).",
            "order": "Bij onafhankelijke safeguards verandert de volgorde niets: de kansen vermenigvuldigen op dezelfde manier, en de truc leidt nog steeds alle drie om de tuin. Zie [Safeguards achter elkaar](/course/open/rai-swiss-cheese?lang=nl).",
        },
    )
    quick_floor
    return (quick_floor,)


@app.cell(hide_code=True)
def quick_check_seed():
    quick_seed = L.choice(
        "quick-seed",
        question="Een cel voert `rng = np.random.default_rng(7)` uit en daarna `rng.random(size=3)`. Je voert de cel twee keer uit. Wat krijg je?",
        options={
            "right": "Twee keer dezelfde drie getallen",
            "new": "Bij elke keer uitvoeren drie nieuwe getallen",
            "whole": "Drie gehele getallen onder 7",
        },
        correct="right",
        why={
            "right": "Een generator met een vaste seed geeft bij elke run dezelfde getallen, zodat je een resultaat kunt herhalen. Daarom zijn de trekkingen in deze les voor iedereen hetzelfde.",
            "new": "Met een vaste seed begint de generator bij elke run in dezelfde toestand, dus geeft hij dezelfde getallen. Zie [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).",
            "whole": "`rng.random` geeft getallen tussen 0 en 1; gehele getallen komen van `rng.integers`. De 7 is de seed, geen grens. Zie [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).",
        },
    )
    quick_seed
    return (quick_seed,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een verzonnen bedrijf draait een chatbot die per dag miljoenen vragen beantwoordt. Het logboek met incidenten staat vol kleine missers: hier een verkeerde datum, daar een onbeleefd antwoord. Dan verspreidt één antwoord zich over het internet en doet het meer kwaad dan alle eerdere incidenten samen.

    Dat is niet alleen pech. Twee ideeën uit de veiligheidstechniek verklaren het: systemen glijden af naar falen via kleine stappen die er elk prima uitzien, en in sommige domeinen weegt één zeldzame gebeurtenis zwaarder dan alles ervoor. Deze les geeft ze allebei, met de spaceshuttle Columbia als gedocumenteerd geval en NumPy om getallen met een lange staart te trekken.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Een spel: je wint 15 dollar met een kans van 99,7%, en je raakt 10.000 dollar kwijt met een kans van 0,3%. Wat levert het spel je gemiddeld per keer op?",
        options={
            "most": "Winst van bijna 15 dollar per keer",
            "five": "Winst van ongeveer 5 dollar per keer",
            "right": "Ongeveer 15 dollar kwijt per keer",
        },
        reveal="**Ongeveer 15 dollar kwijt per keer.** 0,997 × 15 + 0,003 × (−10.000) = 14,955 − 30 = −15,045. Je wint in 997 van de 1000 keer, maar de drie keer dat je verliest, kosten meer dan je al die andere keren wint. Het leerboek gebruikt dit spel: met een kans van 0,1% om te verliezen levert het spel je gemiddeld 4,985 dollar per keer op, en bij 0,3% kost het je 15,045 dollar per keer. Een verandering van 0,2 procentpunt in één zeldzame kans draait het teken om.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Afglijden, één kleine stap tegelijk

    Het leerboek *Introduction to AI Safety, Ethics, and Society* (2024) van Dan Hendrycks beschrijft verschillende modellen van hoe ongelukken gebeuren. In zijn weergave van het raamwerk van Jens Rasmussen zijn ongelukken meestal het gevolg van een samenspel van veel factoren, niet van één enkele oorzaak, en kunnen gedrag en omstandigheden onder druk van buitenaf in de loop van de tijd langzaam opschuiven. Zijn weergave van *Drift into Failure* van Sidney Dekker voegt daaraan toe wat het decrementalism noemt: de processen van een systeem verslechteren door een reeks kleine veranderingen. Elke stap is klein: veel tamelijk kleine beslissingen kunnen samen een groot verschil in risico maken. En het leerboek waarschuwt dat men dat vaak pas merkt als er een ongeluk gebeurt, en dan is het te laat.

    Waarom houdt niemand het tegen? Het leerboek geeft een reden in zijn weergave van het STAMP-model: als minder letten op veiligheid niet snel tot een ongeluk leidt, gaan ontwikkelaars misschien denken dat letten op veiligheid niet nodig is. Dat is **drift into failure**: volgens het leerboek glijden systemen vaak af naar falen door een opeenstapeling van veel ogenschijnlijk kleine gebeurtenissen, waarvan geen enkele op zichzelf een ramp zou zijn. Over AI zegt het leerboek het voorzichtig: het is mogelijk dat veel tamelijk kleine gebeurtenissen zich opstapelen, zodat we afglijden naar een existentieel risico.

    ### Columbia: een afwijking die normaal werd

    Op 1 februari 2003 brak de spaceshuttle Columbia uiteen bij de terugkeer in de atmosfeer, en de zeven bemanningsleden kwamen om. De Columbia Accident Investigation Board (augustus 2003) vond de fysieke oorzaak: een stuk isolatieschuim dat 81,7 seconden na de lancering loskwam van de External Tank en de linkervleugel raakte. Er was eerder schuim losgekomen. Volgens de Board laat de geschiedenis van de beslissingen over het schuim zien hoe NASA begon met vliegen terwijl er schuim loskwam en daarmee doorging, zodat vliegen met deze afwijkingen van het ontwerp als normaal en aanvaardbaar ging gelden. In de loop van 113 missies gingen loskomend schuim en andere inslagen van brokstukken meer gelden als een kwestie van onderhoud tussen twee vluchten, en minder als een gevaar voor het toestel en de bemanning. De Board gebruikt daarvoor de term van socioloog Diane Vaughan: de normalization of deviance.

    De Board beschrijft hoe het geaccepteerde bereik groeide. Toen de eerste afwijkingen eenmaal geaccepteerd waren, golden latere inslagen van schuim niet als tekenen van gevaar, maar als aanwijzing dat het ontwerp nu deed wat men had voorspeld. De technische ervaring die steeds ergere afwijkingen opnam, werkte volgens de Board als een elastische tailleband, die uitrekte om steeds grotere afwijkingen van het oorspronkelijke ontwerp te omvatten. Afwijkingen die niet tot een ramp leidden, werden behandeld als geldige technische gegevens die verdere vluchten rechtvaardigden. De Board haalt ook Richard Feynman aan over het ongeluk met de Challenger in 1986, waar de afdichtingen in de verbindingen van de raketboosters, de O-ringen, op eerdere vluchten al erosie hadden vertoond. Feynman beschrijft dezelfde fout: dat de vluchten werden geaccepteerd en slaagden, werd opgevat als een teken van veiligheid. Daarna draait Feynman het om: de erosie was een aanwijzing dat er iets mis was.
    """)
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Staartgebeurtenissen en zwarte zwanen

    Het leerboek omschrijft staartgebeurtenissen als gebeurtenissen die zelden voorkomen, maar flinke gevolgen hebben als ze gebeuren. Voorbeelden die het geeft zijn de financiële crisis van 2007 en 2008, COVID-19 en het internet. En een staartgebeurtenis verandert vaak het gemiddelde, maar niet de mediaan. Het risico op zulke zeldzame, extreme uitkomsten heet **staartrisico**: als de staart lang is, kunnen die paar uitkomsten het grootste deel van het totaal uitmaken.

    Of een domein staartgebeurtenissen heeft, hangt af van de verdeling. In verdelingen met een dunne staart bestaan staartgebeurtenissen volgens het leerboek niet; voorbeelden zijn menselijke eigenschappen zoals lengte, gewicht en intelligentie. Voorbeelden van een lange staart zijn de verkoop van boeken, de kracht van aardbevingen en hoe vaak woorden voorkomen. In de tabel van het leerboek kan bij een lange staart één gebeurtenis uitgroeien tot iets dat veel groter is dan veel andere samen. Het leerboek waarschuwt dat niet alle verdelingen netjes in twee soorten passen: de tabel geeft vereenvoudigde karikaturen. M. E. J. Newman geeft in een paper over zulke verdelingen een idee van het verschil: de langste en de kleinste volwassen man in het Guinness Book of Records toen Newman schreef, 272 cm en 57 cm, verschillen een factor 4,8; de grootste stad en het kleinste stadje van de VS verschillen in inwonertal een factor van minstens 150.000.

    Volgens het leerboek is een zwarte zwaan een staartgebeurtenis die voor de meeste mensen grotendeels onvoorspelbaar was voordat ze gebeurde, en niet elke staartgebeurtenis is een zwarte zwaan. Een grote aardbeving die seismologen in Californië verwachten, zou een staartgebeurtenis zijn, maar geen zwarte zwaan. De naam: in 1697 kwam een groep Nederlandse ontdekkingsreizigers in Australië zwarte zwanen tegen, al is de beeldspraak zelf ouder. Het leerboek werkt met deze ruime omschrijving, een gebeurtenis met grote gevolgen die voor de meeste mensen grotendeels onverwacht is, en niet met een strenge, een gebeurtenis die helemaal niemand kon zien aankomen. Nassim Nicholas Taleb, van wie het leerboek het boek *The Black Swan* aanhaalt, benadrukt in een later boek, *Statistical Consequences of Fat Tails*, dat het afhangt van wie er kijkt: een zwarte zwaan voor de kalkoen is een witte zwaan voor de slager.

    ### Waarom het verleden weinig zegt over het volgende uiterste

    Bij een lange staart is een gemiddelde uit het verleden een slechte leidraad. Volgens het leerboek kan één extreme waarde het gemiddelde met enkele ordes van grootte veranderen. En uit zijn voorbeelden van zwarte zwanen concludeert het dat we het toekomstige risico niet zomaar goed kunnen uitrekenen uit gegevens over een lange periode. In hetzelfde latere boek schrijft Taleb dat toekomstige uitersten bij zware staarten niet lijken op die van het verleden, en dat dat verschil groter wordt naarmate de staarten zwaarder zijn. En het leerboek: dat je geen aanwijzingen ziet, is geen sterke aanwijzing dat iets er niet is. Voor sommige lange staarten komt het gemiddelde nooit tot rust: in de verdelingen die Newman vindt voor de hevigheid van oorlogen en van zonnevlammen, is het gemiddelde oneindig.

    Newman geeft een recept voor getallen met een lange staart. Neem een willekeurig getal r tussen 0 en 1, en reken (1 − r) tot de macht −1/1,5 uit: meestal iets boven 1, soms enorm. Met de formule van Newman is het gemiddelde van dit recept op de lange duur 3, maar het gemiddelde komt er maar langzaam.

    ### Wat helpt

    Tegen drift is de les die het leerboek uit Rasmussen haalt dat je de veiligheidsgrenzen steeds moet blijven handhaven, zodat het systeem niet naar onveilige toestanden opschuift. Tegen echte zwarte zwanen kun je volgens het leerboek geen specifieke verdediging inrichten, maar stresstests, horizon scanning, red teaming (een systeem met opzet aanvallen om de zwakke plekken te vinden) en wat het safety imagination noemt, kunnen sommige ervan veranderen in risico's die in beeld zijn, ook als ze niet begrepen worden. Over AI stelt het, voorzichtig geformuleerd, dat complexe systemen misschien eerder een risico op zwarte zwanen meebrengen, en dat nieuwe systemen misschien eerder zwarte zwanen opleveren.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **drift into failure** | een systeem dat afglijdt naar een ongeluk via veel kleine stappen, die elk geaccepteerd worden omdat de vorige geen kwaad deed |
        | **staartrisico** | het risico op zeldzame, extreme uitkomsten ver in de staart van een verdeling; bij een lange staart kunnen die het grootste deel van het totale risico uitmaken |
        | normalization of deviance | een afwijking van het ontwerp die als normaal gaat gelden omdat ze nog geen ongeluk heeft veroorzaakt |
        | staartgebeurtenis | een gebeurtenis die zelden voorkomt, maar flinke gevolgen heeft als ze gebeurt |
        | zwarte zwaan | een staartgebeurtenis die de meeste mensen niet zagen aankomen; wie verrast is, hangt af van wie er kijkt |
        | dunne staart | een verdeling zonder staartgebeurtenissen, zoals menselijke lengte: geen waarde ligt ver van de rest |
        | lange staart | een verdeling waarin één waarde groter kan zijn dan veel andere samen, zoals de verkoop van boeken |
        | `x ** (-1 / 1.5)` | x tot de macht −1/1,5 in Python: één gedeeld door x tot de macht 1/1,5 |
        """),
        L.watch_out("Een lange tijd zonder ongeluk is geen teken van veiligheid. Bij drift groeit het risico terwijl er niets gebeurt, en bij een lange staart zegt het ergste tot nu toe weinig over het volgende."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Schrijf de ontwerpgrens op voordat je begint, en houd die apart van wat als normaal is gaan gelden.
    2. Vraag bij elke afwijking of die geaccepteerd wordt omdat je die begrijpt, of alleen omdat er nog niets misging.
    3. Kijk naar een verdeling voordat je een gemiddelde vertrouwt: vergelijk het gemiddelde met de mediaan, en de grootste waarde met het totaal.
    4. Als één waarde groter kan zijn dan veel andere samen, behandel de grootste tot nu toe dan als een ondergrens voor het ergste dat nog kan komen, niet als een bovengrens.
    5. Gebruik tegen de zeldzame gebeurtenissen die je niet kunt opsommen stresstests, horizon scanning en red teaming, en blijf de veiligheidsgrenzen handhaven.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel trekt 1000 verzonnen lengtes, elk 140 plus 80 keer het gemiddelde van 12 willekeurige getallen, zodat ze tussen 140 en 220 cm liggen, rond 180. De cel trekt ook 1000 verzonnen groottes van incidenten met het recept van Newman, en drukt voor beide het gemiddelde, de mediaan en de grootste waarde af.
    """)
    return


@app.cell
def example_tails():
    import numpy as np

    rng = np.random.default_rng(7)
    heights = 140 + 80 * rng.random(size=(1000, 12)).mean(axis=1)   # cm: een dunne staart
    sizes = (1 - rng.random(size=1000)) ** (-1 / 1.5)               # incidenten: een lange staart
    for name, values in [("lengtes", heights), ("incidenten", sizes)]:
        print(name, round(values.mean(), 1), round(np.median(values), 1), round(values.max(), 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De lengtes hebben een gemiddelde van 180,1 en een mediaan van 179,8, en de langste is 199,9 cm, 11% boven het gemiddelde. De incidenten hebben een gemiddelde van 2,9 en een mediaan van 1,6, en het grootste is 177,4: ongeveer 60 keer het gemiddelde, en alleen al ongeveer 6% van het totaal van alle 1000. In de oefeningen schrijf je hoe een geaccepteerde grens oprekt, en het gemiddelde na elke waarde, dat laat zien wat één grote waarde doet.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_stretch():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. stretch: de elastische tailleband

        Elke vlucht heeft een afwijking van het ontwerp, zoals hoeveel schuim er loskwam. Eerst is de grens van wat aanvaardbaar is de ontwerpgrens. Als een vlucht een grotere afwijking heeft en er niets misgaat, wordt die afwijking de nieuwe grens. Schrijf `stretch(deviations, design)`. De functie krijgt de afwijkingen van de vluchten op volgorde en de ontwerpgrens, en geeft een lijst `accepted` terug met de geaccepteerde grens na elke vlucht. Houd de grens bij in een variabele `limit` en loop met een `for`-loop door de vluchten, met `deviation` voor elke vlucht. Vervang de `...`. De check roept `stretch([0.5, 1.4, 1.2, 2.0, 1.8, 3.1], 1)` aan, daarna andere vluchten.
        """),
        L.task("De geaccepteerde grens na elke vlucht."),
    ])
    return


@app.cell(hide_code=True)
def exercise_stretch():
    box_stretch = L.code_box("stretch", lines=8)
    box_stretch
    return (box_stretch,)


@app.cell(hide_code=True)
def check_stretch(box_stretch):
    result_stretch = L.check("stretch", box_stretch.value)
    result_stretch
    return (result_stretch,)


@app.cell(hide_code=True)
def hints_stretch():
    L.hints(
        "stretch",
        [
            "De geaccepteerde grens gaat alleen omhoog: na elke vlucht is die het grootste van de ontwerpgrens en alle afwijkingen tot nu toe. Een kleinere afwijking laat de grens staan waar die staat.",
            "Begin `limit` bij `design` en `accepted` als een lege lijst. Vraag in een `for`-loop over `deviations` of de `deviation` groter is dan `limit`; zo ja, dan wordt die de nieuwe `limit`. Voeg daarna `limit` toe aan `accepted`, en geef na de loop `accepted` terug.",
            "`        if deviation ___ limit:`",
        ],
        why="`limit = design` en `accepted = []`, dan `for deviation in deviations:` met `if deviation > limit:` en `limit = deviation`, en daarna `accepted.append(limit)`; na de loop `return accepted`. In het eerste geval gaat de grens van 1 naar 1,4, 2,0 en 3,1.",
    )
    return


@app.cell(hide_code=True)
def text_after_stretch():
    mo.md(r"""
    In het eerste geval zei het ontwerp 1, en na zes vluchten geldt een afwijking van 3,1 als aanvaardbaar, meer dan drie keer het ontwerp. Elke stap werd gemeten aan de laatst geaccepteerde grens, niet aan het ontwerp: de grens groeide met 0,4, 0,6 en 1,1, en elke keer ging er niets mis.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_mean():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. running_mean: het gemiddelde tot nu toe

        Schrijf `running_mean(values)`. De functie geeft een lijst `means` terug met het gemiddelde van de eerste waarde, van de eerste twee, van de eerste drie, enzovoort. Houd de som tot nu toe bij in een variabele `total` en het aantal waarden tot nu toe in `count`, en loop met een `for`-loop door de waarden, met `value` voor elke waarde. De lijst kan leeg zijn. Vervang de `...`. De check roept `running_mean([2, 4, 6])` aan, daarna andere lijsten.
        """),
        L.task("Het gemiddelde na elke waarde."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mean():
    box_mean = L.code_box("running_mean", lines=9)
    box_mean
    return (box_mean,)


@app.cell(hide_code=True)
def check_mean(box_mean):
    result_mean = L.check("running_mean", box_mean.value)
    result_mean
    return (result_mean,)


@app.cell(hide_code=True)
def hints_mean():
    L.hints(
        "running_mean",
        [
            "Voor het gemiddelde tot nu toe heb je twee getallen nodig: de som tot nu toe en het aantal waarden tot nu toe. Allebei groeien ze met één waarde in elke stap van de loop.",
            "Begin `total` en `count` bij 0 en `means` als een lege lijst. Tel in de loop de `value` op bij `total` en 1 bij `count`, en voeg dan het gemiddelde tot nu toe toe aan `means`. Geef na de loop `means` terug.",
            "`        means.append(total ___ count)`",
        ],
        why="`total = 0`, `count = 0` en `means = []`, dan `for value in values:` met `total = total + value`, `count = count + 1` en `means.append(total / count)`; na de loop `return means`. Voor `[2, 4, 6]` geeft dat `[2.0, 3.0, 4.0]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_mean():
    mo.md(r"""
    In het geval `[1, 1, 1, 97]` is het gemiddelde 1 na drie waarden en 25 na de vierde: één waarde verplaatst het meer dan alle andere samen. De mediaan van die vier waarden is nog steeds 1.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: een dunne staart en een lange staart

    Lengtes en groottes van incidenten, op dezelfde manier getrokken met een vaste seed, tot 1.000.000 van elk. Kies naar hoeveel trekkingen je kijkt, en vergelijk het gemiddelde, de mediaan en de grootste waarde van beide, en het deel van het totaal dat de grootste waarde inneemt. De grafiek toont het gemiddelde van de groottes van de incidenten na elke trekking.
    """)
    return


@app.cell(hide_code=True)
def demo_tails():
    tails = L.explore("draws", "trekkingen van elk", lambda _v: C.demo_frame(_v, L.lang),
                      start=1000, values=C.DRAWS, role="",
                      task="Zoek het kleinste aantal trekkingen op de schuif waarbij één enkel incident meer dan een vijfde van het totaal van alle groottes van incidenten uitmaakt.")
    tails
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Bij 10.000 trekkingen: trekking 6419 is een incident van ongeveer 10.500, ongeveer een kwart van het totaal van alle 10.000. Die ene trekking tilt het gemiddelde in één stap van 2,85 naar 4,49; de mediaan beweegt nauwelijks. Bij de lengtes gebeurt dit nooit: bij 1000 trekkingen is hun langste ongeveer een duizendste van hun totaal, en daarna wordt dat deel alleen maar kleiner. Bij 1.000.000 trekkingen ligt het gemiddelde van de incidenten weer rond 3, en dat ene incident is nog steeds het grootste. Wie alleen naar de eerste 6418 trekkingen had gekeken, had nooit een incident boven 263 gezien, ongeveer een veertigste daarvan.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_drift():
    mo.md(r"""
    ### Oefening 3. Elke maand minder tests

    Een team brengt elke maand een nieuwe versie van zijn chatbot uit, en draait voor elke release een reeks veiligheidstests. Om tijd te besparen, schrapt het elke maand een paar tests die nog nooit gefaald hebben. Na een jaar is de helft van de tests weg, en geen enkele release heeft een incident veroorzaakt.
    """)
    return


@app.cell(hide_code=True)
def choice_drift_cell():
    choice_drift = L.choice(
        "drift",
        question="Wat zegt deze les over de redenering van het team?",
        options={
            "fine": "Die klopt: een test die in een jaar niet faalde, beschermt nergens tegen",
            "right": "Een jaar zonder incidenten laat niet zien dat de overgebleven tests genoeg zijn",
            "swan": "Alle schade die later komt is een zwarte zwaan, dus het team had niets beter kunnen doen",
        },
        correct="right",
        why={
            "right": "Elke schrapping wordt geaccepteerd omdat de vorige geen kwaad deed: dat is drift into failure. De woorden van Feynman over de shuttle passen hier ook: het slagen van de vluchten wordt opgevat als een teken van veiligheid. Bij drift groeit het risico terwijl er niets gebeurt.",
            "fine": "Een test die nooit faalde, is misschien het probleem waarvoor die bestaat nog niet tegengekomen. Een jaar zonder incident is hoe drift er van binnenuit uitziet: elke stap lijkt prima tot het ongeluk. Dat je geen aanwijzingen ziet, is geen sterke aanwijzing dat iets er niet is. Zie *Afglijden, één kleine stap tegelijk* in *Het idee*.",
            "swan": "Een zwarte zwaan is een gebeurtenis die de meeste mensen niet zagen aankomen. Hier koos het team elke stap zelf, en een risico dat stap voor stap groeit, kun je zien en stoppen. Zie *Afglijden, één kleine stap tegelijk* in *Het idee*.",
        },
    )
    choice_drift
    return (choice_drift,)


@app.cell(hide_code=True)
def text_exercise_whose_swan():
    mo.md(r"""
    ### Oefening 4. Voor wie een zwarte zwaan?

    In maart schrijft een red team, een groep die een systeem met opzet aanvalt om de zwakke plekken te vinden, in een rapport dat er een nieuw soort jailbreak zou kunnen opduiken die zich snel verspreidt: een truc die chatbots langs hun safeguards krijgt. In juni duikt die op en wordt die tegen veel chatbots tegelijk gebruikt. De meeste bedrijven die ze draaien, hadden er nog nooit van gehoord.
    """)
    return


@app.cell(hide_code=True)
def choice_whose_swan_cell():
    choice_whose_swan = L.choice(
        "whose_swan",
        question="Is de jailbreak een zwarte zwaan?",
        options={
            "right": "Voor de meeste bedrijven wel; voor het red team was het een voorziene staartgebeurtenis",
            "no": "Nee: een gebeurtenis is alleen een zwarte zwaan als helemaal niemand die had kunnen voorzien",
            "yes": "Ja, voor iedereen: zeldzaam, met grote gevolgen, en het kwam zonder enige waarschuwing",
        },
        correct="right",
        why={
            "right": "Een zwarte zwaan hangt af van wie er kijkt, zoals de kalkoen en de slager van Taleb laten zien. Onder de werkdefinitie van het leerboek telt een gebeurtenis die voor de meeste mensen grotendeels onverwacht was, terwijl wie die zag aankomen een staartgebeurtenis zag.",
            "no": "Dat is de strenge omschrijving, die het leerboek niet gebruikt. Onder zijn werkdefinitie telt een gebeurtenis die voor de meeste mensen grotendeels onverwacht was als zwarte zwaan. Zie *Staartgebeurtenissen en zwarte zwanen* in *Het idee*.",
            "yes": "Het red team zag het wel aankomen, dus voor dat team kwam het met een waarschuwing. Zwarte zwanen hangen af van wie er kijkt. Zie *Staartgebeurtenissen en zwarte zwanen* in *Het idee*.",
        },
    )
    choice_whose_swan
    return (choice_whose_swan,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Safeguards achter elkaar](/course/open/rai-swiss-cheese?lang=nl) eindigde met wat zijn modellen weglaten, en [Willekeurige getallen en seeds](/course/open/np-random?lang=nl) gaf de generator die de staarten in deze les trok. Deze les voegde drift into failure toe, met het schuim van de Columbia als gedocumenteerd geval, en staartgebeurtenissen, zwarte zwanen en staartrisico, waarbij één trekking zwaarder kan wegen dan duizenden. De volgende les, *Toxiciteit meten*, kijkt naar een safeguard die veel chatbots gebruiken: een model dat een score geeft voor hoe schadelijk een tekst is.

    **Een voorbeeld uit het dagelijks leven.** Een brug die elke winter wat meer kraakt en elke dag zonder problemen gebruikt wordt, heeft niet laten zien dat hij veilig is: elke winter zonder instorting is weer een stap van dezelfde drift.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_drift, choice_whose_swan, result_mean, result_stretch):
    L.wrap_up(result_stretch, result_mean, choice_drift, choice_whose_swan)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
