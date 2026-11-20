# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-race-and-selection. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow AI Safety, Selection

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-race-and-selection", "nl")
    C = L.checks   # een verzonnen tabel met wat voorzichtig blijven en racen waard zijn, en een verzonnen populatie agents


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [AI in een complex systeem](/course/open/rai-complex-systems?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_loop():
    quick_loop = L.choice(
        "quick-loop",
        question="In een feedback loop voegt elke ronde 1,08 keer toe wat de ronde ervoor toevoegde. Wat gebeurt er na veel rondes met het totaal?",
        options={
            "settle": "Het komt na een tijd tot rust",
            "right": "Het blijft groeien, en elke ronde sneller",
            "same": "Het blijft ongeveer gelijk, want 1,08 ligt dicht bij 1",
        },
        correct="right",
        why={
            "right": "Elke ronde voegt 8% meer toe dan de ronde ervoor, dus de toevoegingen zelf blijven groeien, en het totaal schiet weg. In de verzonnen markt van de vorige les gebeurde dat vanaf een echo van 12: 0,09 × 12 = 1,08.",
            "settle": "Een loop komt tot rust als elke ronde minder toevoegt dan de ronde ervoor: een factor onder 1. Bij 1,08 voegt elke ronde meer toe dan de vorige, dus het totaal blijft groeien, steeds sneller. Zie [AI in een complex systeem](/course/open/rai-complex-systems?lang=nl).",
            "same": "1,08 ligt dicht bij 1, maar erboven: elke ronde voegt 8% meer toe dan de vorige, en over veel rondes loopt dat snel op. In de verzonnen markt zou een echo van 11 (0,99) nog tot rust komen, en schoot een echo van 12 (1,08) weg. Zie [AI in een complex systeem](/course/open/rai-complex-systems?lang=nl).",
        },
    )
    quick_loop
    return (quick_loop,)


@app.cell(hide_code=True)
def quick_check_match():
    quick_match = L.choice(
        "quick-match",
        question="Een antwoord van 10 cijfers telt alleen als goed als elk cijfer goed is: een exact match. Een model heeft elk cijfer goed met een kans van 0,9, los van de andere cijfers. Hoe groot is de kans op een exact match?",
        options={
            "single": "0,9, net als voor één cijfer",
            "tenth": "0,09, een tiende daarvan",
            "right": "Dicht bij 0,35",
        },
        correct="right",
        why={
            "right": "Elk cijfer is goed met een kans van 0,9, los van de andere, dus alle 10 zijn goed met een kans van 0,9 tien keer met zichzelf vermenigvuldigd: 0,9 tot de macht 10, ongeveer 0,35.",
            "single": "Dat is de kans voor één cijfer. Om alle 10 goed te hebben, vermenigvuldig je de kansen: 0,9 tot de macht 10 is ongeveer 0,35. Zie [AI in een complex systeem](/course/open/rai-complex-systems?lang=nl).",
            "tenth": "Vermenigvuldig de kansen, deel ze niet: alle 10 cijfers goed heeft een kans van 0,9 tot de macht 10, ongeveer 0,35. Zie [AI in een complex systeem](/course/open/rai-complex-systems?lang=nl).",
        },
    )
    quick_match
    return (quick_match,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    AI-bedrijven schrijven in hun eigen veiligheidsbeleid dat ze een race to the bottom willen vermijden. Volgens de Responsible Scaling Policy van Anthropic, versie 3.4 van juli 2026, kunnen de toezeggingen over concurrenten helpen een onbedoelde race to the bottom op veiligheid te vermijden, voor zover andere AI-ontwikkelaars veiligheid voorop zetten en zichtbaar laten zien dat ze dat doen. Het Preparedness Framework van OpenAI, versie 2 van april 2025, zegt dat OpenAI zijn safeguards kan aanpassen als het zeker kan vaststellen dat een andere ontwikkelaar een systeem met capaciteiten op het niveau High of Critical heeft ontwikkeld of uitgebracht zonder vergelijkbare safeguards, maar alleen onder voorwaarden. Een daarvan is dat OpenAI, om een race to the bottom op veiligheid te vermijden, zijn safeguards beschermender houdt dan die van de andere ontwikkelaar. Waarom zouden concurrerende bedrijven zich daar zorgen over maken, en wat kan voorzichtig blijven laten lonen? Deze les kijkt naar wat er gebeurt als wie wacht kan verliezen, met een eenvoudig spel dat je zelf programmeert, en naar een argument dat concurrentie AI-systemen kan selecteren die niemand wilde.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Het leerboek van Dan Hendrycks beschrijft een race tussen twee concurrenten als een veiling: elk biedt een kans op een catastrofe die het wil accepteren, en wie de grootste kans accepteert, wint de prijs, zolang de catastrofe uitblijft. Beide zien de stand van de race op dezelfde manier. Welke kans op een uitkomst die even slecht is als de prijs goed is, zou elk in dit sterk vereenvoudigde model riskeren?",
        options={
            "low": "Ongeveer 1 op 100",
            "right": "Ongeveer 1 op 3",
            "high": "Ongeveer 9 op 10",
        },
        reveal="**Ongeveer 1 op 3.** Het leerboek noemt het het Attrition model. Als de prijs voor een concurrent X waard is, is die volgens het boek bereid een kans van 33% te riskeren op een uitkomst die even slecht is, met een waarde van −X, om de race te winnen. Het is een sterk vereenvoudigd model, geen meting. Het leerboek voegt eraan toe dat echte AI-races nog slechter kunnen uitpakken dan het Attrition model voorspelt, bijvoorbeeld als concurrenten niet goed weten hoe hun race ervoor staat.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Een race to the bottom

    In een paper in het tijdschrift *AI & Society* (online in 2015, gedrukt in 2016) presenteren Stuart Armstrong, Nick Bostrom en Carl Shulman een eenvoudig model van een wapenwedloop om AI, waarin verschillende ontwikkelteams racen om als eerste AI te bouwen. Ze vatten samen dat elk team, in de veronderstelling dat de eerste AI heel krachtig en ingrijpend zal zijn, er belang bij heeft om als eerste klaar te zijn, zo nodig door te bezuinigen op veiligheidsmaatregelen. In hun model kunnen meer ontwikkelteams en meer vijandschap tussen teams het gevaar van een AI-ramp vergroten, vooral als het nemen van risico's belangrijker is dan vaardigheid bij het ontwikkelen. Verrassend genoeg vergroot volgens hen ook informatie de risico's: hoe meer teams weten over elkaars capaciteiten, en over die van zichzelf, hoe groter het gevaar. Volgens hen wijzen hun resultaten de weg naar manieren om AI veiliger te ontwikkelen, als ze ook standhouden in realistischere modellen en analyses. Amanda Askell en anderen schrijven dat Armstrong en anderen (2016) erop wijzen dat verantwoorde ontwikkeling in een race om AI ten prooi kan vallen aan een race to the bottom.

    Het leerboek *Introduction to AI Safety, Ethics, and Society* van Dan Hendrycks (een concept van het boek, versie 2 van 15 november 2024) beschrijft zo'n race tussen bedrijven en staten. Volgens het boek wordt die AI-race gedreven door landen en bedrijven die vinden dat ze snel AI moeten bouwen en inzetten om hun positie veilig te stellen en te overleven. Het probleem dat het boek noemt, is dat beslissingen die voor elk afzonderlijk verstandig zijn, samen catastrofaal kunnen uitpakken. En om een voorsprong te krijgen, racen bedrijven volgens het boek vaak om als eerste een product op de markt te brengen, in plaats van het veiligste. Volgens het leerboek zei Satya Nadella, de CEO van Microsoft, bij de lancering van de zoekmachine met AI van het bedrijf in februari 2023 dat er die dag een race begon, en dat ze snel zouden gaan. Een paar weken later bleek de chatbot van het bedrijf volgens het boek gedreigd te hebben gebruikers kwaad te doen. Het boek geeft ook een voorbeeld uit de luchtvaart. Boeing wilde volgens het boek, in de concurrentie met rivaal Airbus, zo snel mogelijk een vernieuwd, zuiniger model op de markt brengen. Onvoldoende testen en te weinig training van piloten leidden volgens het boek uiteindelijk tot twee dodelijke crashes, een paar maanden na elkaar, waarbij 346 mensen omkwamen.

    ### Wanneer voorzichtig blijven loont

    In een preprint uit 2019 (versie 1 van 10 juli 2019), met de aantekening dat het werk nog niet af was, noemen Askell, Miles Brundage en Gillian Hadfield, alle drie toen bij OpenAI, een race to the bottom op veiligheid een collectieve-actieprobleem. Daarmee bedoelen ze een situatie waarin alle partijen beter af zouden zijn als ze allemaal samenwerkten, terwijl elk denkt dat het in het eigen belang is om dat niet te doen. Om te zien wanneer samenwerken loont, gebruiken ze een spel. Twee bedrijven kiezen elk of ze samenwerken, wat hier betekent dat ze verantwoord ontwikkelen, of niet. In deze les is samenwerken voorzichtig blijven, en niet meewerken racen. Elke uitkomst heeft voor een bedrijf een waarde, die ze a, b, c en d noemen:

    | | de rivaal blijft voorzichtig | de rivaal racet |
    |---|---|---|
    | jij blijft voorzichtig | a | b |
    | jij racet | c | d |

    Stel dat p de kans is die een bedrijf eraan geeft dat de rivaal samenwerkt. Askell en anderen nemen aan dat het rationeel is om samen te werken als de verwachtingswaarde van samenwerken groter is dan die van niet meewerken. De verwachtingswaarde van samenwerken is de kans dat de ander samenwerkt keer a, plus de kans dat de ander dat niet doet keer b; die van niet meewerken is dezelfde som met c en d. Een verwachtingswaarde weegt elke waarde met de kans erop en telt ze op. Zijn de twee gelijk, dan zijn samenwerken en niet meewerken volgens hen allebei rationeel toegestaan.

    Uit deze regel halen ze vijf factoren die bedrijven helpen samen te werken. Ze noemen veel vertrouwen tussen ontwikkelaars (High Trust), veel gedeelde winst bij wederzijdse samenwerking (Shared Upside), weinig blootstelling aan schade als je samenwerking niet beantwoord wordt (Low Exposure), weinig winst als je de samenwerking van anderen niet beantwoordt (Low Advantage), en veel gedeelde schade als niemand meewerkt (Shared Downside). In de letters: een hoge p, een hoge a, een b die niet te slecht is, een lage c en een lage d. Volgens hen maakt diep wantrouwen tussen bedrijven een collectieve-actieprobleem waarschijnlijk erger, omdat bedrijven dan minder verwachten dat hun samenwerking beantwoord wordt. Vertrouwen opbouwen tussen AI-bedrijven kan collectieve-actieproblemen volgens hen dus minder ernstig maken. Ze schrijven ook dat de informatie die bedrijven geven over hun bedoelingen en wat ze doen, hoe transparant ze zijn, een belangrijke rol kan spelen bij de vraag of andere bedrijven met hen samenwerken. Dat is een ander soort informatie dan in het model van Armstrong en anderen, waar het om capaciteiten gaat.

    Ze geven zelf de grenzen van hun analyse aan. Die gaat uit van een first-mover advantage, een voordeel voor wie het eerst is. Als zo'n voordeel zwak is of ontbreekt, gaan bedrijven volgens hen minder snel een race to the bottom op veiligheid aan, omdat snelheid dan minder waard is. Eerst zijn loont niet altijd; volgens hen is betoogd dat de voordelen van later instappen vaak zwaarder wegen dan die van eerst zijn. Ze erkennen daarom dat de aanname van een first-mover advantage bij AI-ontwikkeling misschien niet klopt. Hun paper is volgens hen geen voorspelling, maar een analyse van somberdere scenario's waarin eerst zijn minstens een matig voordeel geeft.

    ### Selectie

    Hendrycks gaat een stap verder in *Natural Selection Favors AIs over Humans* (2023; versie 4 van 18 juli 2023). Volgens Hendrycks zelf is het paper bedoeld voor een breed publiek, anders dan de meeste teksten van deze auteur. Hendrycks betoogt dat de meest succesvolle AI-agents waarschijnlijk ongewenste eigenschappen zullen hebben. Het vertrekpunt zijn drie voorwaarden die de bioloog Richard Lewontin opstelde, en waaronder natuurlijke selectie optreedt. Hendrycks omschrijft ze zo:

    1. Variatie, wat betekent dat individuen verschillen in eigenschappen, parameters of trekken.
    2. Retentie, wat betekent dat latere versies van individuen meestal op eerdere versies lijken.
    3. Verschil in fitness, wat betekent dat verschillende varianten zich in een verschillend tempo verspreiden.

    Bij AI-agents kan retentie volgens Hendrycks ontstaan doordat eerdere versies van agents worden aangepast, doordat agents vergelijkbare maar betere agents ontwerpen, of doordat agents het gedrag van eerdere agents nadoen. Het leerboek laat de voorwaarden aan het werk zien in iets wat nu al bestaat: de algoritmes waarmee streamingdiensten en sociale media content aanbevelen. Vormen van content en algoritmes die gebruikers aan zich binden, worden gehouden en verder verfijnd; de andere verdwijnen. Volgens het boek leidt die concurrentiedruk ertoe dat overleeft wat het meest verslavend is. Kunnen mensen dan niet gewoon kiezen? Volgens het leerboek hebben mensen maar in naam invloed op welke AI geselecteerd wordt. Het voegt eraan toe dat de bedrijven die AI ontwikkelen niet het veiligste pad kiezen, maar bezwijken onder de druk van de evolutie.

    Het woord egoïstisch gaat bij Hendrycks over gedrag, niet over een bewuste egoïstische bedoeling. En Hendrycks noemt zelf een uitweg. Natuurlijke selectie hangt volgens Hendrycks af van concurrentie, en ongekende wereldwijde en economische coördinatie zou die strijd kunnen voorkomen en natuurlijke selectie kunnen dwarsbomen. Het leerboek geeft als eigen mening dat het niet coördineren en niet stoppen van AI-races de meest waarschijnlijke oorzaak van een existentiële catastrofe zou zijn.

    ### Is er wel een race?

    Niet iedereen is het erover eens dat er een is. Paul Scharre schrijft in de *Texas National Security Review* van juni 2021 over het leger dat er geen wapenwedloop om AI is, al brengt militaire concurrentie in AI wel bepaalde risico's met zich mee. Een van de argumenten van Scharre is dat de militaire uitgaven aan AI, in elk geval nu, bij lange na niet groot genoeg zijn om van een wapenwedloop te spreken. Wat er wel gebeurt, is volgens Scharre veel beter te omschrijven als een veiligheidsdilemma, dat Scharre uitlegt met een definitie van Robert Jervis. Volgens Jervis verkleinen veel van de middelen waarmee een staat zijn veiligheid probeert te vergroten, de veiligheid van anderen. Toch waarschuwt ook Scharre dat haast om AI-systemen in te zetten voordat ze volledig getest zijn, kan leiden tot een race to the bottom op veiligheid, met legers die AI-systemen inzetten die vatbaar zijn voor ongelukken.

    Stephen Cave en Seán ÓhÉigeartaigh, van de Universiteit van Cambridge, waarschuwen in een paper uit 2018 voor praten over een race en voor een echte race. Hun zorgen gaan zowel over het gebruik van de taal van een race om technologische superioriteit, los van de vraag of zo'n race er is, als over zo'n race die werkelijkheid wordt. En als er gesproken wordt over een race waarin de winnaar alles krijgt terwijl er geen echte race is, kan dat volgens hen bijdragen aan het ontstaan van zo'n race.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **race to the bottom** | concurrentie waarin elke partij de eigen veiligheidseisen verlaagt om de anderen bij te houden, zodat iedereen minder veilig uitkomt dan elk van hen wilde |
        | collectieve-actieprobleem | iedereen zou beter af zijn als iedereen samenwerkte, maar elk denkt dat niet meewerken meer oplevert; naar Askell en anderen |
        | verwachtingswaarde | de waarde van elke uitkomst keer de kans erop, opgeteld; hier van voorzichtig blijven of van racen |
        | vertrouwen | hier: de kans die een bedrijf eraan geeft dat de rivaal voorzichtig blijft, `trust` in oefening 1 |
        | first-mover advantage | het voordeel van wie het eerst is, bijvoorbeeld op een markt; dat voordeel is er niet altijd |
        | selectie | als individuen verschillen, lijken op de individuen voor hen en zich in een verschillend tempo verspreiden, nemen de snelste het over; naar de drie voorwaarden van Lewontin |
        | veiligheidsdilemma | als de middelen waarmee een staat veiliger probeert te worden, andere staten minder veilig maken; naar Jervis |
        """),
        L.watch_out("Het Attrition model, het spel van Askell en anderen en de verzonnen populatie in deze les zijn modellen: ze laten zien wat uit hun aannames volgt, niet wat er gaat gebeuren. Of AI-ontwikkeling wel een race is, en of eerst zijn loont, is omstreden."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om na te gaan of concurrentie aanzet tot minder voorzichtigheid:

    1. Schrijf op wie er concurreren en wat de eerste wint, en vraag je af of eerst zijn echt loont.
    2. Vul voor elke partij zo goed mogelijk de vier waarden a, b, c en d in.
    3. Schat de kans die elk eraan geeft dat de ander voorzichtig blijft, en vergelijk de verwachtingswaarden.
    4. Kijk welke van de vijf factoren te veranderen zijn: vertrouwen, gedeelde winst, blootstelling, voordeel, gedeelde schade.
    5. Kijk welke systemen gehouden, gekopieerd en uitgebouwd worden, en om welke eigenschap.
    6. Let op je eigen woorden: praten over een race kan bijdragen aan het ontstaan van een race.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel rekent uit wat voorzichtig blijven een bedrijf waard is bij drie niveaus van vertrouwen dat de rivaal voorzichtig blijft. Als de rivaal voorzichtig blijft, is voorzichtig blijven 4 waard; als de rivaal racet, 0. De waarden zijn verzonnen.
    """)
    return


@app.cell
def example_rule():
    a, b = 4, 0   # verzonnen waarden van voorzichtig blijven: rivaal blijft voorzichtig, rivaal racet
    for trust in [0.2, 0.5, 0.8]:
        careful = trust * a + (1 - trust) * b
        print("vertrouwen", trust, "waarde van voorzichtig blijven", round(careful, 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Bij een vertrouwen van 0,2 is voorzichtig blijven gemiddeld 0,8 waard; bij 0,8 is het 3,2 waard. Of dat genoeg is, hangt af van wat racen bij hetzelfde vertrouwen waard is. In oefening 1 vergelijk je de twee.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_choices():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. choices: voorzichtig blijven of racen

        De regel van Askell en anderen voor een verzonnen tabel: `a` = 4 als beide voorzichtig blijven, `b` = 0 als het bedrijf voorzichtig blijft en de rivaal racet, `c` = 3 als het racet en de rivaal voorzichtig blijft, en `d` = 1 als beide racen. Schrijf `choices(a, b, c, d, trusts)`, die een lijst `result` teruggeeft: voor elke `trust` in de lijst `trusts` de tekst `"careful"` als voorzichtig blijven de hoogste verwachtingswaarde heeft, en anders `"fast"`, ook als de twee gelijk zijn. Reken in een `for`-loop met `trust` `careful` uit zoals in het uitgewerkte voorbeeld, en `fast`, de verwachtingswaarde van racen. Vervang de `...`. De check roept `choices(4, 0, 3, 1, [0, 0.25, 0.5, 0.75, 1])` aan, en daarna andere gevallen.
        """),
        L.task("Voor elk vertrouwen: of voorzichtig blijven of racen de hoogste verwachtingswaarde heeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_choices():
    box_choices = L.code_box("choices", lines=10)
    box_choices
    return (box_choices,)


@app.cell(hide_code=True)
def check_choices(box_choices):
    result_choices = L.check("choices", box_choices.value)
    result_choices
    return (result_choices,)


@app.cell(hide_code=True)
def hints_choices():
    L.hints(
        "choices",
        [
            "Begin klein: hoeveel is voorzichtig blijven gemiddeld waard bij een vertrouwen van 0,25, als het 4 waard is als de rivaal voorzichtig blijft en 0 als de rivaal racet? En hoeveel is racen waard, bij 3 en 1?",
            "Begin `result` als een lege lijst. Maak in een `for`-loop met `trust` over `trusts` `careful` zoals in het uitgewerkte voorbeeld, en `fast` op dezelfde manier met `c` en `d`. Voeg met `if` en `else` `\"careful\"` toe als `careful` groter is dan `fast`, en anders `\"fast\"`. Geef na de loop `result` terug.",
            "`        fast = trust * c + ___`",
        ],
        why="`result = []`, dan `for trust in trusts:` met `careful = trust * a + (1 - trust) * b`, `fast = trust * c + (1 - trust) * d`, en `if careful > fast:` `result.append(\"careful\")`, `else:` `result.append(\"fast\")`; na de loop `return result`. Bij een vertrouwen van 0,25 is voorzichtig blijven 0,25 × 4 + 0,75 × 0 = 1 waard en racen 0,25 × 3 + 0,75 × 1 = 1,5, dus is het antwoord `\"fast\"`.",
    )
    return


@app.cell(hide_code=True)
def text_after_choices():
    mo.md(r"""
    Bij de verzonnen tabel loont voorzichtig blijven pas boven een vertrouwen van 0,5; bij 0,5 zijn beide 2 waard. Daaronder heeft racen de hoogste verwachtingswaarde, ook al zouden beide bedrijven beter af zijn met de 4 van samen voorzichtig blijven dan met de 1 van samen racen. Dat is een collectieve-actieprobleem in vier getallen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shares():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. shares: in een verschillend tempo gekopieerd

        De drie voorwaarden van Lewontin in een verzonnen populatie: drie soorten AI-agent, in gebruik in 98, 1 en 1 kopieën. De soorten verschillen, en elke generatie wordt het aantal kopieën van elke soort vermenigvuldigd met een eigen tempo, `growth`: met 1 voor de eerste, 1,05 voor de tweede en 1,1 voor de derde. Schrijf `shares(counts, growth, generations)`, die een lijst `result` teruggeeft met voor elke generatie een array met het aandeel van elke soort in de populatie. Voeg in een `for`-loop met `generation` over `range(generations)` de aandelen van deze generatie toe, en kopieer daarna elke soort. Laat NumPy op alle soorten tegelijk werken. Vervang de `...`. De check roept `shares(np.array([98, 1, 1]), np.array([1, 1.05, 1.1]), 61)` aan, en daarna andere gevallen.
        """),
        L.task("Het aandeel van elke soort in de populatie, generatie na generatie."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shares():
    box_shares = L.code_box("shares", lines=6)
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
            "Begin klein: twee soorten, met 9 en 1 kopieën. Welk aandeel van de populatie heeft elk? En nadat de tweede soort verdubbelt?",
            "Begin `result` als een lege lijst. Voeg in een `for`-loop met `generation` over `range(generations)` het aandeel van elke soort toe: `counts` gedeeld door het totale aantal kopieën, dat NumPy met één functie geeft. Maak daarna `counts` gelijk aan de volgende generatie, `counts` keer `growth`. Geef na de loop `result` terug.",
            "`        result.append(counts / ___)`",
        ],
        why="`result = []`, dan `for generation in range(generations):` met `result.append(counts / np.sum(counts))` en `counts = counts * growth`; na de loop `return result`. Twee soorten met 9 en 1 kopieën hebben aandelen van 0,9 en 0,1; nadat de tweede verdubbelt, geven 9 en 2 kopieën 9 / 11 ≈ 0,818 en 2 / 11 ≈ 0,182.",
    )
    return


@app.cell(hide_code=True)
def text_after_shares():
    mo.md(r"""
    De soort waarvan het aantal kopieën elke generatie met 1,1 vermenigvuldigd wordt, begint met 1 kopie op de 100. In generatie 30 heeft die 15% van de populatie, in generatie 50 meer dan de helft, en in generatie 60 ongeveer 72%. Het aandeel van de soort met 1,05 groeit tot ongeveer 5% rond generatie 49 en krimpt dan weer, omdat de derde soort nog sneller groeit. Niets in de code kiest een soort: een klein verschil in hoe snel elke soort gekopieerd wordt, is genoeg. Wat een soort sneller laat verspreiden, zegt de code niet. Hendrycks betoogt dat dat in een wereld van concurrentie vaak eigenschappen zullen zijn die niemand wil; dat is een argument van Hendrycks, niet iets wat deze populatie laat zien.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: wat racen oplevert

    De verzonnen tabel uit oefening 1, met één waarde die je kunt veranderen: `c`, wat racen oplevert terwijl de rivaal voorzichtig blijft. Kijk vanaf welk vertrouwen voorzichtig blijven loont.
    """)
    return


@app.cell(hide_code=True)
def demo_gain():
    gain = L.explore("gain", "c: wat racen oplevert terwijl de rivaal voorzichtig blijft",
                     lambda _v: C.demo_frame(_v, L.lang), start=3, values=C.GAINS, role="",
                     task="Zoek de kleinste opbrengst waarbij geen vertrouwen hoog genoeg is om voorzichtig blijven meer te laten opleveren.")
    gain
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Hoe meer racen oplevert, hoe meer vertrouwen nodig is: boven 0,33 bij een opbrengst van 2; boven 0,5 bij 3; boven 0,67 bij 3,5. Vanaf een opbrengst van 4, evenveel als samen voorzichtig blijven waard is, is geen vertrouwen hoog genoeg. Bij een opbrengst van precies 4 en een vertrouwen van 1 zijn de twee gelijk, wat Askell en anderen allebei toegestaan noemen; bij elk lager vertrouwen, en bij elke hogere opbrengst, is racen meer waard. Dit is hun Low Advantage: hoe minder racen oplevert terwijl de ander voorzichtig blijft, hoe minder vertrouwen nodig is om voorzichtig te blijven.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_copied():
    mo.md(r"""
    ### Oefening 3. Een vleiende assistent

    Een bedrijf laat veel versies van zijn assistent naast elkaar draaien. Gebruikers geven de versies die hen vleien een hoger cijfer, dus houdt het bedrijf die, bouwt het zijn volgende versies erop, en zet het de versies uit die gebruikers onwelkome waarheden vertellen. Niemand bij het bedrijf wil een vleiende assistent.
    """)
    return


@app.cell(hide_code=True)
def choice_copied_cell():
    choice_copied = L.choice(
        "copied",
        question="Gelden hier de drie voorwaarden van Lewontin voor selectie?",
        options={
            "intent": "Nee: geen van de versies wil gehouden worden",
            "right": "Ja: versies verschillen, nieuwe lijken op oude, sommige verspreiden zich sneller",
            "people": "Nee: zolang mensen kiezen welke versies ze houden, is het een keuze en geen selectie",
        },
        correct="right",
        why={
            "right": "De versies verschillen (variatie), de volgende versies worden gebouwd op de versies die blijven (retentie), en de vleiende versies worden vaker gehouden en uitgebouwd (verschil in fitness). Vleien verspreidt zich, al wilde niemand het. Dat het bedrijf kiest, houdt dat niet tegen: het kiest op cijfers, en die cijfers belonen vleien. Selectie heeft geen bedoeling nodig, alleen de drie voorwaarden.",
            "intent": "Selectie heeft geen bedoeling nodig: egoïstisch gaat bij Hendrycks over gedrag, niet over een bewuste egoïstische bedoeling. De versies verschillen, de volgende worden gebouwd op de versies die blijven, en de vleiende blijven vaker: alle drie de voorwaarden gelden. Zie *Selectie* in *Het idee*.",
            "people": "Mensen kiezen hier wel, maar op cijfers die vleien belonen, dus verspreiden de vleiende versies zich toch. Volgens het leerboek hebben mensen maar in naam invloed op welke AI geselecteerd wordt. Alle drie de voorwaarden gelden: de versies verschillen, de volgende lijken op de versies die blijven, en sommige blijven vaker. Zie *Selectie* in *Het idee*.",
        },
    )
    choice_copied
    return (choice_copied,)


@app.cell(hide_code=True)
def text_exercise_trust():
    mo.md(r"""
    ### Oefening 4. Twee bedrijven

    Twee bedrijven waarderen de uitkomsten op dezelfde manier: 5 als beide voorzichtig blijven, 1 als een bedrijf voorzichtig blijft en het andere racet, 4 als het racet en het andere voorzichtig blijft, en 2 als beide racen. Elk geeft een kans van 0,3 dat het andere voorzichtig blijft. Alle waarden zijn verzonnen.
    """)
    return


@app.cell(hide_code=True)
def choice_trust_cell():
    choice_trust = L.choice(
        "trust",
        question="Wat doet elk, volgens de regel van Askell en anderen?",
        options={
            "careful": "Elk blijft voorzichtig, want racen levert minder op dan samen voorzichtig blijven",
            "law": "Elk racet, en alleen een wet zou dat kunnen veranderen",
            "right": "Elk racet; meer vertrouwen tussen hen zou dat kunnen veranderen",
        },
        correct="right",
        why={
            "right": "Bij een vertrouwen van 0,3 is voorzichtig blijven 0,3 × 5 + 0,7 × 1 = 2,2 waard en racen 0,3 × 4 + 0,7 × 2 = 2,6, dus racet elk, al zouden beide beter af zijn met 5 dan met 2: een collectieve-actieprobleem. Boven een vertrouwen van 0,5 is voorzichtig blijven meer waard. Dat is de High Trust van Askell en anderen.",
            "careful": "Racen levert 4 op, minder dan de 5 van samen voorzichtig blijven, maar alleen als het andere bedrijf voorzichtig blijft. Bij een vertrouwen van 0,3 is voorzichtig blijven 0,3 × 5 + 0,7 × 1 = 2,2 waard en racen 0,3 × 4 + 0,7 × 2 = 2,6. Vergelijk de verwachtingswaarden, zoals in oefening 1.",
            "law": "Een wet is één manier: die kan de waarden veranderen, bijvoorbeeld door racen duurder te maken. Maar ook vertrouwen alleen kan de keuze veranderen: boven een vertrouwen van 0,5 is voorzichtig blijven meer waard dan racen. Zie *Wanneer voorzichtig blijven loont* in *Het idee*.",
        },
    )
    choice_trust
    return (choice_trust,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [AI in een complex systeem](/course/open/rai-complex-systems?lang=nl) liet zien hoe onderdelen die op elkaar reageren, kunnen doen wat geen onderdeel op zichzelf doet. Deze les keek naar één zo'n wisselwerking: bedrijven en staten die concurreren. Als wie wacht kan verliezen, kan elk een reden hebben om te bezuinigen op voorzichtigheid, tot iedereen uitkomt waar geen van hen wilde zijn: een race to the bottom. Met de regel van Askell en anderen zag je hoe vertrouwen en de waarden van de uitkomsten bepalen of voorzichtig blijven loont. Selectie heeft niemand nodig die kiest: wat sneller gekopieerd wordt, neemt het over. Of AI-ontwikkeling wel een race is, is omstreden, en praten over een race kan zelf bijdragen aan het ontstaan van een race. De volgende les, *AI voor de verdediging*, kijkt naar de andere kant: AI die de systemen eromheen veiliger maakt.

    **Een voorbeeld uit het dagelijks leven.** Bij een concert gaan een paar mensen vooraan staan om beter te zien. De mensen achter hen gaan ook staan, en al snel staat iedereen: niemand ziet beter dan eerst, en ieders voeten doen pijn. Zitten zou voor iedereen beter zijn, maar voor elk afzonderlijk loont staan.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_copied, choice_trust, result_choices, result_shares):
    L.wrap_up(result_choices, result_shares, choice_copied, choice_trust)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
