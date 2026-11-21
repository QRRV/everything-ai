# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-ai-for-defense. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow AI Safety, Journal

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-ai-for-defense", "nl")
    C = L.checks   # verzonnen software met 10 kwetsbaarheden, naar het model van Garfinkel en Dafoe


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Races en selectie](/course/open/rai-race-and-selection?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_worth():
    quick_worth = L.choice(
        "quick-worth",
        question="In het spel van Askell en anderen is racen voor een bedrijf 3 waard als de rivaal voorzichtig blijft, en 1 als de rivaal racet. Het vertrouwen van het bedrijf dat de rivaal voorzichtig blijft, is 0,5. Wat is racen gemiddeld waard voor het bedrijf?",
        options={
            "half": "1,5, de helft van 3",
            "right": "2",
            "sum": "4, de twee waarden opgeteld",
        },
        correct="right",
        why={
            "right": "Weeg elke waarde met de kans erop en tel ze op: 0,5 × 3 + 0,5 × 1 = 2. Dat is de verwachtingswaarde van racen, zoals in de regel van Askell en anderen.",
            "half": "Dan laat je weg wat racen waard is als de rivaal racet. Weeg beide waarden met hun kans: 0,5 × 3 + 0,5 × 1 = 2. Zie [Races en selectie](/course/open/rai-race-and-selection?lang=nl).",
            "sum": "Er gebeurt maar een van de twee uitkomsten, dus optellen telt te veel. Weeg elke waarde met de kans erop: 0,5 × 3 + 0,5 × 1 = 2. Zie [Races en selectie](/course/open/rai-race-and-selection?lang=nl).",
        },
    )
    quick_worth
    return (quick_worth,)


@app.cell(hide_code=True)
def quick_check_kinds():
    quick_kinds = L.choice(
        "quick-kinds",
        question="Twee soorten AI-agent beginnen met hetzelfde aantal kopieën. Elke generatie wordt het aantal kopieën van de ene soort met 1,1 vermenigvuldigd, dat van de andere met 1. Wat gebeurt er over veel generaties met het aandeel van de eerste soort?",
        options={
            "half": "Het blijft de helft",
            "right": "Het blijft groeien, richting de hele populatie",
            "once": "Het groeit één keer, tot 1,1 keer de helft, en blijft dan gelijk",
        },
        correct="right",
        why={
            "right": "De eerste soort krijgt er elke generatie 10% bij en de tweede niets, dus het aandeel van de eerste blijft stijgen richting 1, zoals bij de soort met 1,1 in de verzonnen populatie van de vorige les.",
            "half": "Beide soorten houden hun kopieën, maar de eerste krijgt er elke generatie 10% bij en de tweede niets, dus het aandeel van de eerste blijft groeien. Zie [Races en selectie](/course/open/rai-race-and-selection?lang=nl).",
            "once": "Het vermenigvuldigen gebeurt elke generatie, niet één keer: na twee generaties heeft de eerste soort 1,1 × 1,1 keer zoveel kopieën, enzovoort, dus het aandeel blijft groeien. Zie [Races en selectie](/course/open/rai-race-and-selection?lang=nl).",
        },
    )
    quick_kinds
    return (quick_kinds,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Het International AI Safety Report 2026 is de tweede editie van een rapport dat experts schreven onder voorzitterschap van Yoshua Bengio, en verscheen in februari 2026. Volgens het rapport is nu goed onderbouwd dat criminele groepen en aanvallers met steun van een staat AI actief gebruiken bij hun cyberoperaties. Maar of AI-systemen cyberaanvallen in het algemeen omvangrijker of ernstiger hebben gemaakt, is volgens het rapport nog onzeker, omdat oorzaak en gevolg moeilijk vast te stellen zijn. Een AI-ontwikkelaar meldde in november 2025 volgens het rapport dat een aanvaller de modellen van die ontwikkelaar had gebruikt om 80 tot 90% van het werk van een inbraak te automatiseren, terwijl mensen alleen op cruciale beslismomenten ingrepen. Hetzelfde rapport schrijft ook dat veel van dezelfde AI-capaciteiten die bij cyberaanvallen gebruikt worden, ook de verdediging kunnen versterken, zodat onzeker is of AI aanvallers of verdedigers meer oplevert. In een blogpost van 1 november 2024 meldde een team van Google dat de AI-agent Big Sleep een misbruikbare kwetsbaarheid had gevonden in SQLite, een veelgebruikte database-engine, voordat die in een officiële release terechtkwam. Helpt AI wie aanvalt meer dan wie verdedigt? Deze les kijkt naar AI die de systemen eromheen veiliger maakt, en naar een eenvoudig model, dat je zelf programmeert, van hoe de balans tussen aanval en verdediging kan veranderen als het gereedschap beter wordt.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In de finale van de AI Cyber Challenge van DARPA, waarvan DARPA de uitslag op 8 augustus 2025 bekendmaakte, lieten teams hun AI-systemen zoeken in open-source software waarin de organisatoren 63 kwetsbaarheden hadden gestopt. Hoeveel daarvan vonden de systemen, alle teams samen?",
        options={
            "quarter": "Ongeveer 1 op 4",
            "half": "Ongeveer de helft ervan",
            "right": "Ongeveer 6 op 7",
        },
        reveal="**Ongeveer 6 op 7.** Volgens DARPA ontdekten de systemen van de deelnemers 54 van de 63 kwetsbaarheden die voor de finale gemaakt waren, en dichtten ze daarvan 43. Bij de halve finale in augustus 2024 vonden ze er volgens DARPA 37%. De 63 kwetsbaarheden waren gemaakt voor de wedstrijd: ze laten zien wat de systemen deden op deze test, niet hoeveel kwetsbaarheden AI in alle software vindt.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Systemic safety

    In *Unsolved Problems in ML Safety* (versie 5 van 16 juni 2022) schrijven Dan Hendrycks, Nicholas Carlini, John Schulman en Jacob Steinhardt dat ML-systemen eerder falen of verkeerd gestuurd worden als de grotere omgeving waarin ze werken onveilig of onrustig is. Hun antwoord is een eigen onderzoeksgebied. Onderzoek naar systemic safety zet volgens hen ML in om gevaren uit die omgeving te beperken die ML-systemen beslissend kunnen laten falen of verkeerd kunnen laten sturen. Als twee voorbeelden noemen ze onderzoek naar cybersecurity en naar goed geïnformeerde besluitvorming. Vooruitgang in het vertalen en genereren van code wijst er volgens hen op dat toekomstige modellen patches kunnen aanbrengen en code veiliger kunnen maken, zodat systemen kwetsbaarheden niet alleen aanwijzen, maar ook verhelpen.

    Volgens het leerboek *Introduction to AI Safety, Ethics, and Society* van Hendrycks (een concept van het boek, versie 2 van 15 november 2024) kan AI helpen verdedigen tegen de risico's die AI zelf meebrengt. Het boek noemt drie voorbeelden: verdediging tegen cyberaanvallen, beveiliging tegen pandemieën en een betere informatieomgeving. Tegen cyberaanvallen kan AI volgens het boek ook kwetsbaarheden in code vinden, en zo de verdediging tegen hacken met AI versterken. Tegen pandemieën noemt het boek onder meer het opsporen van epidemieën terwijl ze ontstaan, en het controleren van bestellingen van gensequenties om misbruik te voorkomen. Bij informatie kan AI de samenleving volgens het boek helpen door onjuiste beweringen op te sporen en vragen te beantwoorden met onderbouwde antwoorden. Volgens het boek wordt die aanpak systemic safety genoemd, en heet het bredere idee om je te richten op technologie die beschermt tegen risico's voor de samenleving, soms defensive accelerationism. Het boek geeft ook een grens aan. Volgens het boek is AI niet de enige oplossing, en kan investeren in fysieke infrastructuur en uitrusting nog meer opleveren, zoals betere ventilatie en filtering van de lucht in gebouwen, waardoor virussen die via de lucht gaan moeilijker van de ene persoon op de andere overgaan.

    ### De offense-defense balance

    Ben Garfinkel en Allan Dafoe, van het Future of Humanity Institute van de Universiteit van Oxford, schrijven in het *Journal of Strategic Studies* (2019) dat de offense-defense balance, informeel gezegd, gaat over hoe makkelijk het is om aanvallen uit te voeren vergeleken met je ertegen te verdedigen. Zo'n balans is geen getal voor de hele wereld; volgens hen hangt ze altijd af van een bepaald doel, een bepaalde vorm van conflict en een bepaalde groep betrokkenen. Hun vraag is wat er gebeurt als beide kanten meer investeren. Eenvoudige modellen van grondinvasies en van cyberaanvallen wijzen er volgens hun samenvatting op dat meer investeringen de aanvaller bevoordelen zolang de investeringen laag genoeg zijn, en de verdediger als ze hoog genoeg zijn. Dat noemen ze offensive-then-defensive scaling, of OD-scaling.

    Hun model van cyberaanvallen gaat over zero days: kwetsbaarheden die de verdediger nog niet kent. Een aanvaller krijgt volgens hen een zero day als die een kwetsbaarheid ontdekt die de verdediger niet heeft gevonden. Ze nemen aan dat wat de twee kanten ontdekken niet samenhangt, dat er geen nieuwe kwetsbaarheden bijkomen, en dat elke kant kwetsbaarheden vindt in de maat van de eigen inspanning, tot ze allemaal gevonden zijn. Neem n kwetsbaarheden, en tel inspanning zo dat één eenheid één verwachte ontdekking oplevert. De aanvaller vindt dan het kleinste van n en de eigen inspanning, en elke kwetsbaarheid die de aanvaller vindt, is onbekend voor de verdediger met een kans die gelijk is aan het aandeel dat de verdediger niet heeft gevonden. Het verwachte aantal zero days is het product:

    $$\text{zero days} = \min(n, \text{attack}) \cdot \frac{n - \min(n, \text{defend})}{n}$$

    Waarom meer inspanning eerst de aanvaller helpt en dan de verdediger, leggen ze zo uit. In het begin vinden de twee kanten door toeval zelden precies dezelfde kwetsbaarheden. Maar het aantal kwetsbaarheden is eindig, dus de verdediger raakt volgens hen op den duur verzadigd en vindt dan het grootste deel ervan. Ze passen dat toe op het automatisch vinden van kwetsbaarheden. Als beter gereedschap telt als meer inspanning voor beide kanten, en al het andere gelijk blijft, is het volgens hen aannemelijk dat de balans daardoor op korte termijn verder naar de aanval schuift, en op lange termijn naar de verdediging. Ze noemen hun modellen sterk geïdealiseerd, en zeggen er zelf bij dat ze bewust geen concrete conclusies trekken over het verband tussen deze schaaleffecten en internationale veiligheid. Het Preparedness Framework van OpenAI (versie 2 van april 2025) gebruikt hetzelfde idee bij cybersecurity, een van de gebieden die het framework in de gaten houdt. Volgens het framework kan het wegnemen van knelpunten die kwaadaardige cyberactiviteit beperken, de huidige balans tussen cyberaanval en cyberverdediging verstoren, doordat bestaande cyberaanvallen veel meer geautomatiseerd en op veel grotere schaal uitgevoerd worden.

    ### Twee verdedigers aan het werk

    In die blogpost van 1 november 2024 meldt het Big Sleep-team, van Google Project Zero en Google DeepMind, de eerste kwetsbaarheid in echte software die de Big Sleep-agent ontdekte: een misbruikbare stack buffer underflow in SQLite, een veelgebruikte open-source database-engine. Het team ontdekte de kwetsbaarheid volgens de blogpost begin oktober en meldde die aan de ontwikkelaars, die de fout dezelfde dag herstelden. Het team vond het probleem voordat het in een officiële release stond, zodat gebruikers van SQLite er geen last van hadden. Volgens het team gebruikte de geslaagde run Gemini 1.5 Pro. Het team claimt niet dat dit de eerste fout is die ooit door AI gevonden is; de claim is beperkter. Volgens het team is dit het eerste openbare voorbeeld van een AI-agent die een nog onbekend, misbruikbaar probleem met geheugenveiligheid vond in veelgebruikte echte software. Het team schrijft ook dat eerder dat jaar, bij de AI Cyber Challenge van DARPA, Team Atlanta een null-pointer dereference in SQLite vond. Het argument van het team voor de verdediging is dat aanvallers geen kans krijgen om mee te dingen als kwetsbaarheden gevonden worden voordat de software uitkomt. En het team benadrukt nog eens dat het om zeer experimentele resultaten gaat. Volgens het team is een fuzzer die voor één bepaald doel gemaakt is, op dit moment waarschijnlijk minstens zo goed in het vinden van kwetsbaarheden. Een fuzzer is een programma dat software heel veel willekeurige of aangepaste invoer geeft om die te laten vastlopen.

    De AI Cyber Challenge van DARPA, AIxCC, liep twee jaar. Team Atlanta, van Georgia Tech, Samsung Research, KAIST en POSTECH, won de finale. Volgens DARPA vonden de teams ook 18 echte kwetsbaarheden die niet voor de wedstrijd gemaakt waren, en die op een verantwoorde manier gemeld worden aan de beheerders van de open-source projecten. Teams leverden hun patches volgens DARPA gemiddeld na 45 minuten in, en gaven ongeveer 152 dollar per wedstrijdtaak uit. De systemen worden gedeeld. Volgens DARPA komen de CRS'en van alle zeven finalisten beschikbaar als open-source software, onder een licentie die het Open Source Initiative heeft goedgekeurd. Een CRS, een cyber reasoning system, vindt en dicht kwetsbaarheden zonder dat er iemand aan de knoppen zit. DARPA corrigeerde zichzelf ook. In een eerdere versie van het bericht stond volgens DARPA dat de finale 70 kunstmatige kwetsbaarheden bevatte; na nadere controle stelde de organisatie van de wedstrijd vast dat het er 63 waren. De directeur van DARPA zegt dat de technologie verdedigers een broodnodige voorsprong zal geven bij het snel en op grote schaal vinden en dichten van kwetsbaarheden.

    ### Wie wint er meer?

    Het International AI Safety Report 2026 geeft geen eenduidig antwoord. Het rapport noemt de balans tussen aanval en verdediging cruciaal maar veranderlijk. Volgens het rapport kunnen AI-capaciteiten waarmee een aanvaller snel kwetsbaarheden ontdekt, ook door een verdediger gebruikt worden om ze eerder te vinden en te dichten. Hoe die balans zich ontwikkelt, hangt volgens het rapport deels af van keuzes over toegang tot modellen, geld voor onderzoek en normen voor de inzet van AI. Het rapport wijst op een ongelijkheid. Er zijn volgens het rapport geen standaardmethodes om de kwaliteit van AI-gereedschap te borgen, en daardoor is het voor verdedigers lastig om dat gereedschap in te zetten in vitale sectoren waar betrouwbaarheid essentieel is, terwijl aanvallers zulke beperkingen niet hebben. Het rapport noemt ook een dilemma voor wie de regels maakt. Omdat het lastig kan zijn om nuttig gebruik van schadelijk gebruik te onderscheiden, kunnen te strenge safeguards verdedigers volgens het rapport hinderen, bijvoorbeeld als AI-systemen helemaal niet meer op vragen over cybersecurity mogen antwoorden. Tot nu toe hebben AI-capaciteiten volgens het rapport vooral bestaande aanvalsmethodes versneld of opgeschaald, en geen nieuwe soorten aanvallen gemaakt. Ook is volgens het rapport niet gemeld dat AI-systemen voor algemene doeleinden in de echte wereld een cyberaanval van begin tot eind hebben uitgevoerd.

    Hendrycks en anderen waarschuwen dat de afweging van kosten en baten in cybersecurity scheef kan groeien als aanvallers op den duur veel meer mogelijkheden krijgen die bijna geen moeite kosten. Aanvallers kunnen dan volgens hen de overhand krijgen, zoals aanvallers van ML-systemen nu al een groot voordeel hebben op verdedigers. Ze raden aan om onderzoek te richten op technieken die duidelijk defensief zijn, omdat offensieve en defensieve beveiliging in de toekomst misschien minder twee kanten van dezelfde medaille zijn. Hoe ernstig dit risico is, noemen ze zelf speculatief.

    Toby Shevlane en Allan Dafoe, van het Centre for the Governance of AI van hetzelfde instituut, kijken in een paper uit 2020 naar het publiceren van AI-onderzoek. Volgens hen helpt het openbaar maken van kwetsbaarheden in software vaak de verdediging, maar mag je dat voor AI-onderzoek niet zomaar aannemen. Een oplossing voor software kun je in één keer naar veel systemen verspreiden; misbruik van AI vraagt volgens hen vaker om veranderingen in hoe mensen met elkaar omgaan, en die kosten vaak veel meer. Zulke kwetsbaarheden zijn volgens hen gemiddeld moeilijker te dichten dan kwetsbaarheden in software. Ze concluderen voorzichtig dat publiceren binnen AI in een flink deel van de gevallen per saldo slecht uitpakt voor de veiligheid, en vaker dan in de computerbeveiliging. Het Big Sleep-team hoopt juist op een ongelijk voordeel voor verdedigers, en wil het eigen onderzoek blijven delen, zodat het verschil tussen wat openbaar bekend is en wat alleen achter gesloten deuren bekend is zo klein mogelijk blijft.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **systemic safety** | ML inzetten om de omgeving waarin AI-systemen werken veiliger te maken, zoals cybersecurity en goed geïnformeerde besluitvorming; naar Hendrycks en anderen |
        | **offense-defense balance** | hoe makkelijk het is om een aanval uit te voeren vergeleken met je ertegen te verdedigen, voor één doel, vorm van conflict en groep betrokkenen; naar Garfinkel en Dafoe |
        | kwetsbaarheid | een fout in software die een aanvaller kan gebruiken |
        | patch | een wijziging in software die een kwetsbaarheid dicht |
        | zero day | een kwetsbaarheid die de verdediger nog niet kent; `zero_days` in oefening 1 |
        | OD-scaling | offensive-then-defensive scaling: meer investeren aan beide kanten helpt eerst de aanvaller, dan de verdediger |
        | defensive accelerationism | je richten op technologie die beschermt tegen risico's voor de samenleving |
        | $\min(n, \text{attack})$ | het kleinste van de twee; in Python `min(n, attack)` |
        """),
        L.watch_out("Het model van Garfinkel en Dafoe is geïdealiseerd, en de getallen in deze les zijn verzonnen: ze laten zien wat uit de aannames volgt, niet wat er gaat gebeuren. Of AI aanvallers of verdedigers meer helpt, is volgens het International AI Safety Report 2026 nog onzeker."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om te beoordelen of een AI-capaciteit de aanval of de verdediging meer helpt:

    1. Benoem het doel, de vorm van conflict en wie er meedoen: de balans hangt van alle drie af.
    2. Vraag je af wat dezelfde capaciteit voor elke kant doet.
    3. Vraag je af wie die het eerst kan gebruiken, en of de ene kant alles moet vinden terwijl de andere één gat nodig heeft.
    4. Vraag je af of meer ervan nu de aanvaller helpt en later de verdediger, zoals bij OD-scaling.
    5. Zoek wat een kant tegenhoudt: ontbrekende kwaliteitscontroles, dure reparaties, beperkte toegang.
    6. Kies liever toepassingen die duidelijk de verdediging helpen, en zorg dat safeguards verdedigers niet buitensluiten.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel rekent uit hoeveel kwetsbaarheden een aanvaller vindt bij drie niveaus van inspanning, in software met 10 kwetsbaarheden. Eén eenheid inspanning levert één verwachte ontdekking op, zoals in het model van Garfinkel en Dafoe. De functie `min` van Python geeft de kleinste van de waarden die je erin stopt. De getallen zijn verzonnen.
    """)
    return


@app.cell
def example_found():
    n = 10   # verzonnen: het aantal kwetsbaarheden in de software
    for attack in [4, 10, 14]:
        found = min(n, attack)   # de kleinste van de twee: geen kant vindt meer kwetsbaarheden dan er zijn
        print("inspanning", attack, "vindt", found, "kwetsbaarheden")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met een inspanning van 4 vindt de aanvaller 4 kwetsbaarheden; vanaf een inspanning van 10 alle 10, en meer inspanning vindt er niet meer. In oefening 1 voeg je de verdediger toe: van de kwetsbaarheden die de aanvaller vindt, zijn alleen die welke de verdediger niet heeft gevonden zero days.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_zero_days():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. zero_days: wat de aanvaller wint

        Het geïdealiseerde model van Garfinkel en Dafoe in een functie. Schrijf `zero_days(n, attack, defend)`, die het verwachte aantal zero days teruggeeft voor `n` kwetsbaarheden, een inspanning `attack` van de aanvaller en een inspanning `defend` van de verdediger. Maak `found` gelijk aan het aantal kwetsbaarheden dat de aanvaller vindt, zoals in het uitgewerkte voorbeeld, en `unknown` aan het aandeel van alle `n` kwetsbaarheden dat de verdediger niet heeft gevonden; ook de verdediger vindt er hooguit `n`. Geef `found` keer `unknown` terug. Vervang de `...`. De check roept `zero_days(10, 4, 4)` aan, en daarna andere gevallen.
        """),
        L.task("Het verwachte aantal zero days voor een aanvaller en een verdediger."),
    ])
    return


@app.cell(hide_code=True)
def exercise_zero_days():
    box_zero_days = L.code_box("zero_days", lines=4)
    box_zero_days
    return (box_zero_days,)


@app.cell(hide_code=True)
def check_zero_days(box_zero_days):
    result_zero_days = L.check("zero_days", box_zero_days.value)
    result_zero_days
    return (result_zero_days,)


@app.cell(hide_code=True)
def hints_zero_days():
    L.hints(
        "zero_days",
        [
            "Begin klein: 10 kwetsbaarheden, en elke kant vindt er 4. Hoeveel vindt de aanvaller, en welk aandeel van alle 10 heeft de verdediger niet gevonden?",
            "Maak `found` zoals in het uitgewerkte voorbeeld. Neem voor `unknown` wat de verdediger vindt, nooit meer dan `n`, trek dat af van `n` en deel door `n`. Geef `found` keer `unknown` terug.",
            "`    unknown = (n - ___) / n`",
        ],
        why="`found = min(n, attack)`, `unknown = (n - min(n, defend)) / n` en `return found * unknown`. Met 10 kwetsbaarheden en een inspanning van 4 aan elke kant vindt de aanvaller er 4, en heeft de verdediger er 6 van de 10 niet gevonden: 4 × 0,6 = 2,4 zero days.",
    )
    return


@app.cell(hide_code=True)
def text_after_zero_days():
    mo.md(r"""
    Hoe meer de verdediger vindt, hoe minder zero days: tegen een inspanning van 4 van de aanvaller zijn er 2,4 bij een inspanning van 4 van de verdediger, en geen enkele vanaf 10, waar de verdediger alles vindt. Een aanvaller die meer doet dan er te vinden is, wint er niets meer mee: bij een inspanning van 14 tegen 3 vindt de aanvaller alle 10, en daarvan zijn er 7 onbekend voor de verdediger. Volgens Garfinkel en Dafoe wordt een kwetsbaarheid die beide kanten ontdekken geen zero day.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_scaled():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. scaled: beter gereedschap voor beide

        Garfinkel en Dafoe zien beter gereedschap om kwetsbaarheden te vinden als meer inspanning voor beide kanten. Schrijf `scaled(n, attack, defend, factors)`, die een lijst `result` teruggeeft met voor elke `factor` in de lijst `factors` het verwachte aantal zero days als zowel `attack` als `defend` met `factor` vermenigvuldigd worden. De course zet een eigen `zero_days` voor je klaar, dus dit werkt ook als oefening 1 nog niet af is. Voeg in een `for`-loop met `factor` toe wat `zero_days` geeft. Vervang de `...`. De check roept `scaled(10, 2, 1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])` aan, een aanvaller die twee keer zoveel moeite doet als de verdediger, en daarna andere gevallen.
        """),
        L.task("Het verwachte aantal zero days als beter gereedschap de inspanning van beide kanten vermenigvuldigt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_scaled():
    box_scaled = L.code_box("scaled", lines=5)
    box_scaled
    return (box_scaled,)


@app.cell(hide_code=True)
def check_scaled(box_scaled):
    result_scaled = L.check("scaled", box_scaled.value)
    result_scaled
    return (result_scaled,)


@app.cell(hide_code=True)
def hints_scaled():
    L.hints(
        "scaled",
        [
            "Begin klein: reken `zero_days(10, 2, 1)` en `zero_days(10, 4, 2)` met de hand uit. Wat veranderde er tussen de twee aanroepen, en hoe verandert het aantal zero days?",
            "Begin `result` als een lege lijst. Roep in een `for`-loop met `factor` over `factors` `zero_days` aan met `n` en de twee inspanningen, vermenigvuldigd met `factor`, en voeg toe wat die geeft. Geef na de loop `result` terug.",
            "`        result.append(zero_days(n, ___, defend * factor))`",
        ],
        why="`result = []`, dan `for factor in factors:` met `result.append(zero_days(n, attack * factor, defend * factor))`; na de loop `return result`. `zero_days(10, 2, 1)` is 2 × 0,9 = 1,8, en `zero_days(10, 4, 2)` is 4 × 0,8 = 3,2: beide inspanningen verdubbeld, en de aanvaller wint.",
    )
    return


@app.cell(hide_code=True)
def text_after_scaled():
    mo.md(r"""
    Beter gereedschap helpt eerst de aanvaller: het verwachte aantal zero days stijgt van 1,8 bij een factor 1 tot 5 bij een factor 5. Daarna helpt het de verdediger: vanaf daar daalt het, tot 0 bij een factor 10, waar de verdediger alle 10 kwetsbaarheden vindt, ook al doet de aanvaller steeds twee keer zoveel moeite. Dat is OD-scaling in dit model. Of echt gereedschap zo werkt, zegt het model niet: Garfinkel en Dafoe noemen het een aannemelijk effect, als al het andere gelijk blijft. En ze voegen eraan toe dat al het andere misschien niet gelijk blijft: de kwetsbaarheden die sterk geoptimaliseerd gereedschap vindt, hangen misschien meer samen, zodat beide kanten vaker dezelfde vinden, en volgens hun analyse helpt dat de verdediger. Al met al noemen ze het effect van meer automatisering op korte termijn onduidelijk.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: hoeveel kwetsbaarheden

    Beide kanten doen even veel moeite, en jij verandert het aantal kwetsbaarheden in de software. Kijk tot welke inspanning meer moeite de aanvaller nog helpt.
    """)
    return


@app.cell(hide_code=True)
def demo_vulns():
    vulns = L.explore("vulns", "n: het aantal kwetsbaarheden",
                      lambda _v: C.demo_frame(_v, L.lang), start=10, values=C.VULNS, role="",
                      task="Zoek het kleinste aantal kwetsbaarheden waarbij de aanvaller tot een inspanning van 10 nog wint bij meer moeite.")
    vulns
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Bij evenveel moeite aan beide kanten zijn de zero days het hoogst bij een inspanning van de helft van het aantal kwetsbaarheden, en daar zijn het er een kwart van dat aantal: 1,25 bij een inspanning van 2,5 voor 5 kwetsbaarheden; 2,5 bij 5 voor 10; 5 bij 10 voor 20; en 10 bij 20 voor 40. Vanaf 20 kwetsbaarheden helpt meer moeite de aanvaller dus tot een inspanning van 10. Hoe meer kwetsbaarheden er zijn, hoe langer meer moeite de aanvaller blijft helpen voordat meer moeite de verdediger gaat helpen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_percent():
    mo.md(r"""
    ### Oefening 3. Een getal in het bericht van DARPA

    DARPA schrijft over de finale dat de systemen van de teams 86% van de kunstmatige kwetsbaarheden vonden, tegen 37% in de halve finale, en dat ze 68% van de gevonden kwetsbaarheden dichtten. De tellingen zijn 63 kwetsbaarheden erin gestopt, 54 gevonden en 43 gedicht.
    """)
    return


@app.cell(hide_code=True)
def choice_percent_cell():
    choice_percent = L.choice(
        "percent",
        question="Van welk aantal is die 68% een deel, volgens de tellingen?",
        options={
            "found": "43 van de 54 gevonden, precies zoals de zin in het bericht zegt",
            "right": "43 van de 63 erin gestopt; van de 54 gevonden ongeveer 80%",
            "old": "43 van de 70 die DARPA eerst telde",
        },
        correct="right",
        why={
            "right": "43 / 63 ≈ 0,68, en 43 / 54 ≈ 0,80. De 68% is dus een deel van alle 63 kwetsbaarheden die erin gestopt waren, niet van de 54 gevonden; volgens de noot van de redactie van DARPA ontdekten de deelnemers 86% van de kunstmatige kwetsbaarheden en dichtten ze er 68% van. Van wat ze vonden, dichtten de systemen ongeveer 80%. Een percentage zegt weinig zolang je niet weet waarvan het een deel is.",
            "found": "43 / 54 is ongeveer 0,80, geen 0,68. De 68% is 43 van alle 63 kwetsbaarheden: 43 / 63 ≈ 0,68. De zin spreekt van de gevonden kwetsbaarheden, maar de tellingen passen niet bij die lezing; de noot van de redactie van DARPA noemt 68% van de kunstmatige kwetsbaarheden.",
            "old": "43 / 70 is ongeveer 0,61. DARPA verbeterde de 70 in 63, en 43 / 63 ≈ 0,68: de 68% is een deel van alle 63 kwetsbaarheden die erin gestopt waren. Van de 54 gevonden dichtten de systemen ongeveer 80%.",
        },
    )
    choice_percent
    return (choice_percent,)


@app.cell(hide_code=True)
def text_exercise_voice():
    mo.md(r"""
    ### Oefening 4. Een stem aan de telefoon

    Een lab publiceert een methode die ieders stem nabootst uit een paar seconden opgenomen spraak. In de computerbeveiliging helpt het publiceren van een kwetsbaarheid vaak de verdediging: zodra die bekend is, kan een oplossing naar elk systeem verspreid worden.
    """)
    return


@app.cell(hide_code=True)
def choice_voice_cell():
    choice_voice = L.choice(
        "voice",
        question="Helpt publiceren hier ook de verdediging, volgens de redenering van Shevlane en Dafoe?",
        options={
            "yes": "Ja: zodra de methode bekend is, kan iedereen zich voorbereiden, zoals bij software",
            "right": "Niet per se: wat de methode misbruikt, vertrouwen in een bekende stem, is moeilijk te dichten",
            "always": "Nee: volgens hun argument helpt het publiceren van AI-onderzoek aanvallers altijd meer dan verdedigers",
        },
        correct="right",
        why={
            "right": "De methode misbruikt iets wat mensen elke dag doen: ze nemen het geluid van een stem als teken van wie er praat. Anders dan een regel code kun je die gewoonte niet met een update verhelpen. Shevlane en Dafoe geven een vergelijkbaar voorbeeld: criminelen die met gegenereerde spraak de stem van de CEO van een bedrijf nabootsten aan de telefoon. Volgens hun redenering heeft publiceren dan minder waarde voor de verdediging dan bij software.",
            "yes": "Weten dat de methode bestaat helpt wel wat, maar wat ze misbruikt is geen code: mensen nemen een bekende stem als teken van wie er praat, en geen update verandert die gewoonte. Volgens Shevlane en Dafoe zijn zulke kwetsbaarheden gemiddeld moeilijker te dichten dan kwetsbaarheden in software. Zie *Wie wint er meer?* in *Het idee*.",
            "always": "Hun claim is beperkter, en volgens henzelf voorzichtig: publiceren pakt volgens hen in een flink deel van de gevallen per saldo slecht uit, niet altijd. Hier misbruikt de methode vertrouwen in een bekende stem, wat geen update verhelpt, dus publiceren helpt de verdediging minder dan bij software.",
        },
    )
    choice_voice
    return (choice_voice,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Races en selectie](/course/open/rai-race-and-selection?lang=nl) liet zien hoe concurrentie kan aanzetten tot minder voorzichtigheid. Deze les keek naar de andere kant: AI die de systemen eromheen veiliger maakt, wat Hendrycks en anderen systemic safety noemen. Dezelfde capaciteit die een kwetsbaarheid vindt voor een aanvaller, kan die vinden voor een verdediger, en in het model van Garfinkel en Dafoe helpt beter gereedschap eerst de aanvaller en dan de verdediger. Of AI aanvallers of verdedigers meer helpt, is onzeker, en volgens het International AI Safety Report 2026 hangt het deels af van keuzes. De volgende les, *De argumenten voor catastrofaal risico*, kijkt naar de argumenten dat AI tot een catastrofe zou kunnen leiden.

    **Een voorbeeld uit het dagelijks leven.** Met een set om sloten te testen zie je welke sloten in een straat zwak zijn. Een inbreker die de set het eerst heeft, vindt een weg naar binnen. Een bewoner die de set het eerst gebruikt, vervangt de zwakke sloten, en de inbreker vindt niets. Wie er wint met de set, hangt af van wie die het eerst gebruikt, en van hoeveel zwakke sloten er zijn.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_percent, choice_voice, result_scaled, result_zero_days):
    L.wrap_up(result_zero_days, result_scaled, choice_percent, choice_voice)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
