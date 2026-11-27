# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les sys-requirements. Route: course/route/lessen/systemen.yaml. Checks, de drukke dag van de demo en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-requirements", "nl")
    C = L.checks   # de wachttijden, de storingen, de drukke dag van de demo, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_model():
    quick_model = L.choice(
        "quick-model",
        question="Hoeveel van de 118 regels code van de spamdienst van de course waren het model?",
        options={
            "most": "Ongeveer 100 van de 118",
            "half": "De helft",
            "right": "5 van de 118",
            "all": "Alle 118: de instellingen horen bij het model",
        },
        correct="right",
        why={
            "right": "5 regels: scikit-learn doet het leren, en de rest verzamelt, controleert, serveert en monitort. Deze les kijkt naar wat de rest moet opleveren.",
            "most": "Het grootste deel van de code was helemaal geen model. Wat deden de andere delen, zoals verzamelen en serveren? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
            "half": "Het model was veel kleiner dan de rest. Welke library deed het leren? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
            "all": "De instellingen zijn een van de delen rond het model, naast verzamelen, serveren en monitoren. Hoeveel regels was het model zelf? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
        },
    )
    quick_model
    return (quick_model,)


@app.cell(hide_code=True)
def quick_check_production():
    quick_production = L.choice(
        "quick-production",
        question="Wat vraagt productie van een systeem, en onderzoek vaak niet?",
        options={
            "score": "De beste score op een vaste test set",
            "right": "Goede antwoorden voor veel gebruikers, snel en goedkoop genoeg",
            "biggest": "Het grootste model dat er is, één keer getraind op alle data die er is",
            "author": "Code die alleen de maker kan draaien",
        },
        correct="right",
        why={
            "right": "En dat op data die blijft veranderen. Deze les maakt daar vier eisen van, elk met een getal dat je kunt controleren.",
            "score": "Dat vraagt onderzoek. Wat heeft een systeem nodig als veel gebruikers erop vertrouwen? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
            "biggest": "Een groter model is vaak trager en duurder. Wat vragen de gebruikers van een systeem? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
            "author": "Als alleen de maker de code kan draaien, is de code in productie een risico. Wat vraagt productie wel? Zie [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl).",
        },
    )
    quick_production
    return (quick_production,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Op 1 augustus 2012 nam Knight Capital, een groot handelshuis in Amerikaanse aandelen, nieuwe code in gebruik op de acht servers die de orders van het bedrijf naar de beurs stuurden. De nieuwe code gebruikte opnieuw een vlag die vroeger Power Peg aanzette, een oude functie die Knight sinds 2003 niet meer gebruikte en nooit had weggehaald. Een technicus zette de nieuwe code op zeven van de acht servers, en niemand controleerde het werk. Op de achtste server stond de oude code nog, en daar zette de vlag Power Peg weer aan: toen de beurs openging, stuurde deze server een stroom orders, veel meer dan de klanten hadden gevraagd.

    De nieuwe code was het probleem niet: op de zeven servers met de nieuwe code werkte alles. Het systeem eromheen ging mis: oude code die niemand weghaalde, en een deployment met de hand (de nieuwe code in gebruik nemen) zonder een tweede paar ogen. Een systeem in productie wordt op meer beoordeeld dan op de vraag of de antwoorden kloppen. Deze les geeft vier eisen een naam, en laat zien hoe je van elke eis een getal maakt dat je kunt controleren.

    **Een voorbeeld uit het dagelijks leven.** Een brug moet elke dag verkeer dragen, ook in een storm. De brug moet meer verkeer dragen als de stad groeit. Een nieuw team ingenieurs moet de brug kunnen inspecteren en repareren. En de brug moet zwaardere vrachtwagens aankunnen dan toen de brug gebouwd werd. Een sterke brug is geen goede brug als niemand de brug kan inspecteren.
    """)
    return


@app.cell(hide_code=True)
def predict_minutes():
    guess_minutes = L.guess(
        "minutes",
        question="De stroom orders kostte Knight Capital meer dan 460 miljoen dollar. Hoe lang duurde de stroom, denk je?",
        options={
            "right": "45 minuten",
            "days": "Twee dagen",
            "weeks": "Twee weken",
            "half-year": "Een half jaar",
        },
        reveal="**45 minuten.** Volgens de SEC, de toezichthouder op de Amerikaanse beurzen, stuurde de achtste server in 45 minuten meer dan 4 miljoen orders, terwijl de server 212 orders van klanten probeerde uit te voeren, en raakte Knight meer dan 460 miljoen dollar kwijt. In 2013 stemde Knight in met een boete van 12 miljoen dollar van de SEC.",
    )
    guess_minutes
    return (guess_minutes,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Vier eisen.** Chip Huyen (2022) noemt vier eisen waar de meeste ML-systemen in productie aan moeten voldoen. Een model met een goede score op een test set voldoet op zichzelf aan geen van de vier:

        | Eis | De vraag | Een getal dat je kunt controleren | In de spamdienst |
        |---|---|---|---|
        | **betrouwbaarheid** | blijft het systeem het juiste doen op het juiste niveau, ook als hardware, software of mensen fouten maken? | het deel van de berichten dat op tijd een score krijgt | 99% van de berichten krijgt binnen 1 minuut een score |
        | **schaalbaarheid** | houdt het systeem het bij als de berichten, de data of het systeem groeien, en kan het weer krimpen? | het aantal berichten per minuut dat het op het drukste moment bijhoudt | 1.400 berichten per minuut na een stormwaarschuwing |
        | **onderhoudbaarheid** | kunnen anderen het systeem begrijpen, veranderen en de verandering in gebruik nemen? | de tijd van een verandering tot het gebruik, en hoeveel mensen de verandering kunnen maken | een nieuwe collega past de threshold aan en neemt de nieuwe threshold binnen een dag in gebruik |
        | **aanpasbaarheid** | volgt het systeem een veranderende wereld, zonder de dienst stil te leggen? | het aantal dagen van een nieuw soort spam tot een model dat het nieuwe soort herkent | een nieuw soort spam binnen 7 dagen herkend |

        Elke eis is bij een echt bedrijf misgegaan:

        - **betrouwbaarheid:** op 28 februari 2017 gaf iemand bij Amazon een opdracht met een verkeerde invoer en haalde zo meer servers uit de lucht dan de bedoeling was. S3, de opslagdienst van Amazon, deed er in één regio van de VS ongeveer vier uur over om weer helemaal te werken, mede omdat sommige onderdelen al jaren niet meer helemaal opnieuw opgestart waren.
        - **schaalbaarheid:** op 25 maart 2025 gaf OpenAI ChatGPT een nieuwe manier om afbeeldingen te maken. Twee dagen later schreef Sam Altman, de CEO van OpenAI, "our GPUs are melting", en OpenAI beperkte een tijdje hoeveel afbeeldingen mensen konden maken.
        - **onderhoudbaarheid:** Knight Capital, hierboven: oude code die nooit weggehaald was, en een deployment met de hand die niemand controleerde.
        - **aanpasbaarheid:** in november 2021 stopte Zillow met Zillow Offers, het eigen bedrijfsonderdeel dat huizen kocht en verkocht. De CEO van Zillow schreef dat huizenprijzen veel moeilijker te voorspellen bleken dan Zillow had verwacht; Zillow schreef 304 miljoen dollar af op de waarde van de eigen huizen, en wilde ongeveer een kwart van het personeel laten gaan.

        **ML faalt stil.** Als een server plat ligt, valt dat binnen minuten op. Als een model slechtere antwoorden geeft, blijft het model gewoon antwoorden, en er crasht niets; Huyen noemt dat stil falen (failing silently). Bij een ML-systeem gaat betrouwbaarheid dus ook over het deel van de antwoorden dat klopt, en niet alleen over antwoorden op tijd.

        **Van een wens naar een getal.** "Snel en betrouwbaar" is een wens die niemand kan controleren. Het boek *Site Reliability Engineering* van Google (2016) maakt er getallen van. **Beschikbaarheid** is het deel van de tijd waarin een dienst werkt, of het deel van de verzoeken dat lukt: de minuten dat de dienst werkt gedeeld door alle minuten, of de verzoeken die lukten gedeeld door alle verzoeken. Algemener:

        - een **service level indicator** (SLI) is een getal dat je meet, meestal goede gebeurtenissen gedeeld door alle gebeurtenissen, zoals het deel van de berichten dat binnen 1 minuut een score kreeg;
        - een **service level objective** (SLO) is een doel voor een SLI over een periode, zoals 99% over 30 dagen; in het voorbeeld van het boek zelf is 99,9% van de Get RPC calls binnen 100 ms klaar, waarbij een RPC call een verzoek van het ene programma aan het andere is;
        - een **service level agreement** (SLA) is een belofte in een contract, met gevolgen, zoals geld terug, als de belofte niet nagekomen wordt.

        Het **error budget** is wat de SLO overlaat: 100% min de SLO. Een SLO van 99,9% beschikbaarheid over 30 dagen staat 43,2 minuten downtime toe. Zolang er budget over is, mag het team veranderingen in gebruik nemen; als het budget op is, gaat betrouwbaarheid voor. De afbeelding laat zien hoe het budget krimpt met elke negen.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of symbool | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **betrouwbaarheid** (reliability) | | het systeem blijft het juiste doen op het juiste niveau, ook als hardware, software of mensen fouten maken | |
        | **schaalbaarheid** (scalability) | | het systeem houdt het bij als de belasting, de data of het systeem groeit, en kan weer krimpen | `C.RUNS` |
        | **onderhoudbaarheid** (maintainability) | | anderen kunnen het systeem begrijpen, veranderen en de verandering in gebruik nemen | |
        | **aanpasbaarheid** (adaptability) | | het systeem volgt een veranderende wereld, zonder de dienst stil te leggen | |
        | **service level indicator** | es-el-ie | een getal dat je over een dienst meet, meestal goede gebeurtenissen gedeeld door alle gebeurtenissen | `share_within(times, limit)` |
        | **service level objective** | es-el-oo | een doel voor een SLI over een periode, zoals 99% over 30 dagen | `C.SLO_SHARE` |
        | **service level agreement** | es-el-aa | een belofte over een dienst in een contract, met gevolgen als de belofte niet nagekomen wordt | |
        | **error budget** | | wat een SLO overlaat: 100% min de SLO, in minuten of in verzoeken | `minutes_left(availability, days, outages)` |
        """),
        L.watch_out("Een SLO van 100% is geen strengere versie van 99,9%: er blijft helemaal geen error budget over. Eén minuut downtime is al te veel, en elke verandering wordt een risico dat het team zich niet kan veroorloven. Kies de SLO die de gebruikers nodig hebben, niet de hoogste die je kunt bedenken."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    De eisen aan een systeem omzetten in getallen:

    1. Vraag bij elk van de vier eisen wat er voor de gebruikers misgaat als het systeem niet aan de eis voldoet.
    2. Kies voor elke eis een getal dat je kunt meten: een SLI, zoals goede antwoorden gedeeld door alle antwoorden.
    3. Leg een doel en een periode vast: de SLO, zoals 99% over 30 dagen.
    4. Reken het error budget uit: 100% min de SLO, in minuten of in verzoeken.
    5. Meet, haal wat misging van het budget af, en laat betrouwbaarheid voorgaan op nieuwe veranderingen als het budget op is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    `C.OUTAGES` bevat de storingen van de spamdienst in één maand van 30 dagen, in minuten, door de course bedacht. De cel telt de storingen op en rekent de beschikbaarheid uit: de minuten dat de dienst werkte, gedeeld door alle minuten. Voer de cel uit.
    """)
    return


@app.cell
def example_outages():
    outages = C.OUTAGES
    minutes = 30 * 24 * 60
    down = 0
    for m in outages:
        down = down + m
    print("minuten in 30 dagen:", minutes)
    print("minuten plat:", down)
    print("beschikbaarheid:", (minutes - down) / minutes)
    return down, minutes, outages


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In 30 dagen lag de spamdienst 12 + 5 + 20 = 37 van de 43.200 minuten plat: een beschikbaarheid van 0,99914, of 99,914%. Met een SLO van 99,9% is dat net genoeg. Oefening 2 rekent uit hoeveel minuten van het error budget er nog over zijn.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een SLI meten

        `share_within(times, limit)` krijgt de seconden die elk bericht op de score wachtte, en een grens in seconden. De functie geeft het deel van de berichten terug dat binnen de grens een score kreeg: een getal van 0 tot 1. De loop neemt de berichten een voor een als `seconds`, en `good` telt de berichten binnen de grens; ook een bericht precies op de grens telt als goed. Vervang de `...` door het deel van de berichten binnen de grens.

        De check roept `share_within(C.TIMES, 60)` aan, met de 20 berichten van een drukke ochtend. Daarna lijsten van een andere lengte, en andere grenzen.
        """),
        L.task("Geef het deel van de berichten binnen de grens terug."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("share", lines=7)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("share", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "share",
        [
            "Stap 2 van het recept: een SLI is goede gebeurtenissen gedeeld door alle gebeurtenissen. `good` bevat de goede gebeurtenissen; alle gebeurtenissen zijn de berichten in `times`.",
            "Je hebt het aantal berichten in de lijst nodig, niet de lijst zelf. Welke functie van Python geeft hoeveel elementen een lijst heeft?",
            "`    return good / ___`",
        ],
        why="`good / len(times)` deelt de berichten binnen de grens door alle berichten. Bij `C.TIMES` blijven 18 van de 20 berichten binnen 60 seconden, een deel van 0,9: onder de SLO van 0,99.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_budget():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Wat er over is van het error budget

        `minutes_left(availability, days, outages)` krijgt een SLO voor beschikbaarheid als deel, zoals `0.999`, een periode in `days`, en een lijst `outages` met storingen in minuten. De functie geeft terug hoeveel minuten van het error budget er na de storingen over zijn: een negatief getal als de dienst langer plat lag dan de SLO toestaat. De loop haalt elke storing, `minutes`, al van `budget` af. Vervang de `...` door het budget aan het begin: het deel van de tijd dat de dienst plat mag liggen, keer alle minuten van de periode, zoals in het uitgewerkte voorbeeld.

        De check roept `minutes_left(0.999, 30, C.OUTAGES)` aan, de maand van het uitgewerkte voorbeeld. Daarna andere SLO's, periodes en storingen.
        """),
        L.task("Laat het budget beginnen als het deel van de tijd dat mis mag gaan, keer alle minuten van de periode."),
    ])
    return


@app.cell(hide_code=True)
def exercise_budget():
    box_budget = L.code_box("budget", lines=6)
    box_budget
    return (box_budget,)


@app.cell(hide_code=True)
def check_budget(box_budget):
    result_budget = L.check("budget", box_budget.value)
    result_budget
    return (result_budget,)


@app.cell(hide_code=True)
def hints_budget():
    L.hints(
        "budget",
        [
            "Stap 4 van het recept: het error budget is 100% min de SLO. `budget` begint als dat deel keer alle minuten van `days` dagen.",
            "Met `availability` als deel geschreven is 100% het getal 1. Welk deel van de tijd blijft er over voor downtime?",
            "`    budget = (___) * days * 24 * 60`",
        ],
        why="`(1 - availability) * days * 24 * 60` is het error budget in minuten: 0,001 van de 43.200 minuten van 30 dagen is 43,2 minuten. De storingen van `C.OUTAGES` nemen er 37 van, dus zijn er nog 6,2 over; een maand met één storing van 50 minuten eindigt 6,8 minuten onder nul.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Eén drukke dag, meer servers

    De demo volgt de spamdienst door één drukke dag, nu de dienst de post van veel gemeenten scoort. Elke server scoort 300 berichten per minuut. De bovenste grafiek toont de berichten per minuut: rustig in de nacht, een piek in de ochtend, en om 17:00 een stormwaarschuwing, waarna 20 minuten lang 1.400 berichten per minuut binnenkomen. De lijn van streepjes is wat de servers kunnen scoren; erboven wachten berichten in een wachtrij. De onderste grafiek toont hoe lang een bericht wacht, in de kleur van fouten als het langer is dan 1 minuut. De SLO is 99% van de berichten binnen 1 minuut. De getallen zijn door de course bedacht.
    """)
    return


@app.cell(hide_code=True)
def demo_servers():
    L.explore("servers", "servers", lambda _v: C.frame(int(_v), L.lang), start=2, values=list(C.SERVERS), role="",
              task="Zoek het kleinste aantal servers dat de SLO haalt. Hoeveel servers zou de gemiddelde minuut van de dag nodig hebben?")
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    De gemiddelde minuut van de dag brengt 244 berichten, minder dan de 300 die 1 server scoort. Toch heeft de SLO 5 servers nodig, door 20 minuten storm: schaalbaarheid gaat over het drukste moment, niet over het gemiddelde. Met 4 servers is de langste wachttijd 3,3 minuten, en haalt 5,6% van de berichten de grens niet.

    Met vijf servers die de hele dag draaien, betaal je ook in de andere 1.420 minuten voor 20 minuten storm. Huyen wijst erop dat schalen ook omlaag schalen betekent: een systeem dat servers bijzet als de belasting stijgt, en de servers daarna weer weghaalt, betaalt alleen voor de storm zolang de storm duurt. Latere lessen over de cloud komen hierop terug.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_which():
    mo.md(r"""
    ### Oefening 3. Welke eis?

    Slechts één engineer weet hoe het spammodel opnieuw getraind wordt: de stappen staan nergens op papier, en de code heeft geen tests. De engineer gaat een maand op vakantie, en een kleine aanpassing van de threshold wacht tot de engineer terug is. Intussen krijgt elk bericht nog steeds binnen een seconde een juiste score.
    """)
    return


@app.cell(hide_code=True)
def choice_which_cell():
    choice_which = L.choice(
        "which",
        question="Aan welke eis voldoet de spamdienst niet?",
        options={
            "reliability": "Betrouwbaarheid: de antwoorden kloppen niet meer",
            "scalability": "Schaalbaarheid: te veel berichten voor de servers",
            "adaptability": "Aanpasbaarheid: de spam veranderde en het model volgde niet",
            "right": "Onderhoudbaarheid: niemand anders kan het veranderen",
        },
        correct="right",
        why={
            "right": "Niemand anders kan het systeem begrijpen, veranderen of trainen, dus wacht een kleine aanpassing een maand. Bij een onderhoudbaar systeem staan de stappen op papier en is elke stap getest, zodat een nieuwe collega de aanpassing in een dag kan maken.",
            "reliability": "Elk bericht krijgt nog steeds binnen een seconde een juiste score. Wat kan er niet gebeuren zolang de engineer weg is?",
            "scalability": "Niets in het verhaal gaat over het aantal berichten. Welk soort werk wacht op de engineer?",
            "adaptability": "De spam is in het verhaal niet veranderd. Waardoor wacht de kleine aanpassing van de threshold een maand?",
        },
    )
    choice_which
    return (choice_which,)


@app.cell(hide_code=True)
def text_exercise_slo():
    mo.md(r"""
    ### Oefening 4. Welk doel is een SLO?

    De gemeente schrijft vier doelen voor de spamdienst op, en wil elke maand weten of de dienst de doelen haalde.
    """)
    return


@app.cell(hide_code=True)
def choice_slo_cell():
    choice_slo = L.choice(
        "slo",
        question="Welk van de vier doelen is een SLO?",
        options={
            "vague": "De spamdienst moet snel en betrouwbaar zijn voor elke inwoner",
            "right": "99% van de berichten krijgt binnen 1 minuut een score, over 30 dagen",
            "never": "De spamdienst ligt nooit plat, zelfs geen minuut",
            "test": "Het model scoort 95% op de test set voordat het model in gebruik gaat",
        },
        correct="right",
        why={
            "right": "Deze zin noemt het getal om te meten (het deel van de berichten binnen 1 minuut), een doel (99%) en een periode (30 dagen), zodat je na elke maand weet of de dienst het doel haalde. Er blijft een error budget van 1% van de berichten over.",
            "vague": "Snel en betrouwbaar zijn wensen: hoe snel, en hoe gemeten? Welk doel noemt een getal, een doel en een periode?",
            "never": "Dat is een SLO van 100%: helemaal geen error budget, dus is één minuut downtime al te veel. Welk doel laat ruimte voor iets dat misgaat?",
            "test": "Een score op een test set meet je één keer, voordat de dienst draait. Bij welk doel meet je de dienst zelf, maand na maand?",
        },
    )
    choice_slo
    return (choice_slo,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Meer dan een model: wat een ML-systeem is](/course/open/sys-what-is-an-ml-system?lang=nl) vroeg productie goede antwoorden voor veel gebruikers, snel en goedkoop genoeg, op data die blijft veranderen. Deze les maakte daar vier eisen van, en van elke eis een getal met een doel: een SLI, een SLO en het error budget dat de SLO overlaat. In de volgende les, *Het probleem formuleren voor productie*, maak je van een wens van de gebruikers een ML-probleem, en zie je hoe deze keuze bepaalt wat je later kunt doen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_slo, choice_which, result_budget, result_share):
    L.wrap_up(result_share, result_budget, choice_which, choice_slo)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
