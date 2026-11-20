# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-nines. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow AI Safety

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-nines", "nl")
    C = L.checks   # betrouwbaarheden van 90% tot 99,999%, een dag van 1.000.000 verzoeken, en een taak van 100 stappen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Risico als kans maal schade](/course/open/rai-risk-decomposition?lang=nl) en [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_risk():
    quick_risk = L.choice(
        "quick-risk",
        question="Een verzonnen AI-systeem heeft één gevaar: in 1 op de 100 antwoorden geeft het slecht advies, met een ernst van 50 schadepunten. Wat is volgens de klassieke risicoformule het risico per antwoord?",
        options={
            "right": "0,5 schadepunten: 0,01 maal 50",
            "plus": "50,01 schadepunten: de kans plus de ernst",
            "severity": "50 schadepunten: de schade als het misgaat",
        },
        correct="right",
        why={
            "right": "Risico is de kans maal de ernst: 0,01 × 50 = 0,5 schadepunten per antwoord.",
            "plus": "Een kans en een aantal schadepunten tel je niet op; de klassieke risicoformule vermenigvuldigt ze. Zie [Risico als kans maal schade](/course/open/rai-risk-decomposition?lang=nl).",
            "severity": "Dat is alleen de ernst, alsof het advies bij elk antwoord misging; het gaat mis in 1 op de 100. Zie [Risico als kans maal schade](/course/open/rai-risk-decomposition?lang=nl).",
        },
    )
    quick_risk
    return (quick_risk,)


@app.cell(hide_code=True)
def quick_check_prob():
    quick_prob = L.choice(
        "quick-prob",
        question="In een test mislukken 6 van de 2000 verzoeken aan de API van een AI-model. Wat is, zo geteld, de kans dat een verzoek mislukt?",
        options={
            "complement": "0,997: het deel van de verzoeken dat niet mislukt",
            "right": "0,003: 6 mislukte verzoeken gedeeld door 2000",
            "count": "6: het aantal verzoeken dat mislukte",
        },
        correct="right",
        why={
            "right": "Een kans die je uit pogingen telt, is hoe vaak het gebeurde, gedeeld door het aantal pogingen: 6 / 2000 = 0,003.",
            "complement": "Dat is de kans dat een verzoek niet mislukt, 1994 van de 2000; de vraag gaat over de verzoeken die mislukken. Zie [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl).",
            "count": "Dat is hoe vaak het gebeurde, nog geen kans, en een kans ligt tussen 0 en 1: deel door het aantal pogingen. Zie [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl).",
        },
    )
    quick_prob
    return (quick_prob,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een aanbieder van AI-modellen belooft dat de dienst 99,9% van de tijd beschikbaar is. Een team bouwt een AI-agent die in 100 stappen een onkostendeclaratie indient, en elke stap gaat in 99% van de gevallen goed. Beide getallen klinken bijna perfect.

    Maar 99% betekent ook 1 fout op de 100. Een dienst die 1.000.000 vragen per dag krijgt en bij 1% daarvan faalt, faalt elke dag 10.000 keer. Hoe dicht bij 100% dicht genoeg is, hangt af van hoe vaak een systeem wordt gebruikt, van hoeveel stappen allemaal goed moeten gaan, en van wat één fout kost. Deze les telt betrouwbaarheid in negens, leest wat de negens in de belofte van een aanbieder wel en niet dekken, en vermenigvuldigt betrouwbaarheid over de stappen van een taak.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Een aanbieder belooft dat de API van het AI-model 99,9% van de tijd beschikbaar is, geteld in tijd. Hoe lang mag de API per jaar onbeschikbaar zijn en de belofte toch houden?",
        options={
            "minutes": "Ongeveer 5 minuten per jaar",
            "right": "Ongeveer 9 uur per jaar",
            "days": "Ongeveer 4 dagen per jaar",
        },
        reveal="**Ongeveer 9 uur per jaar.** Een jaar heeft 8760 uur, en 0,1% daarvan is 8,76 uur, of 43,2 minuten in een maand van 30 dagen. Drie negens klinken alsof de dienst altijd aan staat, maar ze laten meer dan een werkdag uitvaltijd per jaar toe. Ongeveer 5 minuten per jaar is 99,999%, vijf negens; ongeveer 4 dagen is 99%, twee negens.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Wat een negen is

    Betrouwbaarheid is de kans dat een systeem niet faalt: één min de kans op falen uit [Risico als kans maal schade](/course/open/rai-risk-decomposition?lang=nl). Dicht bij 100% tel je die in negens. In *Site Reliability Engineering* (onlineversie, 2017), het boek waarin engineers van Google beschrijven hoe ze de diensten van Google draaiende houden, streeft een dienst naar een beschikbaarheid van bijvoorbeeld 99,9%, 99,99% of 99,999%, en volgens het boek staat elke extra negen voor een verbetering van een orde van grootte richting 100% beschikbaarheid. Een orde van grootte is een factor 10.

    Het leerboek *Introduction to AI Safety, Ethics, and Society* (2024) van Dan Hendrycks maakt er, naar een voorstel van de wiskundige Terence Tao, een maat van: de **negens van betrouwbaarheid**. Informeel heeft een systeem zoveel negens als er aan het begin staan van de betrouwbaarheid als decimaal getal of percentage. 99% heeft twee negens, 99,9% drie. Formeel, met p de betrouwbaarheid als decimaal getal, zijn de negens k = −log10(1 − p). Daarin is 1 − p de kans op falen, en de logaritme met grondtal 10 uit [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl) geeft de macht van 10 die het is: bij 99,9% is de kans op falen 0,001, en dat is 10 tot de macht −3, dus k = 3. De formule geeft ook de negens tussen de hele in: 50% heeft er ongeveer 0,3, en 99,5% ongeveer 2,3. Elke extra negen maakt de kans op falen tien keer zo klein, zegt het leerboek: tien keer zo weinig fouten, en gemiddeld tien keer zo lang tot de eerste.

    ### Meer gebruik, meer fouten

    Hoe vaker je een systeem gebruikt, hoe groter de kans dat je een fout tegenkomt, schrijft het leerboek. Een zelfrijdende auto maakt veel eerder een fout op een rit waarin hij 1000 beslissingen moet nemen dan op een rit met maar 10 beslissingen. Bij 1.000.000 verzoeken per dag:

    | betrouwbaarheid | negens | mislukte verzoeken per dag |
    |---|---|---|
    | 99% | 2 | 10.000 |
    | 99,9% | 3 | 1000 |
    | 99,99% | 4 | 100 |
    | 99,999% | 5 | 10 |

    Het leerboek geeft ook de beslissing waarbij een fout waarschijnlijk is: beslissing 100 bij 99%, beslissing 1000 bij 99,9%. Dat getal is 1/(1 − p): als elke beslissing dezelfde kans op falen heeft, onafhankelijk van de andere, komt de eerste fout gemiddeld bij die beslissing. Die voorwaarde schrijft het leerboek er niet bij.

    ### Geteld in tijd of in verzoeken

    Het SRE-boek telt beschikbaarheid op twee manieren. In tijd, als het deel van de tijd dat het systeem werkt: een systeem met een doel van 99,99% mag per jaar tot 52,56 minuten onbeschikbaar zijn. In verzoeken, als het deel van de verzoeken dat slaagt: een systeem dat 2,5 miljoen verzoeken per dag afhandelt met een doel van 99,99% per dag, mag die dag tot 250 fouten maken en het doel toch halen. De beschikbaarheidstabel van het boek, die geen gepland onderhoud meetelt, staat bij 99% 3,65 dagen per jaar toe, bij 99,99% 52,6 minuten en bij 99,999% 5,26 minuten.

    ### Wat een aanbieder belooft

    Een service level agreement, of SLA, is de schriftelijke belofte van een aanbieder over de beschikbaarheid, met wat klanten krijgen als de aanbieder tekortschiet. In de SLA voor Bedrock, de dienst waarmee Amazon Web Services (AWS) AI-modellen aanbiedt, zegt AWS commercieel redelijke inspanningen te zullen leveren om Bedrock beschikbaar te houden (de versie die het laatst is bijgewerkt op 4 oktober 2023). Zakt een maand onder 99,9%, dan kan een klant een tegoed aanvragen van 10% van de kosten van die maand, te verrekenen met latere rekeningen; onder 99,0% 25%, onder 95,0% 100%. Dat tegoed geeft geen recht op terugbetaling of een andere betaling. De SLA telt de beschikbaarheid per interval van 5 minuten, als het deel van de verzoeken dat niet met een fout mislukt, en een fout is een verzoek dat foutcode 500 teruggeeft: de code waarmee een server zegt dat er aan zijn kant iets misging. De SLA van Google voor Vertex AI (laatst gewijzigd op 12 februari 2026) telt in minuten: een minuut telt als onbeschikbaar bij meer dan 5% fouten, waarbij een fout een antwoord met code 500 of 503 is. Twee beloftes van 99,9% hoeven niet hetzelfde te tellen.

    ### Een storing

    Op 11 december 2024, van 15.16 tot 19.38 uur Pacific-tijd, werkten alle diensten van OpenAI flink slechter of helemaal niet, staat in de nabeschouwing van OpenAI: 4 uur en 22 minuten. De oorzaak was een nieuwe dienst die telemetrie verzamelt, en die zo was ingesteld dat hij de software overbelastte waarmee OpenAI de grote clusters van servers aanstuurt. De wijziging was getest in een testcluster, en daar waren geen problemen gezien. Als alle 262 minuten als uitvaltijd telden, zou die ene middag een maand van 31 dagen terugbrengen tot ongeveer 99,41%, ruim onder drie negens; de nabeschouwing zegt niet hoeveel ervan echt uitvaltijd was.
    """)
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### De prijs van een negen

    Een negen kost doorgaans meer dan de vorige. Volgens het SRE-boek kan een volgende stap in betrouwbaarheid 100 keer zoveel kosten als de stap ervoor. En vanaf een punt merken gebruikers het meestal niet meer: wie een smartphone gebruikt die zelf 99% betrouwbaar is, ziet geen verschil tussen een dienst van 99,99% en een van 99,999%. Google wil een dienst betrouwbaar genoeg maken, maar niet betrouwbaarder dan nodig. Wat het doel overlaat, is een error budget: als een dienst 99,999% van alle vragen in een kwartaal wil afhandelen, is het error budget een foutpercentage van 0,001% in dat kwartaal. Zolang er nog budget over is, mogen er nieuwe versies uit; als het op is, stoppen de nieuwe versies een tijd.

    ### Negens zijn geen risico

    De negens van betrouwbaarheid meten volgens het leerboek alleen een kans, geen risico. Ze zeggen hoe vaak een systeem faalt, niet wat een fout kost: een AI die een spel speelt, brengt veel minder risico mee dan een zelfrijdende auto, ook als beide drie negens hebben. Voor risico vermenigvuldig je de kans op falen met de ernst, zoals in [Risico als kans maal schade](/course/open/rai-risk-decomposition?lang=nl).

    ### Een keten van stappen

    Een AI-agent die een reis boekt, zoekt, vergelijkt, vult een formulier in en betaalt: elke stap moet goed gaan. Twee gebeurtenissen zijn onafhankelijk als de ene niets verandert aan de kans op de andere. Zijn de stappen van een taak onafhankelijk, dan worden hun kansen vermenigvuldigd: met n stappen die elk slagen met kans p, slaagt de hele taak met kans p × p × … × p = pⁿ. Vijf stappen van 90% slagen in ongeveer 59% van de gevallen. Lange taken verliezen hun negens snel.

    In 2025 betoogde Toby Ord van de Universiteit van Oxford dat de metingen van METR, een onderzoeksorganisatie die test wat AI-agents kunnen, te verklaren zijn met een eenvoudig model: een vaste kans op falen in elke minuut die een mens over de taak zou doen. Het succes valt dan exponentieel met de lengte van een taak, en elke agent heeft een halveringstijd: de taaklengte die de agent in de helft van de gevallen voltooit. Een taak die drie keer zo lang is, is als drie zulke taken achter elkaar: bij 80% per stuk slaagt die in 0,8³ = 0,512 van de gevallen, ongeveer de helft. De taaklengte die een agent in 80% van de gevallen voltooit, is dus ongeveer een derde van de halveringstijd; bij 90% ongeveer een zevende, bij 99% ongeveer 1/70 en bij 99,9% ongeveer 1/700. Vanaf daar deelt volgens Ord elke extra negen de time horizon door 10, waarbij de time horizon de lengte is van de taken, in de tijd die vakmensen ervoor nodig hebben, die een model met een gekozen slagingskans voltooit. In de versie van de paper van METR uit juli 2026 zijn de horizons bij 80% 4 tot 6 keer zo kort als die bij 50%, een groter verschil dan de factor van ongeveer 3 uit het model. Ord benadrukt dat het niet om een precies vaste kans op falen per minuut gaat. Of het model ook voor andere taken geldt, is volgens Ord onbekend. En stappen zijn, net als pogingen, zelden helemaal onafhankelijk: in [Robuust, veerkrachtig, beschikbaar, voorspelbaar](/course/open/rai-reliability?lang=nl) slaagde dezelfde taak telkens weer, of mislukte die telkens weer.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **negens van betrouwbaarheid** | het aantal negens aan het begin van een betrouwbaarheid, of precies k = −log10(1 − p): 99,9% heeft er drie |
        | betrouwbaarheid | de kans dat een systeem niet faalt, per verzoek, per beslissing of per stap |
        | beschikbaarheid | het deel van de tijd, of van de verzoeken, waarin een dienst werkt |
        | orde van grootte | een factor 10 |
        | SLA | service level agreement: de schriftelijke belofte van een aanbieder over de beschikbaarheid, en wat klanten krijgen als die tekortschiet |
        | error budget | de fouten die een doel voor betrouwbaarheid in een periode nog toelaat |
        | onafhankelijk | twee gebeurtenissen zijn onafhankelijk als de ene niets verandert aan de kans op de andere |
        | pⁿ | p n keer met zichzelf vermenigvuldigd: de kans dat n onafhankelijke stappen met betrouwbaarheid p allemaal slagen |
        | halveringstijd | in het model van Ord de taaklengte die een agent in de helft van de gevallen voltooit |
        | time horizon | de maat van METR: de lengte van de taken, in de tijd die vakmensen nodig hebben, die een model met een gekozen slagingskans voltooit |
        | `np.log10` | de logaritme met grondtal 10 van NumPy |
        """),
        L.watch_out("Negens zeggen hoe vaak een systeem faalt, niet wat een fout kost. En pⁿ geldt alleen voor onafhankelijke stappen: echte stappen falen vaak samen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Schrijf de betrouwbaarheid als decimaal getal p; de kans op falen is 1 − p.
    2. Tel de negens: k = −log10(1 − p).
    3. Vermenigvuldig de kans op falen met hoe vaak het systeem wordt gebruikt: mislukte verzoeken per dag, of uitvaltijd per jaar.
    4. Reken voor een taak van n stappen die allemaal moeten slagen pⁿ uit, en zeg erbij dat dat onafhankelijke stappen aanneemt.
    5. Lees wat een belofte als fout telt, en wat de aanbieder geeft als die tekortschiet.
    6. Weeg de fouten naar hun schade voordat je een systeem betrouwbaar genoeg noemt.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel neemt drie negens, 99,9%, en rekent uit hoeveel van 1.000.000 verzoeken mislukken, hoeveel uur per jaar dat is als je beschikbaarheid in tijd telt, en bij welke beslissing gemiddeld de eerste fout komt.
    """)
    return


@app.cell
def example_nines():
    p = 0.999                                # betrouwbaarheid: drie negens
    print(round(1000000 * (1 - p)))          # mislukte verzoeken op de 1.000.000
    print(round(365 * 24 * (1 - p), 2))      # uren onbeschikbaar per jaar, geteld in tijd
    print(round(1 / (1 - p)))                # de beslissing van de eerste fout, gemiddeld
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Drie negens: 1000 mislukte verzoeken op de 1.000.000, 8,76 uur onbeschikbaar per jaar, en de eerste fout gemiddeld bij beslissing 1000. In oefening 1 tel je de negens zelf.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_nines():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. nines: de negens tellen

        Schrijf `nines(p)`. De functie krijgt een betrouwbaarheid `p` als decimaal getal, zoals 0,999, en geeft de negens van betrouwbaarheid terug volgens de formule uit *Wat een negen is*. Het vak importeert NumPy als `np`, dus `np.log10` staat klaar; vervang de `...`. De check roept `nines(0.999)` aan, en daarna andere betrouwbaarheden, ook een paar tussen hele negens in.
        """),
        L.task("De negens van een betrouwbaarheid, volgens de formule."),
    ])
    return


@app.cell(hide_code=True)
def exercise_nines():
    box_nines = L.code_box("nines", lines=5)
    box_nines
    return (box_nines,)


@app.cell(hide_code=True)
def check_nines(box_nines):
    result_nines = L.check("nines", box_nines.value)
    result_nines
    return (result_nines,)


@app.cell(hide_code=True)
def hints_nines():
    L.hints(
        "nines",
        [
            "De formule k = −log10(1 − p) uit *Wat een negen is*, in Python geschreven.",
            "Eerst de kans op falen, één min `p`; dan de logaritme met grondtal 10 daarvan, met de functie van NumPy uit [Exponenten en logaritmen](/course/open/math-exp-log?lang=nl); dan een minteken ervoor, zodat −3 verandert in 3.",
            "`    return ___(1 - p)`",
        ],
        why="`return -np.log10(1 - p)`. Bij 0,999 is de kans op falen 0,001, de logaritme met grondtal 10 daarvan −3, en zijn de negens 3.",
    )
    return


@app.cell(hide_code=True)
def text_after_nines():
    mo.md(r"""
    Drie negens voor 99,9%, vijf voor 99,999%, en 0,3 voor 50%: de formule telt hoe vaak de kans op falen door 10 is gedeeld. Onder de check heeft elke rij vanaf 99% tien keer zo weinig mislukte verzoeken als de rij erboven, en één negen meer.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_success():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. task_success: een taak van veel stappen

        Een agent doet een taak van `steps` stappen. Elke stap slaagt met kans `p`, onafhankelijk van de andere, en de taak slaagt alleen als ze allemaal slagen. Schrijf `task_success(p, steps)`: de functie geeft die kans terug. Reken die uit in een loop: een variabele `chance` die bij 1 begint, en een `for`-loop met `step` over `range(steps)`. Vervang de `...`. De check roept `task_success(0.99, 100)` aan, en daarna andere kansen en aantallen stappen.
        """),
        L.task("De kans dat elke stap van een taak slaagt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_success():
    box_success = L.code_box("task_success", lines=5)
    box_success
    return (box_success,)


@app.cell(hide_code=True)
def check_success(box_success):
    result_success = L.check("task_success", box_success.value)
    result_success
    return (result_success,)


@app.cell(hide_code=True)
def hints_success():
    L.hints(
        "task_success",
        [
            "Elke stap moet slagen. Bij onafhankelijke stappen worden de kansen vermenigvuldigd: één factor `p` per stap.",
            "Begin `chance` bij 1, de kans voor er een stap is gezet. Een `for`-loop over `range(steps)` loopt één keer per stap; vermenigvuldig daarin `chance` met `p`. Geef `chance` na de loop terug.",
            "`        chance = ___`",
        ],
        why="`chance = 1`, `for step in range(steps):` met daarin `chance = chance * p`, en dan `return chance`. Voor 100 stappen bij 0,99: 0,366.",
    )
    return


@app.cell(hide_code=True)
def text_after_success():
    mo.md(r"""
    Een taak van 100 stappen bij 99% slaagt maar in ongeveer 37% van de gevallen: 0,99 tot de macht 100 is 0,366. Elke stap lijkt bijna perfect, en toch mislukt de taak bijna twee keer op de drie. De kans zakt onder de helft na 69 stappen bij 99%, en na 693 stappen bij 99,9%: één negen meer, tien keer zoveel stappen. `p ** steps` geeft hetzelfde in één regel.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: van één negen naar vijf

    De schuif zet de betrouwbaarheid, van 90% tot 99,999%. Het plaatje toont de negens, de mislukte verzoeken op de 1.000.000, de uitvaltijd per jaar als je beschikbaarheid in tijd telt, en de kans dat een taak van 100 onafhankelijke stappen slaagt, met de kromme van die kans over het aantal stappen.
    """)
    return


@app.cell(hide_code=True)
def demo_reliability():
    reliability = L.explore("reliability", "betrouwbaarheid, in %", lambda _v: C.demo_frame(_v, L.lang),
                            start=90, values=C.PERCENTS, role="",
                            task="Zoek de kleinste betrouwbaarheid op de schuif waarbij een taak van 100 stappen in minstens 9 van de 10 gevallen slaagt.")
    reliability
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Vanaf 99,9%: daar slaagt een taak van 100 stappen in 90,5% van de gevallen, en bij 99% maar in 36,6%. Elke negen meer geeft tien keer zo weinig fouten en tien keer zo weinig uitvaltijd, en dan slagen taken die ongeveer tien keer zo lang zijn even vaak.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_risk():
    mo.md(r"""
    ### Oefening 3. Twee keer drie negens

    Een schaakprogramma en een AI-assistent die mensen met diabetes een dosis insuline voorstelt, hebben allebei drie negens van betrouwbaarheid, en nemen allebei 1000 beslissingen per dag.
    """)
    return


@app.cell(hide_code=True)
def choice_risk_cell():
    choice_risk = L.choice(
        "risk",
        question="Welke van de twee brengt meer risico mee?",
        options={
            "same": "Hetzelfde risico: dezelfde negens en hetzelfde aantal beslissingen",
            "right": "De insulineassistent: zijn fouten kunnen veel erger uitpakken",
            "neither": "Geen van beide: één fout per dag is te weinig om mee te tellen",
        },
        correct="right",
        why={
            "right": "Allebei maken ze ongeveer één fout per dag, maar een verkeerde zet kost een partij schaak, en een verkeerde dosis insuline kan een mens schaden. Negens meten alleen een kans, geen risico; risico weegt elke fout naar de ernst.",
            "same": "Dezelfde negens en hetzelfde aantal beslissingen geven hetzelfde aantal fouten, ongeveer één per dag; ze zeggen niets over de schade van een fout. Zie *Negens zijn geen risico* in *Het idee*.",
            "neither": "Eén fout per dag is 365 per jaar; of dat ertoe doet, hangt af van wat elke fout kost, en dat verschilt hier enorm. Zie *Negens zijn geen risico* in *Het idee*.",
        },
    )
    choice_risk
    return (choice_risk,)


@app.cell(hide_code=True)
def text_exercise_available():
    mo.md(r"""
    ### Oefening 4. Beschikbaar, maar fout

    Een team roept een maand lang een AI-model aan via Bedrock. Elk verzoek krijgt een antwoord, en geen enkel geeft foutcode 500 terug, maar 1 antwoord op de 20 is fout.
    """)
    return


@app.cell(hide_code=True)
def choice_available_cell():
    choice_available = L.choice(
        "available",
        question="Wat is volgens de SLA van Bedrock de beschikbaarheid van die maand?",
        options={
            "wrong": "95%: 19 op de 20 antwoorden kloppen",
            "right": "100%: geen verzoek mislukte met een fout",
            "minutes": "Geen: de SLA telt minuten uitvaltijd, geen antwoorden",
        },
        correct="right",
        why={
            "right": "In de SLA van Bedrock is een fout een verzoek dat foutcode 500 teruggeeft. Een verkeerd antwoord dat aankomt, telt als beschikbaar: de SLA gaat erover of de dienst antwoordt, niet of het antwoord klopt.",
            "wrong": "Dat is het deel van de antwoorden dat klopt. Bedrock telt een verzoek alleen als mislukt als het foutcode 500 teruggeeft, en dat gebeurde niet. Zie *Wat een aanbieder belooft* in *Het idee*.",
            "minutes": "Vertex AI telt in minuten; Bedrock telt het deel van de verzoeken dat niet met een fout mislukt, per interval van 5 minuten. Zie *Wat een aanbieder belooft* in *Het idee*.",
        },
    )
    choice_available
    return (choice_available,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Risico als kans maal schade](/course/open/rai-risk-decomposition?lang=nl) woog de kans op falen naar de schade. Deze les draaide die kans om tot betrouwbaarheid en telde die in negens: elke negen is tien keer zo weinig fouten. Over veel verzoeken tellen de fouten op, zoals de tellingen uit [Kans: tellen hoe vaak](/course/open/prob-basics?lang=nl); over veel stappen worden de kansen vermenigvuldigd. De volgende les, *Ontwerpen voor veiligheid*, geeft principes uit de techniek die fouten zeldzamer en minder schadelijk maken, zoals redundantie, scheiding van taken en fail-safes.

    **Een voorbeeld uit het dagelijks leven.** Een oud snoer kerstlampjes gaat helemaal uit als één lampje kapot is. Met 50 lampjes die elk in 99% van de gevallen werken, brandt het hele snoer maar ongeveer 6 keer op de 10.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_available, choice_risk, result_nines, result_success):
    L.wrap_up(result_nines, result_success, choice_risk, choice_available)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
