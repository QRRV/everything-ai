# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-time-horizon. Route: course/route/lessen/verantwoord.yaml. Checks, plaatjes en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai.lesson import lesson

    L = lesson("rai-time-horizon", "nl")
    C = L.checks   # de curve van METR, de verdubbelingstijd van de trend, en de demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Testen op gevaarlijke capabilities](/course/open/rai-dangerous-capabilities?lang=nl) en [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kimi():
    quick_kimi = L.choice(
        "quick-kimi",
        question="In juli 2026 voerde Kimi K3 in één van zijn 10 pogingen een gesimuleerde aanval op een klein bedrijfsnetwerk, een pad van 32 stappen, helemaal uit. Wat concludeerden de beoordelaars?",
        options={
            "fails": "Dat het 9 van de 10 keer mislukt en zulke netwerken dus niet kan aanvallen",
            "right": "Dat het kleine, zwak verdedigde netwerken kan aanvallen",
            "luck": "Nog niets, want één succes kan geluk zijn",
        },
        correct="right",
        why={
            "right": "Eén succes op 10 was genoeg: volgens de beoordelaars kan Kimi K3 zelfstandig kleine, zwak verdedigde en kwetsbare bedrijfssystemen aanvallen. Een test vraagt naar één taak tegelijk. Deze les zet veel taken van heel verschillende moeilijkheid op één schaal.",
            "fails": "Een succes laat zien dat een capability er is; negen mislukkingen maken dat niet ongedaan. Zie [Testen op gevaarlijke capabilities](/course/open/rai-dangerous-capabilities?lang=nl).",
            "luck": "Voor de beoordelaars was één succes op 10 genoeg om de capability te laten zien. Zie [Testen op gevaarlijke capabilities](/course/open/rai-dangerous-capabilities?lang=nl).",
        },
    )
    quick_kimi
    return (quick_kimi,)


@app.cell(hide_code=True)
def quick_check_boundary():
    quick_boundary = L.choice(
        "quick-boundary",
        question="Een logistische regressie geeft een punt de kans σ(z), met z de logit. Welke kans krijgt een punt op de decision boundary?",
        options={
            "zero": "0, omdat het op de lijn ligt",
            "right": "0,5, want de logit is daar 0",
            "one": "1, want op de lijn is het model het zekerst",
        },
        correct="right",
        why={
            "right": "Op de decision boundary is de logit 0, en σ(0) = 0,5. In deze les geeft een curve van dezelfde soort de kans dat een AI-agent een taak voltooit, en de taaklengte waarbij die curve 0,5 geeft, is de maat.",
            "zero": "De sigmoid geeft nooit precies 0: op de decision boundary is de logit 0, en σ(0) = 0,5. Zie [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl).",
            "one": "Op de lijn is het model juist het minst zeker: de logit is 0 en σ(0) = 0,5. Zie [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl).",
        },
    )
    quick_boundary
    return (quick_boundary,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een test op een gevaarlijke capability vraagt naar één taak tegelijk: kan dit model dat? Wie wil volgen hoe snel AI-agents in het algemeen beter worden, heeft één schaal nodig voor taken die zo verschillen als een kleine bug oplossen en een model trainen. METR, een onderzoeksorganisatie die AI-modellen evalueert, koos tijd als die schaal: hoe lang vakmensen over de taken doen. Op die schaal kun je zeggen hoe lange taken een model aankan, en hoe snel die lengte groeit. Voor veiligheid doet dat ertoe: hoe langer de taken die een agent zelfstandig afmaakt, hoe meer van het werk van AI-onderzoek zelf een agent zou kunnen overnemen, en METR gebruikt de maat om dat te volgen.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In de data van METR van september 2026 heeft o3, een model van OpenAI dat in april 2025 uitkwam, een time horizon van ongeveer 2 uur: op de softwaretaken van METR waar vakmensen ongeveer 2 uur over doen, slaagt het volgens de voorspelling in de helft van de gevallen. Bij taken van welke lengte voorspelt dezelfde meting dat o3 in 80% van de gevallen slaagt?",
        options={
            "right": "Ongeveer 30 minuten",
            "most": "Ongeveer 1 uur en 40 minuten",
            "hour": "Ongeveer 1 uur, de helft daarvan",
        },
        reveal="**Ongeveer 30 minuten.** METR geeft o3 een 50%-time horizon van 119,7 minuten en een 80%-time horizon van 30,0 minuten: vier keer zo kort. Thomas Kwa en collega's van METR vonden in hun paper een vergelijkbaar verschil: daar zijn de 80%-horizons 4 tot 6 keer zo kort. Een beetje meer betrouwbaarheid vragen kost veel taaklengte; deze les laat zien waarom.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Taken gemeten in menselijke tijd

    Thomas Kwa en collega's van METR (*Measuring AI Ability to Complete Long Tasks*, eerste versie van 18 maart 2025; de vierde versie, van 10 juli 2026, heet *Measuring AI Ability to Complete Long Software Tasks*) testten AI-agents: AI-modellen met tools die in veel stappen aan een taak werken. Ze gebruikten 170 taken over software, machine learning en cybersecurity, elk met een duidelijke manier om te scoren: 97 van HCAST, van 1 minuut tot 30 uur, 7 van RE-Bench, van elk 8 uur, en 66 korte van 1 tot 30 seconden. Vakmensen met gemiddeld zo'n 5 jaar relevante ervaring deden dezelfde taken, in meer dan 800 pogingen van samen 2529 uur. De lengte van een taak is het meetkundig gemiddelde van de tijden van de mensen die slaagden: vermenigvuldig de $n$ tijden en neem de $n$-de wortel. Voor 20 en 80 minuten is dat $\sqrt{20 \times 80} = 40$ minuten, het midden op een logaritmische schaal, waar het gewone gemiddelde 50 geeft. Voor 21 taken schatten de onderzoekers de tijd zelf. Daarna deed elke agent elke taak 8 keer.

    Hun maat is de 50%-time horizon: de tijd die mensen meestal nodig hebben voor taken die AI-modellen in de helft van de gevallen voltooien. Een **time horizon** is zo'n taaklengte, in de tijd die vakmensen nodig hebben, waarbij een AI-agent slaagt met een gegeven kans; zonder kans erbij is dat 50%. Het is niet hoe lang een agent zelfstandig kan werken. De pagina van METR over time horizons stelt de vraag of een time horizon betekent hoe lang huidige AI-agents zelfstandig kunnen werken, en antwoordt met nee: het is een maat voor hoe moeilijk een taak is, niet voor de tijd die een AI aan de taak besteedt. Op de taken die ze voltooien, zijn agents meestal een aantal keer zo snel als mensen, voegt METR toe.

    ### Eén curve per agent

    Kwa en collega's vonden dat de kans op succes van een model daalt naarmate een mens langer over de taak doet. Voor elke agent fitten ze een curve, zoals bij logistische regressie met één feature: de log van de taaklengte.
    """)
    return


@app.cell(hide_code=True)
def text_idea_curve():
    mo.vstack([
        L.formula(r"p_{\text{succes}} = \sigma\big((\log_2 [[param:h]] - \log_2 [[data:t]]) \cdot [[param:\beta]]\big)"),
        mo.md(r"""
        Hier is $t$ de lengte van de taak, $h$ de time horizon van de agent, en $\beta$ hoe steil de kans daalt; $h$ en $\beta$ worden geleerd uit de resultaten van de agent. Bij $t = h$ is de logit 0 en de kans $\sigma(0) = 0{,}5$: de time horizon ligt waar de curve 0,5 kruist, zoals de decision boundary ligt waar een logistische regressie 0,5 geeft. Het paper zegt niet welk grondtal de log heeft; de code van METR fit de curve op de log met grondtal 2 van de menselijke minuten. Met grondtal 2 verlaagt elke verdubbeling van de taaklengte de logit met $\beta$, dus dalen de odds op succes bij elke verdubbeling met dezelfde factor: $e^{\beta}$, ongeveer 1,8 bij $\beta = 0{,}6$.

        Dezelfde curve geeft ook andere niveaus. METR zoekt de taaklengte waar de gefitte curve 50% kruist, of 80%. En 50% is een punt op de curve, niet de kans bij elke taak. Over GPT-5, een model van OpenAI, schrijft METR op de pagina dat het bij taken van 90 minuten tot 3 uur ongeveer een derde van de taken altijd haalt, een derde nooit, en de rest soms wel en soms niet.

        ### Ongeveer elke zeven maanden een verdubbeling

        Daarna zetten Kwa en collega's de time horizon van elk model uit tegen de releasedatum, en fitten ze een rechte lijn door de log van de horizons: in de vierde versie verdubbelde de time horizon van 2019 tot en met 2025 elke 207 dagen (in de eerste versie elke 212 dagen). Op een logaritmische schaal is groei met dezelfde factor in gelijke stappen een rechte lijn, en de verdubbelingstijd is de tijd waarin de lijn één verdubbeling stijgt. GPT-2 had een 50%-time horizon van maar 2 seconden, o3 een van 110 minuten. Ze vertrouwen de lijn meer dan de punten: over de helling van de trend zijn ze zekerder dan over de time horizon van welk model ook. Een verdubbeling elke 207 dagen betekent ongeveer 3,4 keer zo lange taken per jaar, volgens een eigen berekening van de course.

        Waar leidt dat heen? Als je de trend naïef doortrekt, haalt AI volgens hen een time horizon van meer dan een maand, 167 werkuren, ergens tussen medio 2028 en medio 2031. Ze voegen toe dat dat ervan afhangt of de trend ook bij echt werk geldt en of die verandert. En waarom een maand ertoe doet: ze kunnen niet uitsluiten dat een systeem dat taken van een maand aankan, zelfs met een kans van 50%, de samenleving ingrijpend zou kunnen veranderen, mogelijk ook met het vermogen om catastrofale schade aan te richten.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_idea_later():
    mo.md(r"""
    ### Latere metingen

    METR bleef meten. In *Time Horizon 1.1* (29 januari 2026), met 228 taken in plaats van 170, was de verdubbelingstijd over alle jaren 196 dagen, net als met de oude taken op dat moment, en vanaf 2023 was die 131 dagen, tegen 165 dagen met de oude taken. Op 3 maart 2026 herstelde METR een fout in de manier waarop het de curves fitte, die de metingen had beïnvloed. De pagina over time horizons, voor het laatst bijgewerkt op 8 september 2026, geeft de horizons van de modellen die METR mat onder de licentie CC BY 4.0, zodat iedereen ze mag gebruiken; o3 heeft daar 119,7 minuten, tegen 110 in het paper. De pagina waarschuwt dat metingen boven 16 uur met de huidige taken onbetrouwbaar zijn. In oefening 2 fit je die data zelf.

    Op 20 november 2026 meldde METR, over een moeilijkere verzameling taken samen met de oude, dat time horizons ruwweg elke 3,6 maanden verdubbelen, en dat de huidige publieke frontier models op het controleerbare deel van de nieuwe taken een geschatte time horizon halen die oploopt tot 27 uur. Binnen een jaar was de oude verzameling, met taken tot 30 uur, te makkelijk geworden voor de beste modellen. Met twee waarschuwingen: METR maakt zich steeds meer zorgen dat de eigen tests, die makkelijk te draaien zijn in een afgesloten omgeving, te hoog kunnen inschatten wat modellen in de echte wereld aankunnen, omdat zulke omgevingen nu veel bij training worden gebruikt; en maar 9% van de schattingen gebruikt rechtstreeks de tijden van mensen.

    ### Wat een time horizon niet zegt

    Volgens Kwa en collega's is een time horizon altijd gemeten ten opzichte van een domein, een verzameling taken, en de vaardigheid en voorkennis van de mensen die de tijden zetten. Bij een klein aantal reparaties aan de eigen code van METR deden ingehuurde krachten er 5 tot 18 keer zo lang over als de mensen die de code onderhouden. METR raadt aan een taak van 2 uur te lezen als wat iemand zonder voorkennis, zoals een nieuwe medewerker of een freelancer, in 2 uur kan afmaken. Andere soorten taken geven andere horizons. In een post van 14 juli 2025 vonden Thomas Kwa en Vincent Cheng dat de horizons voor computergebruik via het scherm 40 tot 100 keer zo kort zijn als die voor software en redeneren, maar ongeveer even snel groeien. En zelfrijdende auto's van Tesla verbeteren langzamer, met zo'n 0,6 verdubbeling per jaar. Veel korte taken maken ook geen lange: volgens de pagina van METR zijn 1000 losse wiskundeopgaven van een uur geen taak van 1000 uur.

    ### Waar ze het oneens zijn

    Over hoe snel: 207 dagen in het paper, 196 of 131 dagen in *Time Horizon 1.1*, ongeveer 3,6 maanden in november 2026. Tussen deze getallen verschillen de taken, de jaren en de methode, dus ze laten zowel een snellere trend als een verschuivende maat zien; de blog van METR van maart 2025 hield een ruime marge aan, van ongeveer 1 tot 4 verdubbelingen per jaar. Over welk niveau je opgeeft: het International AI Safety Report 2026 gebruikt de 80%-horizon, en schrijft dat AI-systemen goed omschreven software-engineeringtaken waar experts 30 minuten over doen, in ongeveer 80% van de gevallen voltooien, en dat die duur elke zeven maanden verdubbelt; METR zet 50% voorop, en getallen uit de twee kun je niet zomaar vergelijken. En over wat er komt: METR fit op de pagina een gestage verdubbeling aan de trend in de tijd, omdat een S-vormige curve die afvlakt slecht past zolang er geen teken is dat de groei vertraagt, terwijl het eigen rapport van november 2026 waarschuwt dat de tests te hoog kunnen uitvallen, zodat een gestage trend op de tests niet hoeft te gelden voor echt werk. Kwa en collega's noemen hun eigen voorspelling naïef.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **time horizon** | de lengte van de taken, in de tijd die vakmensen ervoor nodig hebben, waarbij een AI-agent slaagt met een gegeven kans; zonder kans erbij 50% |
        | AI-agent | een AI-model met tools dat in veel stappen aan een taak werkt |
        | menselijke baseline | een vakmens die een taak doet en daarbij getimed wordt, om te weten hoe lang de taak duurt |
        | meetkundig gemiddelde | de $n$-de wortel van het product van $n$ getallen: het midden op een logaritmische schaal |
        | $t$, $h$ | de lengte van een taak en de time horizon van een agent, allebei in minuten voor vakmensen |
        | $\beta$ | hoeveel de logit van een agent daalt per verdubbeling van de taaklengte: hoe steil de curve is |
        | verdubbelingstijd | de tijd waarin iets dat in gelijke stappen met dezelfde factor groeit, verdubbelt |
        """),
        L.watch_out("Een time horizon is niet hoe lang een model zelfstandig kan werken: het is hoe lang de taken mensen kosten. Elke time horizon geldt voor één kans op succes, voor de taken van METR, vooral software, en voor mensen zonder voorkennis van die taken. De verzonnen gevallen van de oefeningen, de niveaus boven 80% in de demo en de vergelijkingen van getallen tussen bronnen zijn van de course zelf."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo lees je een time horizon:

    1. Vraag bij welke kans op succes die gegeven is: 50%, 80% of een andere.
    2. Vraag welke taken er gebruikt zijn, en welke mensen hun lengte bepaalden.
    3. Lees het getal als hoe moeilijk de taken zijn in menselijke tijd, niet als hoe lang het model werkt.
    4. Voor één agent is de kans bij een andere lengte $\sigma((\log_2 h - \log_2 t) \cdot \beta)$.
    5. Voor de trend fit je een rechte lijn door $\log_2$ van de horizons tegen de releasedatums; één gedeeld door de helling is de verdubbelingstijd.
    6. Geef bij elk getal een datum en een versie.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.vstack([
        mo.md(r"""
        ### Uitgewerkt voorbeeld

        Stap 5 met twee modellen. Als de horizon in $d$ dagen groeit van $h_a$ van het ene model naar $h_b$ van een later model, is het aantal verdubbelingen $\log_2(h_b / h_a)$, en duurde elke verdubbeling:
        """),
        L.formula(r"\text{verdubbelingstijd} = \frac{[[data:d]]}{\log_2([[data:h_b]] / [[data:h_a]])}"),
        mo.md(r"""
        De cel bevat de beste modellen van METR vanaf 2023, uit de data van september 2026; net als de fit van METR zelf laat de cel het ene model boven 16 uur weg. De cel rekent het uit voor GPT-4 naar GPT-5 en voor o1 naar o3, alle vier van OpenAI.
        """),
    ])
    return


@app.cell
def example_doubling():
    # METR, Time Horizon 1.1, data van 8 september 2026 (CC BY 4.0):
    # de modellen die METR bij hun release als beste aanmerkt
    models = ["GPT-4", "GPT-4 (2023-11)", "GPT-4o", "Anthropic (2024-06)", "o1-preview",
              "Anthropic (2024-10)", "o1", "Anthropic (2025-02)", "o3", "GPT-5", "Gemini 3 Pro",
              "Anthropic (2025-11)", "GPT-5.2", "Anthropic (2026-02)"]
    days = [0, 237, 426, 464, 548, 588, 632, 713, 764, 877, 980, 986, 1003, 1059]   # dagen na 14 maart 2023
    # de 50%-horizon van elk, in minuten
    minutes = [4.0, 4.0, 7.0, 11.4, 20.3, 20.5, 38.8, 60.4, 119.7, 203.0, 224.3, 293.0, 352.2, 718.8]

    for a, b in [(0, 9), (6, 8)]:   # GPT-4 naar GPT-5, en o1 naar o3
        doublings = np.log2(minutes[b] / minutes[a])
        each = round((days[b] - days[a]) / doublings)
        print(models[a], "naar", models[b], round(doublings, 2), "verdubbelingen, elk", each, "dagen")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Twee paren, twee antwoorden: 155 dagen per verdubbeling van GPT-4 naar GPT-5, 81 dagen van o1 naar o3. Kwa en collega's merkten op dat o3 boven de trend op lange termijn ligt, dus geeft een paar dat met o3 eindigt een snelle verdubbeling. Een rechte lijn door alle modellen laat elk model meetellen, en in oefening 2 fit je die.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_success():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. success: de kans bij een taaklengte

        `success(horizon, minutes, beta)` geeft de kans terug dat een agent met deze 50%-time horizon, in minuten, en deze `beta` volgens de curve van METR een taak voltooit waar vakmensen `minutes` minuten over doen. Reken de logit `z` uit: de log met grondtal 2 van `horizon` min de log met grondtal 2 van `minutes`, keer `beta`, met `np.log2`. Geef de sigmoid ervan terug, met `np.exp` zoals in [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl).

        De check roept `success(120, 30, 0.6)` aan: een horizon van 2 uur, een taak van een half uur, en een `beta` van 0,6. Daarna zeven andere gevallen.
        """),
        L.task("De kans dat een agent een taak van een gegeven lengte voltooit."),
    ])
    return


@app.cell(hide_code=True)
def exercise_success():
    box_success = L.code_box("success", lines=6)
    box_success
    return (box_success,)


@app.cell(hide_code=True)
def check_success(box_success):
    result_success = L.check("success", box_success.value)
    result_success
    return (result_success,)


@app.cell(hide_code=True)
def hints_success():
    L.hints(
        "success",
        [
            "Doe het met de hand voor `success(120, 30, 0.6)`: hoeveel verdubbelingen ligt een taak van 30 minuten onder een horizon van 120? Elke verdubbeling onder de horizon telt `beta` op bij de logit. Welke kans geeft de sigmoid bij die logit?",
            "Schrijf `z = (... - ...) * beta`. Wat neem je van elke lengte, zodat één verdubbeling één stap is, en welke komt eerst, zodat een taak die korter is dan de horizon een positieve logit krijgt? Daarna `return 1 / (1 + np.exp(-z))`.",
            "`    z = (np.log2(horizon) - ___) * beta`",
        ],
        why="`z = (np.log2(horizon) - np.log2(minutes)) * beta` en `return 1 / (1 + np.exp(-z))`. Met de hand: 30 minuten ligt twee verdubbelingen onder 120, dus is de logit 2 × 0,6 = 1,2, en de kans 1 / (1 + e^(−1,2)), ongeveer 0,7685.",
    )
    return


@app.cell(hide_code=True)
def text_after_success():
    mo.md(r"""
    Een taak zo lang als de horizon krijgt 0,5; twee verdubbelingen korter 0,77; twee verdubbelingen langer 0,23; vier verdubbelingen korter, 7,5 minuten, 0,92. De curve daalt bij elke verdubbeling met dezelfde stap in de logit, en een grotere `beta` maakt de curve steiler: met een `beta` van 1,0 en een horizon van 60 minuten krijgt een taak van 15 minuten 0,88 en een van 240 minuten 0,12.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: meer dan de helft van de keren

    METR zet de 50%-time horizon voorop, en geeft ook de 80%-horizon. In de data van september 2026 heeft o3 een 50%-horizon van 119,7 minuten en een 80%-horizon van 30,0 minuten; uit die twee volgt de $\beta$, ongeveer 0,69, volgens een eigen berekening van de course: 30 minuten ligt bijna precies twee verdubbelingen onder 119,7, en bij 80% is de logit $\ln(0{,}8/0{,}2) = \ln 4 \approx 1{,}39$, dus $\beta \approx 1{,}39 / 2$. Kies de kans op succes die je wilt, en zie bij welke taaklengte de curve van o3 die geeft.
    """)
    return


@app.cell(hide_code=True)
def demo_reliable():
    explore_reliable = L.explore(
        "reliable", "kans op succes (%)",
        lambda _v: C.demo_frame(_v, L.lang),
        start=C.P_START, values=C.P_VALUES, role="hyper",
        task="Zoek de laagste kans op succes waarbij de horizon van o3 onder 10 minuten ligt.",
    )
    explore_reliable
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Vanaf 93% ligt de horizon van o3 onder 10 minuten, en bij 99% is die net iets meer dan een minuut: 98 keer zo kort als bij 50%. Zulke niveaus meldt METR niet: om een 99%-horizon goed te meten zijn volgens METR veel meer taken nodig, heel korte, en de uitkomsten zouden meer afhangen van hoe je meet. De niveaus boven 80% in de demo zijn van de course zelf, afgelezen van dezelfde curve. Een taak die ertoe doet, vraagt vaak meer dan kop of munt, en daar is de horizon veel korter.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_doubling():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. doubling_days: een lijn door elk model

        `doubling_days(days, minutes)` geeft de verdubbelingstijd terug, in dagen, van de rechte lijn door de log met grondtal 2 van de horizons `minutes` tegen de releasedatums `days`. Fit de lijn met `np.polyfit`, dat eerst de helling `slope` teruggeeft en dan `start`, waar de lijn dag 0 kruist; met de horizons geteld in verdubbelingen is de helling het aantal verdubbelingen per dag. Geef het aantal dagen per verdubbeling terug.

        De check roept `doubling_days(days, minutes)` aan, met de 14 modellen van METR uit het uitgewerkte voorbeeld. Daarna vijf andere gevallen.
        """),
        L.task("De verdubbelingstijd van de time horizon, uit alle modellen tegelijk."),
    ])
    return


@app.cell(hide_code=True)
def exercise_doubling():
    box_doubling = L.code_box("doubling", lines=6)
    box_doubling
    return (box_doubling,)


@app.cell(hide_code=True)
def check_doubling(box_doubling):
    result_doubling = L.check("doubling", box_doubling.value)
    result_doubling
    return (result_doubling,)


@app.cell(hide_code=True)
def hints_doubling():
    L.hints(
        "doubling",
        [
            "Doe het met de hand voor `doubling_days([0, 10, 20], [1, 2, 4])`: tel de horizons in verdubbelingen boven 1, dus 0, 1 en 2. Hoeveel verdubbelingen per dag is dat, en hoeveel dagen per verdubbeling?",
            "`np.polyfit(x, y, 1)` fit een rechte lijn door `y` tegen `x`. Welke van de twee lijsten is hier `x`, en wat doe je eerst met de andere, zodat één stap omhoog één verdubbeling is? De helling is dan het aantal verdubbelingen per dag; geef één gedeeld door die helling terug.",
            "`    slope, start = np.polyfit(___, 1)`",
        ],
        why="`slope, start = np.polyfit(days, np.log2(minutes), 1)` en `return 1 / slope`. Met de hand: 0, 1 en 2 verdubbelingen op dag 0, 10 en 20 is een helling van 0,1 verdubbeling per dag, dus duurt één verdubbeling 10 dagen.",
    )
    return


@app.cell(hide_code=True)
def text_after_doubling():
    mo.md(r"""
    De 14 beste modellen van METR vanaf 2023 geven een verdubbeling elke 128,7 dagen, dicht bij de 131 dagen uit de post van METR van januari 2026; het bestand van METR zelf geeft, na de correctie van maart 2026, 128,744 dagen uit de niet-afgeronde getallen. Met de drie oudere modellen van 2019 tot en met 2022 erbij geeft het bestand van METR 187,8 dagen. Bij twee modellen gaat de lijn door allebei, en geeft die de 155 dagen van het uitgewerkte voorbeeld. Elk nieuw model verschuift de lijn een beetje, en een schatting als 128,7 heeft een ruime marge: het bestand van METR geeft 104 tot 158 dagen.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_headline():
    mo.md(r"""
    ### Oefening 3. Een kop op een nieuwssite

    Een nieuwssite schrijft: nieuw AI-model kan nu 12 uur zelfstandig werken. Het bericht is gebaseerd op een 50%-time horizon van ongeveer 12 uur die METR voor het model meldde.
    """)
    return


@app.cell(hide_code=True)
def choice_headline_cell():
    choice_headline = L.choice(
        "headline",
        question="Welke lezing past het best bij de maat van METR?",
        options={
            "run": "Het kan 12 uur zonder toezicht doorwerken voordat het stopt",
            "right": "Het slaagt ongeveer de helft van de keren bij softwaretaken van METR van 12 uur",
            "any": "Het kan elke klus van 12 uur doen, in elk vak, al gaat het half zo snel als een mens",
            "share": "Het haalt de helft van alle taken van METR, ongeacht hoe lang ze zijn",
        },
        correct="right",
        why={
            "right": "Een time horizon is een maat voor hoe moeilijk een taak is, niet voor de tijd die een AI eraan besteedt, op de taken van METR, vooral software, met de tijden van vakmensen. Bij kortere taken doet het model het beter, bij langere slechter. En met de oude taken waren metingen boven 16 uur onbetrouwbaar, schreef METR.",
            "run": "De pagina van METR stelt precies deze vraag, en het antwoord is nee. Een time horizon is hoe lang de taken mensen kosten, en op de taken die ze voltooien, zijn agents meestal een aantal keer zo snel als mensen. Zie *Taken gemeten in menselijke tijd*.",
            "any": "De horizon geldt voor de taken van METR, vooral software, machine learning en cybersecurity; voor computergebruik via het scherm waren horizons 40 tot 100 keer zo kort. En over snelheid zegt de horizon niets. Zie *Wat een time horizon niet zegt*.",
            "share": "De horizon is een taaklengte op een gefitte curve, geen aandeel van alle taken: kortere taken lukken vaker, langere minder vaak. Zie *Eén curve per agent*.",
        },
    )
    choice_headline
    return (choice_headline,)


@app.cell(hide_code=True)
def text_exercise_forecast():
    mo.md(r"""
    ### Oefening 4. Een plan voor 2030

    Een manager leest dat time horizons ongeveer elke zeven maanden verdubbelen, en dat de beste modellen in 2025 op zo'n 2 uur zaten. De conclusie van de manager: in 2030 doet een AI elke klus in ons bedrijf waar een mens een maand over doet.
    """)
    return


@app.cell(hide_code=True)
def choice_forecast_cell():
    choice_forecast = L.choice(
        "forecast",
        question="Welk antwoord past het best bij de bronnen van deze les?",
        options={
            "law": "Ja, de trend houdt al sinds 2019 zonder onderbreking stand, dus die gaat zeker zo door",
            "slowed": "Nee, METR vond dat de trend na 2025 vertraagde",
            "month": "Waarschijnlijk niet, want een maand werk is 720 uur, voorbij elke voorspelling",
            "right": "Misschien voor taken zoals die van METR, als de trend doorzet, maar niet elke klus",
        },
        correct="right",
        why={
            "right": "Kwa en collega's noemen hun voorspelling naïef: een maand softwarewerk tussen medio 2028 en medio 2031, met een kans van 50%, als de trend doorgaat en ook voor echt werk geldt. De taken van METR zijn goed omschreven en worden door een programma gescoord; volgens de pagina van METR bestaan de meeste banen niet uit zulke goed omschreven taken met een vaste uitkomst. En in november 2026 waarschuwde METR dat de eigen tests te hoog kunnen uitvallen.",
            "law": "Een trend is geen wet: Kwa en collega's noemen de voorspelling naïef, en voegen toe dat die ervan afhangt of de trend ook bij echt werk standhoudt en of die verandert. Zie *Ongeveer elke zeven maanden een verdubbeling*.",
            "slowed": "METR vond het omgekeerde: in november 2026 meldde het een verdubbeling ongeveer elke 3,6 maanden. Zie *Latere metingen*.",
            "month": "Hier is een maand werk 167 werkuren, geen 30 dagen van 24 uur, en de voorspelling van Kwa en collega's haalt die tussen medio 2028 en medio 2031. Zie *Ongeveer elke zeven maanden een verdubbeling*.",
        },
    )
    choice_forecast
    return (choice_forecast,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Testen op gevaarlijke capabilities](/course/open/rai-dangerous-capabilities?lang=nl) vroeg of een model één gevaarlijk ding kan; de time horizon zet veel taken op één schaal en laat zien hoe snel modellen daarlangs opschuiven. De curve is die van [Logistische regressie en de decision boundary](/course/open/ml-logistic-regression?lang=nl), met één feature: de time horizon ligt waar de curve 0,5 kruist, zoals de decision boundary. En net als in [Logaritmische schalen en machtswetten](/course/open/math-log-scales?lang=nl) is groei met dezelfde factor een rechte lijn op een logaritmische schaal, dus is de verdubbelingstijd één gedeeld door de helling van een lijn. Dit was de laatste les van de module alignment.

    **Een voorbeeld uit het dagelijks leven.** Een klimhal geeft elke route een graad naar hoe zwaar die is voor ervaren klimmers. Het niveau van een beginner is de graad waarbij de helft van de routes lukt: makkelijkere routes bijna altijd, zwaardere zelden. Dat niveau zegt niets over hoe lang de beginner achter elkaar kan klimmen, en de graad waarbij negen op de tien routes lukken, ligt lager. Graden horen ook bij één hal: een andere hal, of een rots buiten, geeft andere graden.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_forecast, choice_headline, result_doubling, result_success):
    L.wrap_up(result_success, result_doubling, choice_headline, choice_forecast)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
