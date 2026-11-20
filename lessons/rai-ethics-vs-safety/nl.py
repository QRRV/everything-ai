# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-ethics-vs-safety. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow AI Safety, longtermism, artificial general intelligence, Stochastic Parrots

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-ethics-vs-safety", "nl")
    C = L.checks   # de getallen van Greaves en MacAskill: drie plannen, drie schattingen van de toekomst, de klamboes


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Soorten risico's van AI](/course/open/rai-kinds-of-ai-risk?lang=nl) en [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_risk():
    quick_risk = L.choice(
        "quick-risk",
        question="Veel nieuwssites gaan hun berichten schrijven met hetzelfde AI-model. Geen model maakt een fout en niemand wil schade doen, toch gaat het nieuws steeds meer op elkaar lijken. Wat voor risico is dit, in de termen van Zwetsloot en Dafoe?",
        options={
            "misuse": "Misbruik: de nieuwssites gebruiken het model om hun lezers te schaden",
            "right": "Structureel: het komt voort uit hoe AI de omgeving en de prikkels verandert",
            "accident": "Een ongeluk: het model doet iets wat zijn makers niet bedoelden, zoals fouten in berichten schrijven",
        },
        correct="right",
        why={
            "right": "Niemand misbruikt het model en het faalt niet; de schade komt doordat veel sites dezelfde keuze maken. Dat is het structurele perspectief dat Zwetsloot en Dafoe naast misbruik en ongelukken zetten.",
            "misuse": "Niemand wil hier schade doen, en voor misbruik is iemand nodig die dat wel wil. Zie [Soorten risico's van AI](/course/open/rai-kinds-of-ai-risk?lang=nl).",
            "accident": "Het model doet waarvoor het gemaakt is: berichten schrijven, zonder fouten. De schade komt doordat veel sites dezelfde keuze maken. Zie [Soorten risico's van AI](/course/open/rai-kinds-of-ai-risk?lang=nl).",
        },
    )
    quick_risk
    return (quick_risk,)


@app.cell(hide_code=True)
def quick_check_theory():
    quick_theory = L.choice(
        "quick-theory",
        question="Een ziekenhuis kan een programma betalen dat dit jaar 10 levens redt, of een dat 12 levens redt maar een belofte aan zijn patiënten breekt. Welke theorie zegt dat het tweede beter is omdat er in totaal meer levens worden gered?",
        options={
            "duty": "Deontologie: een belofte telt, wat de uitkomst ook is",
            "right": "Consequentialisme: het telt op wat een keuze teweegbrengt",
            "virtue": "Deugdethiek: die vraagt wat iemand met een goed karakter in de plaats van het ziekenhuis zou doen",
        },
        correct="right",
        why={
            "right": "Het consequentialisme beoordeelt een keuze op wat ze teweegbrengt, opgeteld over iedereen: 12 levens zijn meer dan 10. Deze les komt hetzelfde soort som tegen, met veel grotere getallen.",
            "duty": "De deontologie is juist de theorie die bezwaar zou maken: voor de deontologie telt de gebroken belofte, wat de uitkomst ook is. Zie [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).",
            "virtue": "De deugdethiek vraagt naar karakter, niet naar totalen. Zie [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl).",
        },
    )
    quick_theory
    return (quick_theory,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In het voorjaar van 2023 verschenen er twee verklaringen over AI, twee maanden na elkaar. De ene zei dat de schade van AI echt is en nu al gebeurt; de andere dat het risico op uitsterven door AI wereldwijd voorrang moet krijgen. Beide kwamen van mensen die onderzoeken wat er mis kan gaan met AI, en ze zijn het oneens over waar je eerst moet kijken.

    Deze les kijkt naar de twee groepen, AI-ethiek en AI-veiligheid, naar het argument waarmee een deel van de tweede groep de verre toekomst voorop zet, naar de kritiek op dat argument, en naar wat de twee delen. De sommen van dat argument maak je zelf opnieuw.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Een veilige optie redt zeker 1000 levens. Een riskante optie heeft een kans van één op een biljoen (10⁻¹²) om een triljoen (10¹⁸) levens te redden, en redt anders niemand. Als je elke uitkomst weegt naar zijn kans, hoeveel levens redt de riskante optie dan gemiddeld?",
        options={
            "none": "Bijna geen: hij redt bijna altijd niemand",
            "right": "Een miljoen: duizend keer zoveel",
            "same": "Ongeveer 1000: de kans maakt de omvang goed",
        },
        reveal="**Een miljoen: duizend keer zoveel.** 10¹⁸ × 10⁻¹² = 10⁶. Toch redt hij bijna altijd niemand. Het voorbeeld is van Greaves en MacAskill zelf, bij de zorg die ze fanatisme noemen: *Hun eigen twijfels* onder *Het idee* komt erop terug.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Twee groepen, één ruzie

    De eerdere modules van dit domein, van ethiek en fairness tot privacy, horen bij wat meestal AI-ethiek heet: schade die AI-systemen nu doen, aan mensen die nu leven, zoals beslissingen die groepen benadelen, uitgebuite werkers en macht in weinig handen. Deze module gaat over AI-veiligheid, die ook kijkt naar rampen die veel krachtigere systemen later zouden kunnen veroorzaken. In 2023 botsten de twee in het openbaar.

    In maart 2023 publiceerde het Future of Life Institute een brief die vroeg om een pauze van minstens zes maanden in het trainen van AI-systemen die krachtiger zijn dan GPT-4; volgens de critici hadden er op 31 maart meer dan 2000 mensen hem ondertekend. Die dag schreven Timnit Gebru, Emily M. Bender en Angelina McMillan-Major, drie van de auteurs van het paper dat bekendstaat als "Stochastic Parrots", in een verklaring op de website van het DAIR Institute dat de schade van zogenaamde AI echt is en nu al gebeurt, en voortkomt uit wat mensen en bedrijven doen die geautomatiseerde systemen inzetten. De hypothetische risico's uit de brief staan volgens hen centraal in een gevaarlijke ideologie, het longtermisme, die de echte schade negeert van de AI-systemen die nu worden ingezet.

    Op 30 mei 2023 zette het Center for AI Safety één zin online: het beperken van het risico op uitsterven door AI zou wereldwijd voorrang moeten krijgen, naast andere risico's op de schaal van de samenleving, zoals pandemieën en kernoorlog. Op de pagina staan in november 2026 als eerste namen Geoffrey Hinton, Yoshua Bengio, Demis Hassabis, Sam Altman en Dario Amodei. De zin noemt geen kans en geen datum.

    ### Longtermisme

    **Longtermisme** is volgens de filosoof William MacAskill de opvatting dat we veel meer zouden moeten doen om toekomstige generaties te beschermen. Het berust op drie ideeën: toekomstige mensen tellen moreel mee, er kunnen heel veel toekomstige mensen komen, en wat we nu doen kan bepalen hoe goed of slecht hun leven wordt. Daarom richten longtermisten zich op existentiële risico's, risico's die het langetermijnpotentieel van de mensheid dreigen te vernietigen; de risico's die ze noemen zijn onder meer geavanceerde kunstmatige intelligentie, kunstmatig gemaakte ziekteverwekkers en kernoorlog.

    In een working paper van juni 2021, *The case for strong longtermism*, gaan Hilary Greaves en William MacAskill verder: sterk longtermisme is volgens hen de opvatting dat het effect op de verre toekomst het belangrijkste is aan wat we nu doen. De verre toekomst begint in hun voorbeeld 100 jaar na een beslissing.

    ### De som erachter

    Greaves en MacAskill onderbouwen het met getallen. Als benchmark voor het goede dat je nu kunt doen, nemen ze klamboes tegen malaria, die een leven redden voor ongeveer 4000 dollar: 0,025 levens per 100 dollar. Ze gebruiken de klamboes als ruwe bovengrens voor wat je per uitgegeven dollar op korte termijn aan goeds kunt bereiken. Daartegenover zetten ze plannen die het risico op een ramp in de komende 100 jaar verkleinen: het uitsterven van de mensheid bij asteroïden en pandemieën, en bij AI een ramp die ze minstens zo erg rekenen. Zo'n ramp zou alle toekomstige levens kosten, dus de levens die een plan gemiddeld redt, zijn de daling van dat risico maal het verwachte aantal toekomstige levens: het getal dat je krijgt als je elke mogelijke toekomst weegt naar zijn kans, een verwachtingswaarde, die [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl) helemaal behandelt.

    Hoeveel toekomstige levens? Hun tabel geeft zes scenario's, in het plaatje hieronder, van een mensheid die een miljoen jaar op aarde blijft, 10¹⁴ levens, tot digitaal leven door de hele Melkweg, 10⁴⁵. Ze werken met drie schattingen: de hoofdschatting, 10²⁴, volgens hen het minste wat redelijk is; de lage, 10¹⁸; en de beperkte, 10¹⁴, alsof de mensheid op aarde blijft en niets groters enige kans heeft.

    Hun drie plannen, die de course in `C.PLANS` bewaart, elk met een risico, het deel ervan dat het plan weghaalt (in de code `cut`) en kosten:

    - asteroïden: de kans dat in de komende 100 jaar een asteroïde van 10 km of meer de aarde raakt, wordt geschat op 1 op 150 miljoen, en ze nemen aan dat 1% van zulke inslagen de mensheid zou uitroeien. Alle overgebleven asteroïden vinden kost hoogstens 1,2 miljard dollar, en als er een op ramkoers wordt gevonden, halen afbuigen en voorbereiden 5% van het risico weg.
    - pandemieën: de kans op een pandemie die de mensheid uitroeit, wordt voor de komende 100 jaar geschat op ongeveer 1 op 22.000, een gemiddelde dat ze nemen van twee van drie gepubliceerde schattingen, die lopen van 1 op 600.000 tot 1 op 50; en 250 miljard dollar voor een sterkere gezondheidszorg zou daar minstens 1% van weghalen.
    - AI: zelfs een zeer voorzichtige inschatting geeft volgens hen minstens 0,1% kans op een ramp door AI in de komende eeuw, even erg als het uitsterven van de mensheid of erger, en 1 miljard dollar aan gericht uitgegeven geld zou die ramp in minstens 1% van de gevallen voorkomen.

    Voor deze sommen tellen ze waarde als totaal welzijn, het consequentialistische soort som uit de instapcheck. Ze betogen ook dat een versie van hun stelling zonder dat standpunt overeind blijft: een deontische versie van sterk longtermisme zou volgens hen best waar kunnen zijn, ook als het consequentialisme niet klopt.
    """)
    return


@app.cell(hide_code=True)
def size_cell():
    mo.Html(C.size_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### Hun eigen twijfels

    Greaves en MacAskill noemen de zwakke plekken zelf: de schatting van wat een poging om de verre toekomst te verbeteren per dollar oplevert, hoe je moet omgaan met het feit dat we de verre gevolgen van ons handelen nauwelijks kunnen overzien, en de vraag of rekenen met verwachtingswaarden hier te fanatiek is. De gok aan het begin van de les is hun voorbeeld van fanatisme: een piepkleine kans op een enorm getal wint de som, terwijl hij bijna altijd op niets uitloopt. Ze zien het als een van de meest aannemelijke manieren waarop het argument voor sterk longtermisme zou kunnen falen. Toch concluderen ze voorzichtig dat de zorg over fanatisme hun argument niet ondermijnt, onder meer omdat de kansen op het niveau van de samenleving daarvoor misschien niet klein genoeg zijn. Hun stelling vergelijkt ook alleen: ze beweren niet, en geloven niet, dat opties alleen veel goeds kunnen doen als ze bijna het beste zijn voor de verre toekomst, of dat wat je op korte termijn kunt bereiken op zichzelf weinig is.

    ### TESCREAL

    De critici zien meer dan een som. In 2024 publiceerden Timnit Gebru en Émile P. Torres *The TESCREAL bundle: Eugenics and the promise of utopia through artificial general intelligence*. **TESCREAL** is een acroniem dat ze bedachten toen ze een eerste versie van dat paper schreven, voor zeven ideologieën: transhumanisme, extropianisme, singularitarianisme, (modern) kosmisme, rationalisme, effectief altruïsme en longtermisme. Ze betogen dat het streven naar AGI, algemene kunstmatige intelligentie, uit deze bundel voortkomt, en dat het normatieve kader achter dat doel geworteld is in de Anglo-Amerikaanse eugenetica van de twintigste eeuw. Over AI-veiligheid schrijven ze dat bedrijven die naar AGI toewerken, zich door hun streven naar AGI een veiligheidskwestie te noemen kunnen presenteren als organisaties voor AI-veiligheid die de toekomst van de mensheid beschermen, terwijl ze tegelijk onveilige producten maken, macht naar zich toe trekken en zich aan verantwoording onttrekken.

    Ze voegen eraan toe dat niet iedereen die bij deze ideologieën hoort, alle overheersende opvattingen in de bundel deelt, en dat sommigen er zelfs bezwaar tegen kunnen hebben om zo gebundeld te worden. Gebru richtte DAIR op; volgens het paper zelf was de andere auteur bijna tien jaar een pleitbezorger van de beweging, voordat die auteur ermee brak. Ze citeren ook een versie van het paper van Greaves en MacAskill uit 2019, waarin staat dat effecten op korte termijn weinig meer zijn dan wat de doorslag geeft als al het andere gelijk is. Die zin staat niet in de versie van 2021 die hier wordt aangehaald.

    ### Meer gemeen dan het lijkt

    In 2020 schreven Carina Prunkl en Jess Whittlestone dat de grens tussen 'korte termijn' en 'lange termijn' slecht gedefinieerd is, en door verschillende mensen in verschillende situaties anders wordt gebruikt. Ze splitsen hem in vier vragen, elk een schaal en geen keuze uit twee: capabilities (huidige systemen of veel geavanceerdere), gevolgen (snel of veel later), zekerheid (goed begrepen of speculatief) en ernst (schade op elke schaal of vooral de grootste). In 2022 stelden Benjamin Bucknall en Shiri Dori-Hacohen voor dat sommige effecten van AI die al zijn vastgesteld, als existentiële risicofactoren kunnen werken: schade van nu die een ramp uit een andere bron waarschijnlijker maakt, of moeilijker te beantwoorden. Dan Hendrycks, de directeur van het Center for AI Safety, zei tegen BBC News dat risico's van later en zorgen van nu niet als tegenpolen gezien moeten worden: wie nu sommige problemen aanpakt, kan daarmee ook veel van de latere risico's helpen aanpakken. En de verklaring van DAIR was het eens met een paar voorstellen uit de brief die ze bekritiseerde, zoals systemen voor herkomst en watermerken die echte van synthetische media helpen onderscheiden.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **longtermisme** | de opvatting dat we veel meer zouden moeten doen om toekomstige generaties te beschermen; sterk longtermisme houdt in dat het effect op de verre toekomst het belangrijkste is aan wat we nu doen |
        | **TESCREAL** | de naam die Gebru en Torres gaven aan zeven verwante ideologieën, van transhumanisme tot longtermisme, die volgens hen het streven naar AGI aandrijven |
        | existentieel risico | een risico dat het langetermijnpotentieel van de mensheid dreigt te vernietigen, bijvoorbeeld door uitsterven |
        | verwacht aantal toekomstige levens | hoeveel wezens er nog zullen leven, met elke mogelijke toekomst gewogen naar zijn kans |
        | benchmark | waarmee een plan wordt vergeleken: hier de klamboes, 0,025 levens per 100 dollar |
        | `C.PLANS` | de drie plannen op naam, `"asteroids"`, `"pandemics"` en `"AI"`: dicts met de sleutels `"risk"` (de kans in de komende 100 jaar op uitsterven door die oorzaak, of bij AI op een ramp die minstens zo erg is), `"cut"` (het deel van dat risico dat het plan weghaalt) en `"cost"` (in dollars) |
        | `C.ESTIMATES` | het verwachte aantal toekomstige levens: `"main"` 10²⁴, `"low"` 10¹⁸ en `"restricted"` 10¹⁴ |
        | `C.BEDNETS` | de benchmark: 100 / 4000 = 0,025 levens per 100 dollar |
        | `1e24` | zo schrijft Python 10²⁴, een 1 met 24 nullen; `5e-16` is 5 × 10⁻¹⁶ |
        """),
        L.watch_out("Een som met een enorm getal erin is maar zo goed als zijn zwakste gok: hier de kans dat een plan werkt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    1. Vraag welke schade elke kant bedoelt, voor wie, en wanneer.
    2. Zoek de getallen achter een voorrang, en geef aan welke gemeten zijn en welke gegokt.
    3. Reken uit vanaf welke waarde van een gok de conclusie omslaat.
    4. Lees de critici en de verdedigers in hun eigen woorden, en zeg wie wat beweert.
    5. Beschrijf een zorg op meerdere schalen, zoals hoe krachtig de systemen zijn en hoe zeker de schade is, in plaats van met één etiket.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Vanaf 1996 bracht de Spaceguard Survey van NASA asteroïden in de buurt van de aarde in kaart, voor 71 miljoen dollar tot 2012. Greaves en MacAskill schatten dat dit de kans op uitsterven met minstens 5 × 10⁻¹⁶ per 100 dollar verlaagde. De cel rekent uit hoeveel levens dat waard is bij elk van hun drie schattingen, naast de klamboes.
    """)
    return


@app.cell
def example_spaceguard():
    print("bednets:", 100 / 4000, "lives per 100 dollars")
    drop = 5e-16                            # Spaceguard: de daling van de kans op uitsterven per 100 dollar
    for name in C.ESTIMATES:                # main, low en restricted
        lives = drop * C.ESTIMATES[name]    # levens gemiddeld per 100 dollar
        print(name, round(lives, 2), "lives per 100 dollars")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Dat zijn hun getallen: 500 miljoen, 500 en 0,05 levens per 100 dollar. Bij de hoofdschatting en de lage schatting doet de survey het veel beter dan de klamboes; bij de beperkte, 0,05 tegen 0,025, twee keer zo goed. De daling per 100 dollar was hier gegeven; in de oefening bereken je hem uit een risico, het deel dat een plan ervan weghaalt, en kosten.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lives():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. lives_per_100: wat 100 dollar van een plan waard is

        Schrijf `lives_per_100(plan, estimates)`. `plan` is een dict met de sleutels `"risk"`, `"cut"` en `"cost"`, zoals de plannen in `C.PLANS`; `estimates` is een dict met voor elke naam een aantal toekomstige levens, zoals `C.ESTIMATES`. Reken `drop` uit, het risico maal het deel dat het plan weghaalt: hoeveel de volledige kosten dat risico verlagen. Maak een lege dict `result` en zet daarin, met een `for`-loop over `estimates` die elke `name` neemt, onder die naam de daling maal zijn aantal toekomstige levens, gedeeld door de kosten, maal 100. Geef `result` terug. Vervang de `...`. De check roept `lives_per_100(PLANS["AI"], ESTIMATES)` aan, daarna de twee andere plannen, en eigen plannen en schattingen.
        """),
        L.task("Voor elke schatting: het risico maal het deel dat het plan weghaalt, maal het aantal toekomstige levens, per 100 dollar van de kosten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lives():
    box_lives = L.code_box("lives_per_100", lines=6)
    box_lives
    return (box_lives,)


@app.cell(hide_code=True)
def check_lives(box_lives):
    result_lives = L.check("lives_per_100", box_lives.value)
    result_lives
    return (result_lives,)


@app.cell(hide_code=True)
def hints_lives():
    L.hints(
        "lives_per_100",
        [
            "Eerst de daling die het hele plan koopt, dan voor elke schatting wat die daling waard is in levens, en als laatste per 100 dollar in plaats van voor de volledige kosten.",
            "`drop = plan[\"risk\"] * plan[\"cut\"]` geeft de daling. `result = {}` begint een lege dict, en `result[name] = ...` zet een waarde onder een naam. Het aantal toekomstige levens van een schatting is zijn waarde in de dict `estimates`.",
            "`        result[name] = drop * ___ / plan[\"cost\"] * 100`",
        ],
        why="`drop = plan[\"risk\"] * plan[\"cut\"]`, `result = {}`, dan `for name in estimates:` met `result[name] = drop * estimates[name] / plan[\"cost\"] * 100`, en `return result`. Voor het AI-plan: 0,001 × 0,01 × 10¹⁴ / 10⁹ × 100 = 100 levens bij de beperkte schatting.",
    )
    return


@app.cell(hide_code=True)
def text_after_lives():
    mo.md(r"""
    Voor het AI-plan is 100 dollar een biljoen, een miljoen en 100 levens waard bij de hoofdschatting, de lage en de beperkte: de getallen van Greaves en MacAskill zelf. Asteroïden geven ongeveer 280.000, 0,28 en 0,000028, en pandemieën ongeveer 180 miljoen, 180 en 0,018; hun getallen, 300.000, 0,3 en 0,00003 en 200 miljoen, 200 en 0,02, zijn dezelfde, afgerond op één cijfer. Tegen 0,025 voor de klamboes winnen alle drie de plannen bij de hoofdschatting en de lage. Bij de beperkte schatting wint alleen het AI-plan; over de asteroïden schrijven ze zelf dat het voorbeeld sterk longtermisme steunt bij hun hoofdschatting en hun lage schatting, maar niet bij de beperkte.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_break_even():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. break_even: vanaf hoeveel toekomstige levens een plan wint

        De schattingen van de toekomst verschillen tien machten van tien, dus het helpt om de vraag om te draaien. Schrijf `break_even(plan, benchmark)`: het aantal toekomstige levens waarbij 100 dollar van het plan precies `benchmark` levens waard is. Reken `drop` uit zoals in oefening 1. De levens per 100 dollar zijn daling × levens / kosten × 100; reken uit bij welk aantal levens dat gelijk is aan `benchmark`, en geef dat terug. Vervang de `...`. De check roept `break_even(PLANS["AI"], BEDNETS)` aan, het AI-plan tegen de klamboes, daarna de twee andere plannen en eigen gevallen.
        """),
        L.task("Draai de som van oefening 1 om: bij welk aantal toekomstige levens is 100 dollar van het plan precies de benchmark waard?"),
    ])
    return


@app.cell(hide_code=True)
def exercise_break_even():
    box_break_even = L.code_box("break_even", lines=3)
    box_break_even
    return (box_break_even,)


@app.cell(hide_code=True)
def check_break_even(box_break_even):
    result_break_even = L.check("break_even", box_break_even.value)
    result_break_even
    return (result_break_even,)


@app.cell(hide_code=True)
def hints_break_even():
    L.hints(
        "break_even",
        [
            "Schrijf benchmark = daling × levens / kosten × 100 op, en zet het aantal levens alleen aan één kant.",
            "Vermenigvuldig beide kanten met de kosten, en deel ze dan allebei door de daling en door 100. In Python staat een product waardoor je deelt tussen haakjes.",
            "`    return benchmark * plan[\"cost\"] / ___`",
        ],
        why="`drop = plan[\"risk\"] * plan[\"cut\"]` en `return benchmark * plan[\"cost\"] / (drop * 100)`. Voor het AI-plan: 0,025 × 10⁹ / (10⁻⁵ × 100) = 2,5 × 10¹⁰.",
    )
    return


@app.cell(hide_code=True)
def text_after_break_even():
    mo.md(r"""
    Het asteroïdenplan heeft 9 × 10¹⁶ toekomstige levens nodig, het pandemieplan 1,4 × 10¹⁴ en het AI-plan 2,5 × 10¹⁰: 25 miljard, wat een mensheid op aarde met de 10¹⁰ levens per eeuw uit hun tabel in tweeënhalve eeuw haalt. Het AI-argument heeft de enorme getallen dus nauwelijks nodig; het rust op de 0,1% en de 1%. Daarvoor zijn volgens hen geen harde kwantitatieve gegevens die de schattingen voor werk aan AI-veiligheid kunnen sturen. Deskundigen schatten de kans op een existentiële ramp door kunstmatige superintelligentie wel meestal op 1 tot 10%, voegen ze eraan toe. Zulke getallen schatten is de eerste van de zwakke plekken die ze zelf noemen.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: hoeveel toekomstige levens?

    De schuif zet het verwachte aantal toekomstige levens, als macht van tien, van 10¹⁰ tot 10²⁴. De balken tonen hoeveel levens 100 dollar van elk plan waard is, op een schaal waarop elk streepje tien keer zoveel is; de gestippelde lijn is de klamboes.
    """)
    return


@app.cell(hide_code=True)
def demo_future():
    future = L.explore("future", "toekomstige levens, als macht van tien", lambda _v: C.demo_frame(_v, L.lang),
                       start=14, values=C.POWERS, role="",
                       task="Zoek het kleinste getal op de schuif waarbij alle drie de plannen het beter doen dan de klamboes.")
    future
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Vanaf 10¹⁷ doen alle drie de plannen het beter dan de klamboes, het asteroïdenplan maar net: 0,0278 tegen 0,025. Bij 10¹⁰ doet geen enkel plan het beter. Daartussen hangt het antwoord af van een getal dat niemand kan meten, de verwachte omvang van de toekomst; de scenario's in hun eigen tabel beslaan 31 machten van tien. Daar richten de critici zich op, en daar leggen Greaves en MacAskill hun eigen twijfels.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_label():
    mo.md(r"""
    ### Oefening 3. Een etiket op een blog

    Een onderzoeker controleert sollicitatiemodellen op beslissingen die groepen benadelen, en geeft geld aan een goed doel dat klamboes koopt en door effectief altruïsme wordt aanbevolen. Een blogpost noemt de onderzoeker "een TESCREAList, dus een eugeneticus".
    """)
    return


@app.cell(hide_code=True)
def choice_label_cell():
    choice_label = L.choice(
        "label",
        question="Wat zegt het paper van Gebru en Torres zelf dat hierover gaat?",
        options={
            "self": "Dat TESCREAL de naam is die de leden van deze bewegingen zelf kozen",
            "right": "Dat niet iedereen die bij deze ideologieën hoort, al hun overheersende opvattingen deelt",
            "outside": "Dat effectief altruïsme niet bij de zeven ideologieën van de bundel hoort, dus het etiket past niet",
        },
        correct="right",
        why={
            "right": "Het paper zegt dat niet iedereen die bij deze ideologieën hoort, alle overheersende opvattingen in de bundel deelt. Het verband met eugenetica is de stelling van de auteurs over de ideologieën, geen vaststelling over iedereen die aan een goed doel geeft.",
            "self": "Gebru en Torres bedachten het acroniem zelf, toen ze een eerste versie van hun paper schreven, en zeggen dat sommigen bezwaar kunnen hebben tegen het bundelen. Zie *TESCREAL* onder *Het idee*.",
            "outside": "Effectief altruïsme is een van de zeven, de EA in TESCREAL. Voor de blog telt wat het paper over mensen zegt: niet iedereen die bij deze ideologieën hoort, deelt al hun overheersende opvattingen. Zie *TESCREAL* onder *Het idee*.",
        },
    )
    choice_label
    return (choice_label,)


@app.cell(hide_code=True)
def text_exercise_scales():
    mo.md(r"""
    ### Oefening 4. Dichtbij of ver weg?

    Een onderzoeksgroep bestudeert hoe de algoritmen die nu kiezen welke berichten mensen te zien krijgen, het vertrouwen in verkiezingen in de loop van tientallen jaren kunnen uithollen, tot een land het niet meer eens kan worden over hoe het een crisis aanpakt.
    """)
    return


@app.cell(hide_code=True)
def choice_scales_cell():
    choice_scales = L.choice(
        "scales",
        question="Is dit werk voor de korte of voor de lange termijn?",
        options={
            "near": "Korte termijn: het gaat over systemen die nu bestaan, dus het hoort bij AI-ethiek",
            "long": "Lange termijn: de schade komt over tientallen jaren en kan een heel land raken, dus het hoort alleen bij AI-veiligheid",
            "right": "Eén etiket past niet: het gaat over huidige systemen, maar de gevolgen komen laat en zijn groot",
        },
        correct="right",
        why={
            "right": "Op de schalen van Prunkl en Whittlestone zit het dichtbij wat capabilities betreft, ver weg wat de gevolgen betreft en hoog op ernst, met de zekerheid ergens ertussenin. Bucknall en Dori-Hacohen zouden zo'n schade een mogelijke risicofactor noemen: schade van nu die een latere ramp moeilijker te beantwoorden maakt.",
            "near": "Het gaat over de systemen van nu, maar de schade die het onderzoekt komt laat en kan groot zijn: één etiket verbergt dat. Zie *Meer gemeen dan het lijkt* onder *Het idee*.",
            "long": "De schade komt laat, maar de systemen zijn die van nu, en de schade bouwt zich op uit effecten die al zijn vastgesteld. Zie *Meer gemeen dan het lijkt* onder *Het idee*.",
        },
    )
    choice_scales
    return (choice_scales,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Soorten risico's van AI](/course/open/rai-kinds-of-ai-risk?lang=nl) sorteerde risico's naar waar ze vandaan komen, en [Ethische theorieën toegepast op AI](/course/open/rai-ethical-theories?lang=nl) liet theorieën zien die dezelfde keuze elk anders wegen. Hier komen de twee samen: de voorrang die een deel van de AI-veiligheid aan de verre toekomst geeft, wordt onderbouwd met een consequentialistische som met heel grote getallen, al menen Greaves en MacAskill dat hij ook zonder consequentialisme overeind kan blijven, en de critici hebben bezwaar tegen die som, tegen wat ze weglaat en tegen de ideeën erachter. De volgende les, *Wie werkt er aan AI-veiligheid?*, kijkt naar de instituten, labs en cursussen die samen het vakgebied vormen.

    **Een voorbeeld uit het dagelijks leven.** Een gezin dat het repareren van een lekkend dak nu afweegt tegen sparen voor de studie van een kind over vijftien jaar, staat in het klein voor dezelfde vraag: hoeveel moet wat later komt tellen tegenover wat nu pijn doet?
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_label, choice_scales, result_break_even, result_lives):
    L.wrap_up(result_lives, result_break_even, choice_label, choice_scales)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
