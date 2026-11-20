# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rai-complex-systems. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow AI Safety

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-complex-systems", "nl")
    C = L.checks   # een verzonnen markt met een verkoopalgoritme, en acht verzonnen modellen die optellen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Afglijden naar falen, en zwarte zwanen](/course/open/rai-drift-and-tails?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_drift():
    quick_drift = L.choice(
        "quick-drift",
        question="Een team houdt het foutpercentage van een model in de gaten, met een alarm bij 2%. Elke keer dat het percentage een beetje over de lijn gaat en er niets ergs gebeurt, zet het team de lijn een stukje hoger. Na een jaar staat die op 5%. Wat is dit?",
        options={
            "swan": "Een zwarte zwaan: niemand zag het aankomen",
            "right": "Drift into failure",
            "tail": "Een lange staart: een zeldzame waarde ver van de rest, die zwaarder weegt dan alle andere",
        },
        correct="right",
        why={
            "right": "Elke stap wordt geaccepteerd omdat de vorige geen kwaad deed, en zo kruipt de lijn omhoog: dat is drift into failure. Er is nog niets misgegaan, maar de ruimte die het alarm moest geven, is weg.",
            "swan": "Een zwarte zwaan is een zeldzame gebeurtenis met grote gevolgen die de meeste mensen niet zagen aankomen. Hier is nog niets gebeurd: de lijn gaat stap voor stap omhoog, en elke stap wordt geaccepteerd omdat de vorige geen kwaad deed. Dat is drift into failure. Zie [Afglijden naar falen, en zwarte zwanen](/course/open/rai-drift-and-tails?lang=nl).",
            "tail": "Een lange staart gaat over hoe groot zeldzame waarden kunnen worden. Hier gaat de lijn stap voor stap omhoog, en elke stap wordt geaccepteerd omdat de vorige geen kwaad deed: dat is drift into failure. Zie [Afglijden naar falen, en zwarte zwanen](/course/open/rai-drift-and-tails?lang=nl).",
        },
    )
    quick_drift
    return (quick_drift,)


@app.cell(hide_code=True)
def quick_check_tail():
    quick_tail = L.choice(
        "quick-tail",
        question="De schade van incidenten in een systeem heeft een lange staart. In tien jaar kostte het ergste incident 2 miljoen. Wat zegt dat over het ergste incident van de komende tien jaar?",
        options={
            "right": "Weinig: één nieuw incident kan veel groter zijn",
            "same": "Dat kost ook ongeveer 2 miljoen",
            "cap": "Hooguit 2 miljoen, want dat is het ergste in tien jaar aan gegevens",
        },
        correct="right",
        why={
            "right": "Bij een lange staart kan één nieuwe waarde groter zijn dan alle eerdere samen, dus het ergste tot nu toe is geen plafond. Tien jaar aan gegevens zegt weinig over het volgende incident.",
            "same": "Dat geldt voor een dunne staart, zoals lengte. Bij een lange staart kan één nieuwe waarde veel groter zijn dan alles ervoor, dus het ergste tot nu toe zegt weinig over het volgende. Zie [Afglijden naar falen, en zwarte zwanen](/course/open/rai-drift-and-tails?lang=nl).",
            "cap": "Het ergste tot nu toe is geen plafond: bij een lange staart kan één nieuw incident groter zijn dan alle eerdere samen. Zie [Afglijden naar falen, en zwarte zwanen](/course/open/rai-drift-and-tails?lang=nl).",
        },
    )
    quick_tail
    return (quick_tail,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De lessen tot nu toe keken naar één systeem tegelijk: een model, zijn safeguards, zijn tests. Maar een AI-systeem draait niet alleen. Het antwoordt mensen die veranderen wat ze vragen, het handelt op markten waar ook andere programma's handelen, en wat het maakt, komt terecht in de wereld waarvan het leert. Als onderdelen zo op elkaar reageren, kan het geheel dingen doen die geen enkel onderdeel zelf doet, en kan een oplossing op de ene plek het op een andere plek erger maken. Deze les laat zien hoe dat eruitziet, in een beurscrash en in de manier waarop vaardigheden van language models gemeten worden, en waarom veiligheid dan geen bug is die je één keer oplost.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Op 6 mei 2010 crashten de Amerikaanse aandelenkoersen, en binnen een paar minuten herstelden ze weer. In 15 seconden van de crash verhandelden flitshandelaren, bedrijven waarvan de computers heel snel handelen, meer dan 27.000 futurescontracten. Hoeveel kochten ze er meer dan ze verkochten?",
        options={
            "right": "Ongeveer 200",
            "half": "Ongeveer 13.500: de helft van wat ze verhandelden",
            "all": "Ongeveer 27.000: ze kochten alleen maar",
        },
        reveal="**Ongeveer 200.** Volgens het gezamenlijke rapport van medewerkers van de CFTC en de SEC, twee Amerikaanse marktwaakhonden, verhandelden de flitshandelaren tussen 14:45:13 en 14:45:27 uur meer dan 27.000 contracten, ongeveer 49 procent van alle handel, terwijl ze per saldo maar ongeveer 200 contracten bijkochten. Ze kochten en verkochten dezelfde contracten heen en weer; het rapport noemt dat een hete-aardappeleffect. Er werd veel verhandeld, maar bijna niets van wat er gekocht werd, bleef bij de koper.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Veel onderdelen die op elkaar reageren

    Het leerboek *Introduction to AI Safety, Ethics, and Society* van Dan Hendrycks (2024) begint zijn hoofdstuk over complexe systemen met de stelling dat AI-systemen en de samenlevingen waarin ze werken, tot de complexe systemen horen. Het omschrijft een complex systeem als een systeem van veel onderdelen die met elkaar verbonden zijn, en die samen emergente eigenschappen laten zien die je in de praktijk niet kunt afleiden uit de losse onderdelen.

    Het sleutelwoord is **emergentie**: opvallende eigenschappen van het hele systeem die je in geen van de onderdelen terugvindt. Mieren laten het zien. Honderd legermieren op de grond gedragen zich volgens het boek niet als een verzwakte kolonie, maar als helemaal geen kolonie: ze lopen in rondjes tot ze sterven. Met tienduizenden gaat de kolonie zich gedragen als één intelligent geheel. Geen enkele mier heeft dat gedrag; het ontstaat uit hoe ze op elkaar reageren.

    Naast emergentie noemt het boek zelforganisatie, feedback en niet-lineariteit, kritikaliteit, gedrag dat zich aanpast, functies die over het systeem verspreid zijn, en een structuur die meegroeit. Feedback gaat over kringprocessen waarin een systeem en zijn omgeving elkaar beïnvloeden: wat eruit komt, gaat er weer in, een feedback loop. En het boek merkt zelf op dat complexiteit ook in ons begrip van een systeem kan zitten, in plaats van in het systeem zelf. Een gas was lastig te verklaren uit zijn moleculen, tot tientallen jaren natuurkunde het uitzochten.

    ### De flash crash van 6 mei 2010

    Die middag zakten grote Amerikaanse aandelenindexen, die al meer dan 4% lager stonden dan bij het slot van de dag ervoor, binnen een paar minuten nog eens 5 tot 6%, om bijna even snel weer te herstellen. Meer dan 20.000 transacties in meer dan 300 effecten gingen door tegen prijzen die meer dan 60% afweken van de waarde van vlak ervoor. Dat schrijven medewerkers van de CFTC en de SEC, twee Amerikaanse marktwaakhonden, in hun gezamenlijke rapport van 30 september 2010.

    Om 14:32 uur begon een grote handelaar, een groep beleggingsfondsen, 75.000 E-Mini-contracten te verkopen, futures die de beursindex S&P 500 volgen, samen ongeveer 4,1 miljard dollar waard. Dat gebeurde met een verkoopalgoritme dat was ingesteld op 9% van het handelsvolume van de minuut ervoor, zonder te letten op prijs of tijd. Een eerdere verkoop van die omvang had meer dan 5 uur geduurd; deze duurde 20 minuten.

    Flitshandelaren kochten eerst, en gaven daarna contracten aan elkaar door: de hete aardappel uit de gok. Al die handel telde als volume, en volgens het rapport reageerde het verkoopalgoritme op het hogere volume door zijn orders sneller de markt in te sturen. Meer verkopen, meer handel, meer volume, meer verkopen: een feedback loop. Om 14:45:28 uur legde de beurs de handel in de E-Mini vijf seconden stil, om een verdere reeks koersdalingen te voorkomen. Toen de handel weer begon, stabiliseerden de prijzen, en kort daarna begon de E-Mini te herstellen.

    Geen enkel onderdeel was gebouwd om een markt te laten crashen. Een van de lessen in het rapport is dat de geautomatiseerde uitvoering van een grote verkooporder in een markt onder druk extreme koersbewegingen kan veroorzaken, vooral als het algoritme niet naar de prijs kijkt. Daarnaast kan volgens het rapport de wisselwerking tussen geautomatiseerde uitvoeringsprogramma's en algoritmische handelsstrategieën snel de liquiditeit uithollen en tot ordeloze markten leiden. Na de crash kwam er een proefprogramma, eerst voor de aandelen in de S&P 500 en tot 10 december 2010: de handel in één aandeel werd vijf minuten stilgelegd als de koers in de vijf minuten ervoor 10% veranderd was. Dat is een rem die werkt, wat de oorzaak ook is. Het rapport spreekt van algoritmes, niet van AI. Het leerboek gebruikt de flash crash als voorbeeld van hoe snel een fout in een geautomatiseerd systeem kan escaleren, en spreekt van een feedback loop tussen geautomatiseerde handelsalgoritmes.

    ### Vaardigheden die lijken te verschijnen

    Bij language models heeft het woord een tweede betekenis, en of zulke vaardigheden echt verschijnen, is omstreden. Volgens Jason Wei en anderen (2022) is een vaardigheid emergent als die bij kleinere modellen ontbreekt en bij grotere aanwezig is. Op een grafiek van prestatie tegen grootte blijft de prestatie dan rond het niveau van willekeurig gokken, tot een bepaalde kritische schaal bereikt is. Ze merken ook op dat de schaal waarop een vaardigheid voor het eerst opduikt, geen vaste eigenschap van die vaardigheid is. Het leerboek stelt als feit dat grotere LLM's vanzelf kwalitatief nieuwe vaardigheden krijgen, zoals tekst vertalen of rekenen met getallen van drie cijfers.

    Rylan Schaeffer en anderen (2023) betwijfelen dat. Volgens hen zouden emergent abilities weleens door de keuzes van de onderzoeker kunnen ontstaan, in plaats van een wezenlijke eigenschap van de modelfamilie te zijn. Hun voorbeeld is een metric die een optelling 1 geeft als alle L cijfers van het antwoord precies kloppen met die van het juiste antwoord, en anders 0: een exact match. Als elk cijfer los van de andere goed of fout is, daalt de kans op een helemaal goed antwoord snel met de lengte ervan. Volgens hun telling duikt meer dan 92% van de emergent abilities op BIG-Bench-taken op bij een van twee metrics: exact match en een score voor meerkeuzevragen, die allebei een antwoord 1 of 0 geven. Ze begrenzen hun eigen claim: volgens hen betekent niets in hun paper dat large language models geen emergent abilities kunnen vertonen.

    ### Oplossingen die terugslaan

    In een paper uit 1973 schreven Horst Rittel en Melvin Webber, een hoogleraar ontwerpen en een hoogleraar stadsplanning in Berkeley, over de problemen van planning en sociaal beleid. Die noemden ze wicked problems, tegenover de tamme problemen waar de wetenschap op ingesteld is. Wicked betekent hier niet slecht in morele zin: ze gebruiken het woord in de zin van kwaadaardig (tegenover goedaardig), vicieus (zoals een cirkel) of geniepig (zoals een kobold). Ze noemen tien eigenschappen, waaronder dat er geen definitieve formulering van een wicked problem is, dat een wicked problem geen stopregel heeft, en dat een oplossing ervan niet waar of onwaar is, maar goed of slecht. Zulke problemen los je niet één keer op; hooguit los je ze telkens opnieuw op.

    Rittel en Webber schreven over planning, niet over AI. Het leerboek gebruikt de term wel voor AI. Volgens het boek ontstaan wicked problems meestal in complexe systemen en hebben ze vaak een sociale kant, en omdat de systemen waar het om gaat complex zijn, is AI-veiligheid een wicked problem. Om dezelfde reden is AI veilig maken volgens het boek niet zoiets als een wiskundeprobleem oplossen of een horloge repareren. Een oplossing kan ook nieuwe problemen scheppen: sommige oplossingen hebben volgens het boek nadelige gevolgen, ook als ze het oorspronkelijke wicked problem kleiner maken. En sommige pogingen werken niet, hebben nadelige gevolgen, of werken zelfs averechts.

    Op één punt verschillen ze. Volgens Rittel en Webber is elke oplossing van een wicked problem eenmalig: je kunt niet leren door te proberen, dus elke poging telt zwaar. Het leerboek waarschuwt dat het vaak riskant is om een wicked problem aan te pakken, maar schrijft ook dat leren hoe je AI veilig maakt, wat vallen en opstaan zal vragen.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **emergentie** | gedrag van een geheel dat geen van de onderdelen op zichzelf heeft, zoals dat van een mierenkolonie |
        | **wicked problem** | een probleem zonder definitieve formulering of definitieve oplossing, waarbij elke poging het probleem verandert; naar Rittel en Webber (1973) |
        | complex systeem | een systeem van veel onderdelen die op elkaar reageren, waarvan je het gedrag niet uit de losse onderdelen kunt afleiden |
        | feedback loop | wat eruit komt, gaat er weer in, zoals het volume terugkwam bij het verkoopalgoritme |
        | flitshandelaar | een handelsbedrijf waarvan de computers heel veel contracten verhandelen, maar er weinig lang houden |
        | emergent ability | een vaardigheid die grotere language models lijken te hebben en kleinere niet; of de sprong echt is, is omstreden |
        | exact match | een metric die een antwoord alleen goed rekent als het helemaal goed is |
        """),
        L.watch_out("Emergentie heeft hier twee betekenissen. In een complex systeem is het gedrag van het geheel dat geen onderdeel heeft. Bij language models is het een vaardigheid die grotere modellen hebben en kleinere niet. Of zulke vaardigheden bij language models echt verschijnen, is omstreden."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een AI-systeem te bekijken als deel van een complex systeem:

    1. Schrijf op wie en wat op het systeem reageren: de gebruikers, andere programma's, de mensen die het raakt, en de data waarvan het leert.
    2. Zoek de loops: wat eruit komt en er weer in gaat, zoals het volume terugkwam bij het verkoopalgoritme.
    3. Test onderdelen samen, niet alleen een voor een: gedrag kan ontstaan uit hoe ze op elkaar reageren.
    4. Bouw remmen in die werken wat de oorzaak ook is, zoals een pauze als alles te snel gaat.
    5. Reken erop dat een oplossing het systeem verandert, en kijk wat er na de oplossing gebeurt, niet alleen ervoor.
    6. Reken erop dat je er opnieuw aan werkt: een wicked problem los je niet één keer op.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Het verkoopalgoritme van de flash crash verkocht 9% van het volume van de minuut ervoor. De cel rekent uit wat zo'n algoritme verkoopt in een markt waarvan het volume over vier minuten stijgt. De volumes zijn verzonnen.
    """)
    return


@app.cell
def example_rule():
    rate = 0.09                         # het algoritme verkoopt dit deel van het volume van de minuut ervoor
    traded = [4000, 6000, 9000, 14000]  # verhandelde contracten in vier minuten, verzonnen
    for minute in range(1, len(traded)):
        sold = rate * traded[minute - 1]
        print("in minuut", minute + 1, "verkoopt het algoritme", round(sold), "contracten")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Hoe meer er verhandeld wordt, hoe meer het algoritme verkoopt: 360, 540 en dan 810 contracten. Het kijkt niet naar de prijs, alleen naar het volume. Maar wie maakt het volume? Voor een deel het algoritme zelf, zodra wat het verkoopt wordt doorverhandeld. In oefening 1 maak je die loop rond.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_volumes():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. volumes: een loop die zichzelf voedt

        Een verzonnen, vereenvoudigde markt met de regel uit het uitgewerkte voorbeeld. Elke minuut verhandelen de andere handelaren `base` contracten. Het verkoopalgoritme verkoopt `rate` keer het volume van de minuut ervoor. De snelle handelaren geven elk contract dat het verkoopt aan elkaar door, zodat elk ervan in diezelfde minuut `echo` keer verhandeld wordt. Schrijf `volumes(base, rate, echo, minutes)`, die een lijst `result` teruggeeft met het volume van elke minuut. Begin `volume` bij `base`. Voeg in een `for`-loop met `minute` over `range(minutes)` het `volume` toe aan `result`, reken `sold` uit, wat het algoritme in de volgende minuut verkoopt, en daarna het `volume` van die volgende minuut. Vervang de `...`. De check roept `volumes(1000, 0.09, 12, 8)` aan, en daarna andere gevallen.
        """),
        L.task("Het volume van elke minuut in de verzonnen markt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_volumes():
    box_volumes = L.code_box("volumes", lines=8)
    box_volumes
    return (box_volumes,)


@app.cell(hide_code=True)
def check_volumes(box_volumes):
    result_volumes = L.check("volumes", box_volumes.value)
    result_volumes
    return (result_volumes,)


@app.cell(hide_code=True)
def hints_volumes():
    L.hints(
        "volumes",
        [
            "Zet het volume in de lijst voordat je het verandert: de eerste minuut heeft alleen de `base` van de andere handelaren.",
            "Begin `volume` bij `base` en `result` als een lege lijst. Voeg in een `for`-loop over `range(minutes)` het `volume` toe, maak `sold` gelijk aan `rate` keer `volume`, en maak `volume` daarna gelijk aan het volume van de volgende minuut: de `base` van de andere handelaren, plus elk verkocht contract `echo` keer verhandeld. Geef na de loop `result` terug.",
            "`        volume = base + ___`",
        ],
        why="`volume = base` en `result = []`, dan `for minute in range(minutes):` met `result.append(volume)`, `sold = rate * volume` en `volume = base + echo * sold`; na de loop `return result`. In het eerste geval heeft de eerste minuut 1000 contracten; het algoritme verkoopt er 90 van, die elk 12 keer verhandeld worden, dus de tweede minuut heeft 1000 + 1080 = 2080, enzovoort.",
    )
    return


@app.cell(hide_code=True)
def text_after_volumes():
    mo.md(r"""
    Met een echo van 12 voegt elke minuut meer toe dan de minuut ervoor: van 1000 naar 10.637 contracten in acht minuten. Met een echo van 5 komt dezelfde markt uit bij ongeveer 1818. De regel van het algoritme is in beide gelijk; alleen hoe sterk de anderen erop reageren, verschilt. Bij de flash crash was een deel van het volume dat het algoritme volgde, de hete aardappel van de flitshandelaren.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_accuracies():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. accuracies: één cijfer of allemaal

        Het voorbeeld van Schaeffer en anderen: een model telt getallen op, en een antwoord van `length` cijfers wordt alleen goed gerekend als elk cijfer goed is, een exact match. De kansen dat acht verzonnen modellen, elk groter dan het vorige, één cijfer goed hebben, staan in een NumPy-array `per_digit`. Neem aan, zoals in hun paper, dat elk cijfer los van de andere goed of fout is. Schrijf `accuracies(per_digit, lengths)`, die een lijst `rows` teruggeeft: voor elke `length` in de lijst `lengths` een array met de kans van elk model op een exact match bij een antwoord van die lengte. Gebruik een `for`-loop met `length`, en laat NumPy met alle modellen tegelijk werken. Vervang de `...`. De check roept `accuracies(np.array([0.5, 0.6, 0.7, 0.8, 0.88, 0.94, 0.97, 0.99]), [1, 10])` aan, en daarna andere gevallen.
        """),
        L.task("De kans van elk model op een exact match, voor elke lengte."),
    ])
    return


@app.cell(hide_code=True)
def exercise_accuracies():
    box_accuracies = L.code_box("accuracies", lines=5)
    box_accuracies
    return (box_accuracies,)


@app.cell(hide_code=True)
def check_accuracies(box_accuracies):
    result_accuracies = L.check("accuracies", box_accuracies.value)
    result_accuracies
    return (result_accuracies,)


@app.cell(hide_code=True)
def hints_accuracies():
    L.hints(
        "accuracies",
        [
            "Begin klein: als één cijfer goed is met een kans van 0,9, en elk cijfer los van de andere goed of fout is, wat is dan de kans dat twee cijfers allebei goed zijn?",
            "Begin `rows` als een lege lijst. Voeg in een `for`-loop over `lengths` de kans toe dat alle `length` cijfers goed zijn: `per_digit`, `length` keer met zichzelf vermenigvuldigd. NumPy doet dat voor alle modellen tegelijk, met één macht van de array. Geef na de loop `rows` terug.",
            "`        rows.append(___)`",
        ],
        why="`rows = []`, dan `for length in lengths:` met `rows.append(per_digit ** length)`; na de loop `return rows`. Twee cijfers goed met elk 0,9: 0,9 × 0,9 = 0,81; tien cijfers: 0,9 tot de macht 10, ongeveer 0,35. In het eerste geval is de rij voor 10 cijfers 0,001; 0,006; 0,028; 0,107; 0,279; 0,539; 0,737 en 0,904.",
    )
    return


@app.cell(hide_code=True)
def text_after_accuracies():
    mo.md(r"""
    Eén cijfer stijgt geleidelijk van model tot model, van 0,5 naar 0,99. Alle 10 cijfers goed blijft bij de eerste drie modellen dicht bij 0, op 0,001; 0,006 en 0,028, en stijgt dan tot 0,904. Op een grafiek van exact match lijkt dat een vaardigheid die bij een bepaalde grootte verschijnt; per cijfer worden de modellen stap voor stap beter. Dat is het punt van Schaeffer en anderen: dezelfde modellen kunnen eruitzien als een sprong of als gestage vooruitgang, afhankelijk van de metric. Het laat niet zien dat een vaardigheid nooit plotseling kan verschijnen, zoals ze zelf ook zeggen.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: hoe sterk de markt reageert

    De verzonnen markt uit oefening 1, over twintig minuten. Zet de echo hoger, hoe vaak elk contract dat het algoritme verkoopt wordt doorverhandeld, en kijk wanneer het volume niet meer tot rust komt.
    """)
    return


@app.cell(hide_code=True)
def demo_echo():
    echo = L.explore("echo", "echo: hoe vaak elk verkocht contract wordt doorverhandeld",
                     lambda _v: C.demo_frame(_v, L.lang), start=5, values=C.ECHOES, role="",
                     task="Zoek de kleinste echo waarbij elke minuut meer toevoegt dan de minuut ervoor.")
    echo
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Vanaf een echo van 12 ligt 0,09 × 12 = 1,08 boven 1: elke minuut voegt meer toe dan de vorige, en het volume schiet weg. Bij 11 ligt 0,99 net onder 1: op den duur zou het volume tot rust komen, al pas bij ongeveer 100.000, en in deze twintig minuten stijgt het nog. De regel van het algoritme en de 1000 contracten van de andere handelaren bleven hetzelfde; alleen de echo, hoe vaak de snelle handelaren doorgeven wat het verkoopt, ging één stap omhoog. Een kleine verandering in één onderdeel gaf een heel ander geheel.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_agents():
    mo.md(r"""
    ### Oefening 3. Duizend agents

    Een bedrijf test zijn handelsagent, een AI-systeem dat zelf koopt en verkoopt, alleen in een gesimuleerde markt, en de agent gedraagt zich goed. Daarna gaan duizend kopieën ervan live, op een markt waar ook agents van andere bedrijven handelen.
    """)
    return


@app.cell(hide_code=True)
def choice_agents_cell():
    choice_agents = L.choice(
        "agents",
        question="Wat zeggen de tests van één agent alleen over die markt?",
        options={
            "enough": "Genoeg, als elke agent zich goed gedraagt",
            "right": "Niet veel: samen kunnen de agents doen wat geen agent alleen doet",
            "more": "Alles, zolang de gesimuleerde markt groot genoeg was en lang genoeg draaide",
        },
        correct="right",
        why={
            "right": "Tests van één agent laten zien wat die alleen doet. Een markt vol agents is een complex systeem: net als bij de flash crash, waar elk onderdeel zijn eigen regel volgde, kan het probleem komen uit hoe de regels elkaar versterken. De mieren laten het andersom zien: honderd ervan gedragen zich als helemaal geen kolonie, tienduizenden als één geheel. Test de agents samen, en bouw remmen in die werken wat de oorzaak ook is.",
            "enough": "Dat elk onderdeel zich goed gedraagt, betekent niet dat het geheel zich goed gedraagt. Bij de flash crash deed elk algoritme wat het moest doen; de lessen in het rapport wijzen zowel naar een grote order die verkocht werd zonder op de prijs te letten, als naar hoe de geautomatiseerde programma's op elkaar inwerkten. Zie *De flash crash van 6 mei 2010* in *Het idee*.",
            "more": "Een grotere simulatie met één agent heeft nog steeds één agent. Wat duizend agents samen doen, en met de agents van andere bedrijven, kan pas ontstaan als ze elkaar tegenkomen. Zie *Veel onderdelen die op elkaar reageren* in *Het idee*.",
        },
    )
    choice_agents
    return (choice_agents,)


@app.cell(hide_code=True)
def text_exercise_backfire():
    mo.md(r"""
    ### Oefening 4. Op minder dagen rijden

    In 1989 pakte Mexico-Stad het verkeer en de luchtvervuiling aan met een programma dat Hoy No Circula heette: afhankelijk van het laatste cijfer op hun kenteken mochten mensen alleen op bepaalde dagen van de week rijden. Eerst daalde de uitstoot.
    """)
    return


@app.cell(hide_code=True)
def choice_backfire_cell():
    choice_backfire = L.choice(
        "backfire",
        question="Wat gebeurde er daarna, volgens het leerboek?",
        options={
            "stayed": "De uitstoot bleef laag, omdat er elke dag minder auto's reden",
            "right": "De uitstoot steeg weer, zelfs boven het niveau van voor de regel",
            "buses": "De uitstoot bleef jarenlang dalen, omdat mensen voorgoed overstapten op bus en trein",
        },
        correct="right",
        why={
            "right": "Volgens het leerboek steeg de uitstoot al snel weer, tot boven het niveau van voor de maatregel. Volgens een onderzoek dat het aanhaalt, kochten mensen extra auto's om op meer dagen te kunnen rijden, en die extra auto's waren vaak goedkoper, ouder en vervuilender. De stad veranderde één regel, de mensen erin reageerden, en het probleem kwam groter terug: een oplossing die averechts werkte omdat het systeem erop reageerde.",
            "stayed": "Eerst reden er elke dag minder auto's, maar mensen reageerden op de regel: volgens een onderzoek kochten ze extra auto's om op meer dagen te kunnen rijden. De uitstoot steeg weer, tot boven het niveau van voor de regel. Zie *Oplossingen die terugslaan* in *Het idee*.",
            "buses": "Het leerboek meldt het omgekeerde: mensen kochten extra auto's, vaak oudere die meer vervuilden, en de uitstoot steeg tot boven het niveau van voor de regel. Zie *Oplossingen die terugslaan* in *Het idee*.",
        },
    )
    choice_backfire
    return (choice_backfire,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Afglijden naar falen, en zwarte zwanen](/course/open/rai-drift-and-tails?lang=nl) liet zien hoe één systeem stap voor stap naar een ongeluk afglijdt. Deze les keek naar veel onderdelen die op elkaar reageren. Gedrag dat geen onderdeel heeft, zoals een flash crash, kan ontstaan uit hoe ze op elkaar inwerken; een sprong op een grafiek kan uit de metric komen in plaats van uit het model; en een oplossing kan het systeem veranderen waarvoor die bedoeld was. Het leerboek noemt AI-veiligheid een wicked problem omdat AI en de samenleving eromheen complexe systemen zijn: niet één keer opgelost, maar telkens opnieuw aangepakt. De volgende les, *Races en selectie*, kijkt naar één zo'n wisselwerking: bedrijven en staten die met elkaar concurreren.

    **Een voorbeeld uit het dagelijks leven.** Een file kan achteruit over de weg schuiven terwijl elke auto erin vooruit rijdt. Geen bestuurder maakt de file; die ontstaat uit hoe elke bestuurder op de auto ervoor reageert.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_agents, choice_backfire, result_accuracies, result_volumes):
    L.wrap_up(result_volumes, result_accuracies, choice_agents, choice_backfire)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
