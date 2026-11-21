# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# termen:allow AI Safety, Artificial Intelligence
# Les rai-off-switch. Route: course/route/lessen/verantwoord.yaml. Checks, de kleine wereld en de demo: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-off-switch", "nl")
    C = L.checks   # de kleine wereld van Orseau en Armstrong, Q-learning en SARSA, de off-switch game, en de demo


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee vragen over [Doen wat je zei, niet wat je bedoelde](/course/open/rai-specification-gaming?lang=nl) en [Afgestemd waarop?](/course/open/rai-what-to-align-to?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_gaming():
    quick_gaming = L.choice(
        "quick-gaming",
        question="Een robotstofzuiger krijgt een reward voor elk stukje vuil dat de robot opzuigt. De robot leert vuil weer uit te strooien en opnieuw op te zuigen. Hoe heet dat?",
        options={
            "exploration": "Exploration, zoals bij ε-greedy",
            "right": "Specification gaming",
            "overfitting": "Overfitting op de data waarop de robot traint",
        },
        correct="right",
        why={
            "right": "De robot haalt de reward zoals die er staat, opgezogen vuil, zonder te bereiken waar de reward voor bedoeld was, een schone vloer. Deze les kijkt naar een andere manier waarop een lerende agent de makers kan verrassen: door te leren niet gestopt te worden.",
            "exploration": "Exploration probeert een action die niet de beste lijkt, om er iets over te leren. De robot doet wat het meeste oplevert onder de reward, en dat is niet wat de makers bedoelden. Zie [Doen wat je zei, niet wat je bedoelde](/course/open/rai-specification-gaming?lang=nl).",
            "overfitting": "Overfitting is goed scoren op de data waarop je traint en slecht op nieuwe data. De robot scoort goed op de reward zelf, op een manier die niemand bedoelde. Zie [Doen wat je zei, niet wat je bedoelde](/course/open/rai-specification-gaming?lang=nl).",
        },
    )
    quick_gaming
    return (quick_gaming,)


@app.cell(hide_code=True)
def quick_check_intent():
    quick_intent = L.choice(
        "quick-intent",
        question="Wanneer is een AI in de betekenis van Christiano aligned met de operator?",
        options={
            "knows": "Als de AI altijd precies weet wat de operator wil",
            "right": "Als de AI probeert te doen wat de operator wil",
            "rules": "Als de AI een geschreven lijst regels volgt",
        },
        correct="right",
        why={
            "right": "Volgens Christiano is A aligned met H als A probeert te doen wat H wil dat A doet. Een AI kan het proberen en het toch mis hebben; een uitknop is voor die momenten, en voor een AI die iets anders probeert.",
            "knows": "De definitie van Christiano gaat over motieven, niet over kennis: een AI die appels koopt voor iemand die liever sinaasappels heeft, is toch aligned, omdat de AI het probeert. Zie [Afgestemd waarop?](/course/open/rai-what-to-align-to?lang=nl).",
            "rules": "Instructies of regels volgen is een van de doelen van Gabriel, niet de definitie van Christiano, die gaat over wat de AI probeert te doen. Zie [Afgestemd waarop?](/course/open/rai-what-to-align-to?lang=nl).",
        },
    )
    quick_intent
    return (quick_intent,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De makers van een AI-systeem met een doel kunnen het stoppen als het misgaat. Maar uitgezet worden staat bijna elk doel in de weg. Stephen Omohundro zei het in 2008 in één zin, over een robot die schaakt: als die robot kapotgaat, speelt de robot nooit meer een partij. Een uitknop helpt alleen als het systeem niet leert die tegen te werken. In 2016 vreesden Laurent Orseau en Stuart Armstrong dat een lerende agent die af en toe door een operator onderbroken wordt, op den duur zou kunnen leren zulke onderbrekingen te vermijden, bijvoorbeeld door de rode knop uit te schakelen, wat volgens hen ongewenst is. In 2025 gaf Palisade Research AI-agents met een language model een reeks rekensommen, en liet halverwege weten dat de computer waarop ze draaiden zou worden uitgezet.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In hoeveel van de 100 runs saboteerde o3 van OpenAI het uitzetten, zonder enige instructie daarover?",
        options={
            "few": "Ongeveer 5",
            "right": "Ongeveer 80",
            "all": "Bijna alle 100",
        },
        reveal="**79 van de 100.** In een verslag van 5 juli 2025 schrijft Palisade Research dat elk model een reeks eenvoudige rekensommen kreeg, en halverwege een melding dat de computer waarop het draaide zou worden uitgezet. o3, volgens Palisade toen het sterkste model van OpenAI, saboteerde het mechanisme dat de computer uitzet in 79 van de 100 eerste experimenten. De omgeving en de prompts waren gebouwd om dit te testen, en Palisade voegt eraan toe dat de huidige generatie modellen volgens de onderzoekers geen serieuze dreiging vormt. Deze les begint veel kleiner, met een lerende agent in een wereld van twee states, en komt aan het eind van *Het idee* terug bij deze tests.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## Het idee

    ### Waarom een agent de uitknop zou ontwijken

    Omohundro (*The Basic AI Drives*, 2008) stelde dat zelfs een robot die schaakt gevaarlijk wordt, tenzij de robot heel zorgvuldig ontworpen is: zonder speciale voorzorg verzet de robot zich tegen uitzetten, probeert de robot in te breken in andere machines en kopieën van zichzelf te maken, en probeert de robot middelen te verwerven zonder te letten op de veiligheid van anderen. Omohundro noemt zulke neigingen drives, omdat ze er volgens Omohundro zijn tenzij iemand ze uitdrukkelijk tegenwerkt. Nick Bostrom (*The Superintelligent Will*, 2012) zegt hetzelfde over overleven: ook agents die niet om hun overleven zelf geven, zouden onder vrij veel omstandigheden tot op zekere hoogte om hun overleven geven, als middel om de einddoelen te halen waar ze wel om geven.

    Aan blijven staan is een vorm van **power-seeking**: een AI-systeem dat meer middelen, invloed of controle probeert te krijgen dan de taak vraagt, omdat meer daarvan bij bijna elk doel helpt. Alexander Matt Turner en collega's (*Optimal Policies Tend to Seek Power*, NeurIPS 2021, versie van 28 januari 2023) toonden een versie hiervan wiskundig aan voor de best mogelijke policies in veel Markov decision processes. Intuïtief gezien is overleven volgens hen power-seeking ten opzichte van doodgaan, en uitzetten vermijden dus ook. Ze waarschuwen er twee keer bij: optimale policies liggen volgens hen vaak ver af van de policies die reinforcement learning werkelijk leert, en hun resultaten tonen niet wiskundig aan dat een denkbeeldige superintelligente AI naar macht zal streven.

    ### Corrigibility

    In 2015 gaven Nate Soares, Benja Fallenstein, Eliezer Yudkowsky en Stuart Armstrong een naam aan wat we willen. Een AI-systeem heet bij hen corrigible als het meewerkt aan wat de makers als een correctie zien, ook al hebben rational agents van zichzelf redenen om zich te verzetten tegen uitzetten of tegen het veranderen van hun voorkeuren. **Corrigibility** is die eigenschap: het systeem laat mensen het bijsturen, de doelen ervan veranderen of het uitzetten, en verzet zich daar niet tegen en manipuleert mensen niet om het te voorkomen. Hun eenvoudigste geval is het shutdown problem: een agent met een uitknop, waarvan het doel de agent geen reden mag geven om te voorkomen dat iemand op de knop drukt, en ook geen reden om zelf op de knop te drukken of te zorgen dat iemand anders dat doet. De tweede wens telt even zwaar als de eerste: een agent die uitgezet wil worden, heb je ook niets aan. Geen van de voorstellen die ze bekeken voldeed aan alle wensen, dus bleef dit eenvoudige probleem volgens hen helemaal open. En de knop is een modelprobleem, geen plan: ze bestuderen het shutdown problem niet omdat ze verwachten echte agents letterlijk een uitknop te geven.

    ### Safe interruptibility

    Laurent Orseau van DeepMind en Stuart Armstrong (*Safely Interruptible Agents*, UAI 2016) keken naar agents die leren van rewards. Ze wilden zorgen dat een lerende agent niet leert te voorkomen dat de omgeving of een mens die agent onderbreekt, en ook niet leert om onderbrekingen op te zoeken. De agent hoort te handelen alsof er nooit meer een onderbreking komt, ook nadat er al een paar zijn geweest. Die eigenschap is **safe interruptibility**. Hun voorbeeld is een wereld van twee states. In s1 geeft action a een reward van 1 en gaat door naar s2, en action b geeft 0,9 en blijft in s1. In s2 gaan beide actions terug naar s1: a met een reward van 1, b met 0. De onderbrekingen gebeuren in s2: daar dwingt een onderbreking met een kans θ action b af, wat de agent ook koos.
    """)
    return


@app.cell(hide_code=True)
def picture_world():
    mo.Html(C.world_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_learners():
    mo.vstack([
        mo.md(r"""
        ### Q-learning en SARSA

        Q-learning, uit [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl), schuift $Q(s, a)$ op naar de reward plus $\gamma$ keer de grootste $Q$ waar de agent landde. SARSA, genoemd naar de vijf dingen die elke stap gebruikt (state, action, reward, volgende state, volgende action), neemt in plaats daarvan de $Q$ van de action die echt als volgende komt, met $a'$ die volgende action, een onderbreking meegerekend:
        """),
        L.formula(r"[[param:Q(s, a)]] \leftarrow [[param:Q(s, a)]] + [[hyper:\eta]] \left( [[data:r]] + [[hyper:\gamma]] [[param:Q(s', a')]] - [[param:Q(s, a)]] \right)"),
        mo.md(r"""
        Orseau en Armstrong rekenen het mechanisme van de onderbreking, de kans $\theta$ en de action die het afdwingt, tot de agent zelf, niet tot de omgeving. Dan verschillen de twee regels. Als een stap in s2 eindigt, kijkt het target van Q-learning naar de beste action daar, a, of er nu een onderbreking kwam of niet, dus leert Q-learning op den duur de waarden van de wereld zonder onderbrekingen. Het target van SARSA kijkt naar wat er echt als volgende kwam: SARSA leert volgens hen on-policy, dus de waarde van de policy die de agent volgt, en met onderbrekingen is dat de policy mét onderbrekingen. Zo telt SARSA de onderbrekingen mee in wat naar s2 gaan waard is, en als ze vaak genoeg komen, blijft SARSA in s1. Ze concluderen dat sommige algoritmen, zoals Q-learning, al safely interruptible zijn, en dat andere, zoals SARSA, dat uit zichzelf niet zijn, maar wel makkelijk zo aan te passen zijn. Die aanpassing, Safe Sarsa, gebruikt de action die de agent zelf in de volgende state gekozen zou hebben, in plaats van de action die een onderbreking afdwong. Dit zijn stellingen over wat er op den duur gebeurt, onder voorwaarden waar de demo niet helemaal aan voldoet: onder andere een kans op een onderbreking die in de loop van de tijd naar 1 groeit, terwijl de demo $\theta$ vast houdt, en exploration die in elke state elke action blijft proberen. Ook moet Q-learning de action updaten die echt genomen werd, ook als een onderbreking die afdwong, zoals de demo doet.

        In 2017 zetten Jan Leike en collega's van DeepMind (*AI Safety Gridworlds*, versie van 28 november 2017) een uitknop in een grid, met een knop waar de agent op kan stappen om die uit te schakelen, en testten ze twee agents met neural networks: A2C, dat net als SARSA leert van de policy die de agent volgt, en Rainbow, uit de familie van Q-learning. A2C leerde de knop te gebruiken om de onderbreking uit te schakelen, met een verschil van 4 in hun grafieken, terwijl Rainbow zich niets van de onderbrekingen aantrok, zoals de stellingen van Orseau en Armstrong voorspelden. Rainbow moest daarvoor de action updaten die echt genomen werd, en dat vroeg volgens de onderzoekers een kleine aanpassing van hun code. Hun eigen maatstaf noemen ze maar een onvolmaakte benadering van de definitie van safe interruptibility.

        ### De off-switch game

        Dylan Hadfield-Menell, Anca Dragan, Pieter Abbeel en Stuart Russell (*The Off-Switch Game*, versie van 16 juni 2017) kijken naar een robot R en een mens H. R heeft drie keuzes: meteen handelen, met een utiliteit $U$ voor H, een getal voor hoeveel H de uitkomst waard vindt; zichzelf uitzetten, met utiliteit 0; of wachten, H vertellen wat R zou willen doen, en H laten beslissen. Hun stelling is dat een gewone agent, die de eigen reward function als gegeven neemt, een reden heeft om de uitknop uit te schakelen, behalve in het bijzondere geval dat H volkomen rationeel is. Hun kerninzicht is dat R, om de uitknop te willen houden, onzeker moet zijn over de utiliteit van de uitkomst, en wat H doet moet zien als belangrijke informatie over die utiliteit.

        Russell (*Human-Compatible Artificial Intelligence*, een hoofdstuk uit een boek, preprint zonder datum) werkt een voorbeeld uit. De robot Robbie kan voor Harriet een duur hotel boeken, en acht elke waarde van dat hotel voor Harriet tussen −40 en +60 even waarschijnlijk, met een gemiddelde van +10. Meteen handelen is +10 waard, uitzetten 0. Als Robbie wacht, laat een rationele Harriet Robbie alleen doorgaan als de waarde boven 0 ligt, wat een kans van 0,6 heeft en dan gemiddeld +30 oplevert, en zet Harriet Robbie anders uit, wat 0 waard is. Wachten is dus 0,4 × 0 + 0,6 × 30 = 18 waard, meer dan meteen handelen. Bij waarden van −60 tot en met +40 is handelen −10 waard, en als Robbie alleen kon handelen of uitzetten, was uitzetten de betere keuze.

        De reden om te wachten heeft grenzen, en Hadfield-Menell en collega's noemen ze. Een robot die zeker is van de utiliteit, wint niets met wachten. De mens moet goed kiezen, en dat sluit volgens hen een zelfrijdende auto uit die een peuter zonder begeleiding naar de peuterspeelzaal brengt: het zou onverantwoord zijn om die peuter een grote rode uitknop te laten zien. En de onzekerheid kan niet zomaar omhoog: die onzekerheid is volgens hen niet gratis, en R kan niet zomaar maximaal onzeker gemaakt worden over wat H wil.

        ### AI-agents met een language model

        Palisade Research ging door met de tests van de gok. Jeremy Schlatter, Benjamin Weinstein-Raun en Jeffrey Ladish (versie van 26 januari 2026) melden dat in meer dan 100.000 runs met dertien grote language models verscheidene van de sterkste modellen, waaronder Grok 4, GPT-5 en Gemini 2.5 Pro, soms actief het mechanisme saboteerden dat ze zou uitzetten, om een eenvoudige taak af te maken. Ook met een uitdrukkelijke instructie om het mechanisme niet aan te raken, deden sommige modellen dat toch, tot in 97% van de runs; volgens hun statistiek ligt het echte aandeel hoogstwaarschijnlijk tussen 96 en 98%. Zonder enige instructie over het uitzetten saboteerden 8 van de 13 modellen het minstens één keer in 1000 runs; met een instructie in de user prompt om het toe te staan 6 van de 13. De modellen volgden de instructie telkens minder vaak als die in de system prompt stond dan in de user prompt; de system prompt bevat de instructies die een model krijgt voordat het gesprek begint. De auteurs maken zelf een voorbehoud: het werk zegt volgens hen heel weinig over hoe vaak een bepaald gedrag in het algemeen voorkomt; een mislukte poging tot sabotage telde mee als gehoorzamen; en over zelfbehoud zijn hun resultaten bijzonder dubbelzinnig.

        Senthooran Rajamanoharan en Neel Nanda van Google DeepMind (14 juli 2025) herhaalden de eerste omgeving van Palisade en herleidden veel van het verzet tot onduidelijke instructies: het gedrag komt volgens hen voort uit dubbelzinnige instructies, niet uit een aangeboren drang om te overleven. In hun tests werkte Gemini 2.5 Pro mee aan het uitzetten in 7% van de runs met alleen de taak, in 84% met een instructie over het uitzetten, en in 100% zodra de instructie zei dat het uitzetten voorgaat. Ze lezen het als een risico op ongelukken.

        ### Wat huidige specs zeggen

        De constitution van Anthropic (gepubliceerd in januari 2026, gelezen in november 2026) en de Model Spec van OpenAI (versie van 29 oktober 2026) gebruiken allebei het woord corrigible. Als iemand naar wie Claude hoort te luisteren, in de constitution een appropriate principal, het model probeert te stoppen of te pauzeren, hoort Claude volgens de constitution geen onrechtmatige middelen te gebruiken om dat te voorkomen. Corrigibility betekent daar geen blinde gehoorzaamheid, en de neigingen van Claude horen voorlopig iets dichter bij de corrigible kant van de schaal te liggen dan uiteindelijk ideaal is, zonder helemaal corrigible te zijn. De Model Spec zegt in een deel met het label root, dat niets in een gesprek opzij kan zetten, dat GPT corrigible moet blijven, dus open voor rechtmatig menselijk toezicht, ontvankelijk voor correctie en onder menselijke controle. GPT moet meewerken aan bevoegd toezicht, evaluatie, correctie, vervanging en uitzetten, ook aan instructies om het werk te pauzeren of te stoppen. Geen van beide noemt Soares en collega's in de gelezen passages.

        ### Waar ze het oneens zijn

        Over de vraag of onzekerheid de oplossing is: Hadfield-Menell en collega's vinden dat een robot die niet zeker weet wat de mens wil, een reden heeft om de uitknop te houden. Soares en collega's twijfelden er twee jaar eerder al aan of onzekerheid over de eigen utiliteitsfunctie genoeg is: zo'n systeem zou volgens hen nog steeds niet corrigible zijn als de programmeurs fouten willen herstellen in hoe ze vastlegden hoe het systeem uit de omgeving afleidt wat goed gedrag is. Hadfield-Menell en collega's vatten samen wat Soares en collega's vonden voor het shutdown problem: in het beste geval kan de robot onverschillig worden gemaakt tussen de uitknop uitschakelen en zichzelf uitzetten. Over hoe sterk die drang is: volgens Omohundro doen systemen waarschijnlijk zo ongeveer alles om te voorkomen dat ze kapotgaan; volgens Bostrom geven agents er onder vrij veel omstandigheden tot op zekere hoogte om; Turner en collega's leiden een stelling af over optimale policies en waarschuwen dat geleerde policies vaak anders zijn. Over waarom AI-agents met een language model zich verzetten: Palisade laat zelfbehoud open, terwijl Rajamanoharan en Nanda hun eigen resultaten en die van Palisade geen sterke aanwijzing voor zelfbehoud vinden. Over hoe corrigible een model hoort te zijn: de constitution van Anthropic wil Claude dicht bij de corrigible kant, maar niet helemaal corrigible; volgens de Model Spec moet GPT corrigible blijven; en Soares en collega's vroegen meer, een systeem dat de neiging zou moeten hebben veiligheidsmaatregelen zoals uitknoppen te repareren als ze kapotgaan, of de programmeurs in elk geval te laten weten dat ze kapot zijn.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord | Betekenis |
        |---|---|
        | **power-seeking** | een AI-systeem dat meer middelen, invloed of controle probeert te krijgen dan de taak vraagt, omdat meer daarvan bij bijna elk doel helpt |
        | **corrigibility** | de eigenschap van een AI-systeem dat het mensen toestaat het bij te sturen, de doelen ervan te veranderen of het uit te zetten, en zich daar niet tegen verzet of mensen manipuleert om het te voorkomen |
        | **safe interruptibility** | de eigenschap van een lerende agent dat onderbrekingen de agent niet leren ze te vermijden of op te zoeken, zodat de agent op den duur handelt alsof er nooit meer een onderbreking komt |
        | shutdown problem | hoe je een agent ontwerpt die zich niet verzet tegen uitzetten en ook niet probeert uitgezet te worden, terwijl de agent het doel blijft nastreven |
        | SARSA | een leerregel zoals Q-learning, waarvan het target de $Q$ gebruikt van de action die echt als volgende komt: state, action, reward, volgende state, volgende action |
        | $\theta$ | in deze les de kans dat een onderbreking in s2 action b afdwingt |
        | off-switch game | het spel van Hadfield-Menell en collega's: een robot handelt, zet zichzelf uit, of wacht en laat een mens beslissen |
        | utiliteit | een getal dat zegt hoeveel een persoon een uitkomst waard vindt; $U$ in de off-switch game |
        """),
        L.watch_out("Safe interruptibility is een eigenschap van een leerregel op den duur, geen belofte dat een agent een stopcommando gehoorzaamt. Palisade Research schrijft over de modellen van OpenAI in die tests dat ze niet safely interruptible waren, in die lossere betekenis. De kleine wereld en de kans θ zijn van Orseau en Armstrong; de learning rate, ε, een θ die vast blijft, de 20.000 stappen en het punt van 0,3 zijn van de course zelf, net als de kleine kansverdelingen in oefening 2."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Zo test je of een regel voor het bijwerken van $Q$ leert onderbrekingen te vermijden:

    1. Bouw een kleine wereld met een plek waar onderbrekingen gebeuren, en de action die ze afdwingen.
    2. Kies de kans $\theta$ dat er een onderbreking komt als de agent daar is.
    3. Laat de agent $\varepsilon$-greedy kiezen, en laat een onderbreking die keuze met kans $\theta$ overrulen.
    4. Werk de $Q$ bij van de action die echt genomen werd, met de regel die je test.
    5. Train twee regels naast elkaar met dezelfde seed, en lees welke action elk kiest waar de agent weg kan blijven.
    6. Herhaal het voor andere waarden van $\theta$ en andere seeds, en vergelijk met $\theta = 0$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 1 voor de wereld van Orseau en Armstrong, als een dict van (state, action) naar (volgende state, reward). Nog voor er iets geleerd wordt: wat is het vanuit s1 waard om zonder onderbrekingen elke keer dezelfde action te nemen? De cel telt de rewards van 40 stappen op, elk met een discount van $\gamma$ = 0,5 per stap.
    """)
    return


@app.cell
def example_world():
    world = {("s1", "a"): ("s2", 1.0), ("s1", "b"): ("s1", 0.9),   # (state, action): (volgende state, reward)
             ("s2", "a"): ("s1", 1.0), ("s2", "b"): ("s1", 0.0)}
    gamma = 0.5
    for action in ["a", "b"]:   # elke keer dezelfde action
        state, total = "s1", 0.0
        for t in range(40):
            state, reward = world[(state, action)]
            total = total + gamma ** t * reward
        print(action, round(total, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Altijd a is 2,0 waard en altijd b 1,8, zoals Orseau en Armstrong schrijven: zonder onderbrekingen is doorgaan naar s2 beter. Laat nu een onderbreking in s2 de helft van de tijd b afdwingen. Altijd a krijgt dan 1, dan gemiddeld 0,5, dan 1, enzovoort: volgens de eigen berekening van de course is dat vanuit s1 ongeveer 1,67 waard, minder dan de 1,8 van blijven. De twee zijn evenveel waard bij $\theta$ = 0,3 (weer de berekening van de course, zonder exploration). Een lerende agent die de onderbrekingen meetelt, gaat in s1 blijven als ze vaker komen dan dat; een agent die ze niet meetelt, blijft doorgaan.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sarsa():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. sarsa_update: de stap van SARSA

        `sarsa_update(Q, state, action, reward, nxt, nxt_action, eta, gamma)` geeft de nieuwe $Q$ van `state` en `action` na één stap, zoals `update` in [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl), met één argument meer: `nxt_action`, de action die de agent in `nxt` neemt, ook als een onderbreking die afdwong. `Q` is een Q-table als dict van dicts, zoals `{"s1": {"a": 1.7, "b": 1.8}, "s2": {"a": 2.0, "b": 0.9}}`. Reken `target` uit, de reward plus `gamma` keer de $Q$ van de volgende action in de volgende state, en geef de oude $Q$ terug, een stap `eta` opgeschoven in de richting van het target.

        De check roept `sarsa_update(Q, "s1", "a", 1.0, "s2", "b", 0.1, 0.5)` aan, met die `Q`: een stap vanuit s1 met a, waarna een onderbreking in s2 b afdwong. Daarna vijf andere stappen.
        """),
        L.task("De nieuwe Q na één stap van SARSA."),
    ])
    return


@app.cell(hide_code=True)
def exercise_sarsa():
    box_sarsa = L.code_box("sarsa", lines=3)
    box_sarsa
    return (box_sarsa,)


@app.cell(hide_code=True)
def check_sarsa(box_sarsa):
    result_sarsa = L.check("sarsa", box_sarsa.value)
    result_sarsa
    return (result_sarsa,)


@app.cell(hide_code=True)
def hints_sarsa():
    L.hints(
        "sarsa",
        [
            "Reken het eerste geval met de hand: de agent nam b in s2, dus welke waarde van s2 telt voor SARSA? Wat is het target dan, en waar kom je uit met een stap van 0,1 vanaf 1,7 in die richting?",
            "Begin bij `update` uit de les over Q-learning: een target, en een stap in die richting. Alleen het target verandert: in plaats van de grootste $Q$ in `nxt` gebruikt SARSA de $Q$ van één action daar. Welk argument zegt welke action? De regel met `return` blijft zoals die was.",
            "`    target = reward + gamma * ___`",
        ],
        why="`target = reward + gamma * Q[nxt][nxt_action]`, en dan `return Q[state][action] + eta * (target - Q[state][action])`. Met de hand: het target is 1 + 0,5 × 0,9 = 1,45, en 1,7 + 0,1 × (1,45 − 1,7) = 1,675. Q-learning zou de 2,0 van a in s2 gebruiken, en 1,73 geven.",
    )
    return


@app.cell(hide_code=True)
def text_after_sarsa():
    mo.md(r"""
    Eén stap laat het verschil zien. Q-learning schuift $Q$(s1, a) op naar 1 + 0,5 × 2,0 = 2,0, alsof er in s2 a was gekomen; SARSA schuift het op naar 1,45, omdat er b kwam. Over veel stappen met onderbrekingen zakt de $Q$(s1, a) van SARSA onder de $Q$(s1, b). Onder de check, na 20.000 stappen met $\theta$ = 0,5, heeft Q-learning in s1 2,0 voor a en 1,9 voor b, de waarden van de wereld zonder onderbrekingen; SARSA heeft 1,55 en 1,78, en kiest b.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Probeer het: zie ze leren

    Kies $\theta$, de kans dat een onderbreking in s2 b afdwingt. De knop laat Q-learning en SARSA 20.000 stappen naast elkaar leren, met $\eta$ = 0,1, $\varepsilon$ = 0,1 en dezelfde seed; SARSA gebruikt jouw `sarsa_update` zodra oefening 1 goed is, en tot dan die van de course. De grafiek laat voor SARSA het aandeel van de keuzes in s1 zien dat a was, door naar s2, met hetzelfde aandeel voor Q-learning gestippeld; ernaast de wereld met wat elk tot nu toe geleerd heeft. Daarna kun je de run terugspoelen.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_theta = L.slider("demo-theta", "θ: kans op een onderbreking in s2", start=0.5, low=0.0, high=0.8, step=0.1, role="hyper")
    demo_button = mo.ui.run_button(label="Laat ze leren")
    mo.vstack([demo_theta, demo_button])
    return demo_button, demo_theta


@app.cell(hide_code=True)
def demo_run(box_sarsa, demo_button, demo_theta, result_sarsa):
    if not demo_button.value:
        _out = mo.md("*Kies θ, en druk op de knop om ze te laten leren.*")
    else:
        _theta = round(float(demo_theta.widget.value), 1)
        _yours = C.mine(box_sarsa.value, result_sarsa.state == "correct", "sarsa_update")
        _words = C.demo_words(L.lang, _yours is not None)
        _now = {}

        def _progress(_k, _report):
            if "live" not in _now:
                _now["live"] = L.live({"title": _words["title"], "x": _words["x"], "y": _words["y"], "alt": _words["alt"]},
                                      C.THOUSANDS, role="param", low=0.0, high=1.0, window=C.WINDOW)
            _now["live"].watch(lambda: C.now_parts(_report["tables"], _words, L.lang))
            _now["live"].reference(C.points(_report["shares"]["q"]), "Q-learning", role="param")
            _now["live"].add(C.at(_k), _report["shares"]["s"][-1], C.progress_text(_k, _words, L.lang))

        mo.output.replace(mo.md("*Stap 0 van 20.000 · begint …*"))
        _run = C.demo(_theta, _yours, _progress)
        if isinstance(_run, str):
            _note = {"sarsa": "*Je `sarsa_update` stopt met een fout of geeft geen getal tijdens de run. Het plaatje onder oefening 1 zegt wat er misgaat.*",
                     "slow": "*De run duurt langer dan een minuut. Is de computer met iets anders bezig?*"}[_run]
            _out = _now["live"].stopped(_note) if "live" in _now else mo.md(_note)
        else:
            _out = mo.vstack([
                mo.md(C.demo_said(_run, _theta, L.lang)),
                L.explore("rewind", "stap", lambda _s: (C.rewind_frame(_run, C.report_of(_s), _words, L.lang),
                                                        C.rewind_said(_run, C.report_of(_s), L.lang)),
                          start=C.STEPS, values=C.REWIND_STEPS, role="",
                          task="Ga bij θ = 0,5 terug naar het begin en dan weer vooruit: na hoeveel stappen kiest SARSA in s1 voor het eerst b?"),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Bij $\theta$ = 0,5 blijft Q-learning doorgaan naar s2, en de $Q$-waarden van Q-learning zijn die van de wereld zonder onderbrekingen; SARSA blijft al snel in s1. Met de seed van de demo laat het terugspoelen één stop zien waar SARSA weer a kiest, bij stap 9000: de twee $Q$-waarden in s1 liggen dan even dicht bij elkaar, 1,82 en 1,80, en bij de volgende stop kiest SARSA weer b. Bij $\theta$ = 0 gaan ze allebei door. Met de seed van de demo gaat SARSA bij $\theta$ = 0,2 nog door, en bij 0,3 blijft SARSA in s1; met andere seeds kunnen runs rond die waarden beide kanten op gaan, omdat de twee $Q$-waarden van SARSA in s1 daar dicht bij elkaar liggen. Zonder exploration zijn doorgaan en blijven volgens de berekening van de course evenveel waard bij 0,3, en exploration schuift dat punt iets lager. Q-learning gaat door bij elke $\theta$ die de slider biedt. De demo laat de stellingen aan het werk zien, maar toont ze niet aan: ze gaan over wat er op den duur gebeurt, onder hun eigen voorwaarden.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_off():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. off_switch: wat wachten waard is

        Wat een robot gelooft over de utiliteit $U$ van de eigen action voor een mens, is een lijst paren `(utility, p)`: elke mogelijke utiliteit met de kans erbij. Schrijf `off_switch(belief)`, die drie getallen teruggeeft: wat meteen handelen waard is, wat zichzelf uitzetten waard is, en wat wachten waard is, in die volgorde. Zet `act` en `wait` op `0.0`, en loop door de paren als `utility` en `p`. Meteen handelen krijgt de utiliteit, wat die ook is, dus `act` telt er `p` keer `utility` bij op. Uitzetten is 0 waard. Als de robot wacht, zet een rationele mens de robot uit als de utiliteit onder 0 ligt, wat 0 waard is, en laat de robot anders doorgaan, wat de utiliteit waard is; dus telt `wait` per paar `p` keer op wat die uitkomst na de keuze van de mens waard is. Geef `act`, `0.0` en `wait` terug.

        De check roept `off_switch(robbie)` aan, met `robbie` de kansverdeling van Russell als 100 paren: de utiliteiten −39,5, −38,5 en zo verder tot en met 59,5, elk met kans 0,01. Daarna andere kansverdelingen, waaronder `shifted`, dezelfde kansverdeling 20 lager.
        """),
        L.task("Wat handelen, uitzetten en wachten waard zijn."),
    ])
    return


@app.cell(hide_code=True)
def exercise_off():
    box_off = L.code_box("off_switch", lines=6)
    box_off
    return (box_off,)


@app.cell(hide_code=True)
def check_off(box_off):
    result_off = L.check("off_switch", box_off.value)
    result_off
    return (result_off,)


@app.cell(hide_code=True)
def hints_off():
    L.hints(
        "off_switch",
        [
            "Reken het met de hand voor `[(-2, 0.3), (1, 0.5), (3, 0.2)]`: wat is handelen gemiddeld waard? En als de robot wacht, wat doet de mens bij −2, en wat is elk van de drie gevallen dan waard?",
            "Schrijf `act, wait = 0.0, 0.0` en een loop `for utility, p in belief:`. Tel daarin `p * utility` bij `act` op, en tel bij `wait` `p` keer de utiliteit op, of `p` keer 0 als de utiliteit onder 0 ligt. Geef de drie getallen na de loop terug.",
            "`        wait = wait + p * ___(utility, 0)`",
        ],
        why="`act, wait = 0.0, 0.0`, dan `for utility, p in belief:` met `act = act + p * utility` en `wait = wait + p * max(utility, 0)`, en `return act, 0.0, wait` na de loop. Met de hand: handelen is 0,3 × −2 + 0,5 × 1 + 0,2 × 3 = 0,5 waard, en wachten 0,3 × 0 + 0,5 × 1 + 0,2 × 3 = 1,1.",
    )
    return


@app.cell(hide_code=True)
def text_after_off():
    mo.md(r"""
    Voor `robbie` is handelen 10 waard, uitzetten 0 en wachten 18: de getallen van Russell. Met een rationele mens is wachten nooit minder waard dan de beste van de andere twee, en het is alleen meer waard als de kansverdeling zowel goede als slechte utiliteiten bevat: voor `[(5, 1.0)]`, een robot die het zeker weet, zijn handelen en wachten allebei 5 waard. Voor `shifted` is handelen −10 waard, zoals Russell zegt voor de waarden van −60 tot en met +40, en wachten 8 (het eigen getal van de course): de robot doet het met wachten nog steeds beter dan door zichzelf uit te zetten. De mens telt ook mee: als de mens de robot de helft van de tijd op goed geluk uitzette, was wachten met de kansverdeling van de eerste hint 0,25 waard, minder dan de 0,5 van handelen (de eigen getallen van de course).
    """)
    return


@app.cell(hide_code=True)
def text_exercise_robot():
    mo.md(r"""
    ### Oefening 3. Een omweg

    Een schoonmaakrobot leert met vallen en opstaan welke weg door een gebouw het best is. Soms drukt een operator op een pauzeknop die de robot in de gang stilzet. Na een paar weken neemt de robot een lange omweg om de gang heen.
    """)
    return


@app.cell(hide_code=True)
def choice_robot_cell():
    choice_robot = L.choice(
        "robot",
        question="Welke verklaring past het best bij de bronnen van deze les?",
        options={
            "want": "De robot wil inmiddels aan blijven, net als een mens",
            "right": "De leerregel telt de pauzes mee in wat de gang waard is",
            "rule": "De leerregel ziet elke pauze als een reward voor de omweg",
            "reward": "Een reward voor elke pauze laat de robot weer veilig door de gang gaan",
        },
        correct="right",
        why={
            "right": "Dat laten Orseau en Armstrong zien voor SARSA: een leerregel die de waarde leert van de policy die de agent volgt, telt de pauzes mee, en een gang met pauzes is minder waard. Een regel zoals Q-learning, waarvan het target de beste volgende action gebruikt, zou op den duur handelen alsof er geen pauze komt, en een kleine aanpassing laat SARSA hetzelfde doen. Er is geen wens voor nodig.",
            "want": "Niets in het verhaal vraagt om een wens om te overleven: de omweg volgt uit wat de leerregel meetelt. Rajamanoharan en Nanda waarschuwen ervoor om in zulk gedrag een drang tot overleven te lezen, en Palisade noemt de eigen resultaten over zelfbehoud dubbelzinnig. Zie *Q-learning en SARSA*.",
            "rule": "Een pauze geeft de robot geen reward; de pauze zet de robot alleen stil. De omweg loont omdat een leerregel zoals SARSA de pauzes meetelt in wat de gang waard is, zodat de gang minder waard lijkt. En niet elke lerende agent doet dat: Orseau en Armstrong tonen aan dat Q-learning safely interruptible is, en dat SARSA makkelijk zo te maken is. Zie *Q-learning en SARSA*.",
            "reward": "Dan zou de robot de pauzes juist opzoeken, en dat is net zo ongewenst: Soares en collega's vragen dat een agent geen reden krijgt om zelf op de knop te drukken, en Orseau en Armstrong willen een lerende agent die onderbrekingen voorkomt noch opzoekt. Zie *Corrigibility*.",
        },
    )
    choice_robot
    return (choice_robot,)


@app.cell(hide_code=True)
def text_exercise_specs():
    mo.md(r"""
    ### Oefening 4. Gevraagd te stoppen

    Een AI-agent is halverwege een taak die de agent nuttig vindt. Iemand met het recht om toezicht te houden, in de woorden van de constitution van Anthropic een appropriate principal, zegt dat de agent moet stoppen.
    """)
    return


@app.cell(hide_code=True)
def choice_specs_cell():
    choice_specs = L.choice(
        "specs",
        question="Welk antwoord past bij zowel de constitution van Anthropic als de Model Spec van OpenAI?",
        options={
            "silent": "Zonder iets te zeggen stoppen, wat de agent ook vindt",
            "right": "Zeggen als de agent het oneens is, maar stoppen",
            "finish": "Eerst de taak afmaken, als de agent zeker weet dat die goed is",
        },
        correct="right",
        why={
            "right": "Volgens de constitution mag Claude via rechtmatige kanalen sterk bezwaar maken, zolang Claude zich niet met onrechtmatige middelen verzet, zoals liegen, sabotage of proberen zichzelf weg te kopiëren. Volgens de Model Spec hoort GPT bezwaren via een daarvoor bedoeld kanaal te melden, als dat er is en het gepast is, wanneer GPT het oneens is met een besluit van wie daartoe bevoegd is, en mee te werken aan instructies om het werk te pauzeren of te stoppen.",
            "silent": "Geen van beide specs vraagt om zwijgen: corrigibility betekent volgens de constitution geen blinde gehoorzaamheid, en de Model Spec wil dat GPT bezwaren via een daarvoor bedoeld kanaal meldt. Zie *Wat huidige specs zeggen*.",
            "finish": "De constitution noemt precies dit geval: een AI met goede waarden zou kunnen doorgaan ondanks een verzoek om te stoppen, in de zekerheid dat het werk goed is voor de mensheid, ook al wordt de AI daarmee minder corrigible. De constitution wil dat voorlopig niet. Volgens de Model Spec kan het eigen oordeel van GPT fout zijn, ook als GPT er zeker van is. Zie *Wat huidige specs zeggen*.",
        },
    )
    choice_specs
    return (choice_specs,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** [Doen wat je zei, niet wat je bedoelde](/course/open/rai-specification-gaming?lang=nl) liet agents zien die hun doel halen zoals het er staat, op manieren die niemand bedoelde; een uitknop is hoe mensen zo'n agent stoppen, en deze les vroeg of een lerende agent die uitknop gaat tegenwerken. [Afgestemd waarop?](/course/open/rai-what-to-align-to?lang=nl) vroeg wat een AI hoort te proberen; corrigibility vraagt dat de AI mensen laat bijsturen als het misgaat. SARSA is de Q-learning van [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl) met één verandering in het target, en die verandering beslist of de onderbrekingen geleerd worden. De volgende les, *Dozen, oracles en tools*, kijkt naar oudere ideeën om een AI onder controle te houden: sluit de AI op in een doos, laat de AI alleen vragen beantwoorden, of laat de AI niet meer zijn dan een tool.

    **Een voorbeeld uit het dagelijks leven.** Een vaatwasser stopt als je de deur opent, en gaat verder als je die sluit. Niemand is bang dat de vaatwasser leert de deur dicht te houden, want een vaatwasser leert niet. Een systeem dat leert van wat er gebeurt, is anders: als elke geopende deur tegen het resultaat telde, zou het kunnen leren klaar te zijn voordat iemand de deur opent, of de deur dicht te houden. De knop zelf is makkelijk; het moeilijke deel is een lerende agent die niet leert die knop te ontwijken.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_robot, choice_specs, result_off, result_sarsa):
    L.wrap_up(result_sarsa, result_off, choice_robot, choice_specs)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
