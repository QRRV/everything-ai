# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-monte-carlo-control. Route: course/route/lessen/reinforcement-learning.yaml. Checks, de blackjack van de course, de uitgerekende waarden en runs, en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-monte-carlo-control", "nl")
    C = L.checks   # de blackjack van de course, de uitgerekende waarden en runs, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Leren van hele potjes: Monte Carlo prediction](/course/open/rl-monte-carlo-prediction?lang=nl) en [Steeds beter: policy iteration](/course/open/rl-policy-iteration?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_first():
    quick_first = L.choice(
        "quick-first",
        question="Met first-visit Monte Carlo schat een agent $V(s)$ van een policy uit episodes die met deze policy gespeeld zijn. Wat is de schatting voor een state $s$ na 50 episodes die elk $s$ bezochten?",
        options={
            "last": "De return van de laatste van de 50 episodes, de meest recente",
            "right": "Het gemiddelde van de 50 returns na het eerste bezoek aan $s$ in elke episode",
            "best": "De grootste van de 50 returns, het beste wat de policy vanuit $s$ deed",
            "reward": "Het gemiddelde van de rewards direct na elk eerste bezoek aan $s$, één per episode, over alle 50 episodes",
        },
        correct="right",
        why={
            "right": "Elk eerste bezoek geeft één return die echt gebeurde, en het gemiddelde van deze returns komt dicht bij de return die je vanuit $s$ kunt verwachten. Deze les doet hetzelfde voor $Q(s, a)$: het gemiddelde van de returns na action $a$ in state $s$.",
            "last": "Eén return kan ver van de verwachte return liggen: de returns van de robot vanuit de hoek hebben een spreiding van ongeveer 32 stappen. Wat doet Monte Carlo met alle 50? Zie [Leren van hele potjes: Monte Carlo prediction](/course/open/rl-monte-carlo-prediction?lang=nl).",
            "best": "De waarde van een state is de return die je kunt verwachten als je de policy volgt, niet de beste return ooit: welk getal van de 50 komt dicht bij wat je kunt verwachten? Zie [Leren van hele potjes: Monte Carlo prediction](/course/open/rl-monte-carlo-prediction?lang=nl).",
            "reward": "De waarde van een state telt elke reward tot het einde van de episode, niet alleen de eerste: welk getal van elke episode telt alle rewards mee? Zie [Leren van hele potjes: Monte Carlo prediction](/course/open/rl-monte-carlo-prediction?lang=nl).",
        },
    )
    quick_first
    return (quick_first,)


@app.cell(hide_code=True)
def quick_check_improve():
    quick_improve = L.choice(
        "quick-improve",
        question="In policy iteration maakt de stap policy improvement een nieuwe policy uit de waarden $V$ van de oude. Hoe?",
        options={
            "copy": "De stap behoudt de oude policy en schat alleen $V$ beter",
            "random": "De stap kiest een willekeurige action in elke state waar $V$ onder het gemiddelde van alle states ligt, om iets nieuws te proberen",
            "right": "In elke state kiest de stap de action met de grootste $Q$, uitgerekend uit $V$ en de tabel `T`",
            "highest": "De stap gaat naar de naburige state met de grootste $V$, zonder naar `T` te kijken",
        },
        correct="right",
        why={
            "right": "$Q(s, a)$ is wat het waard is om één keer $a$ te nemen en daarna de oude policy te volgen, en om dit uit te rekenen heb je `T` nodig: waar elke action heen kan leiden, en met welke kans. Zonder `T` schat deze les $Q(s, a)$ zelf, uit gespeelde handen.",
            "copy": "Policy iteration wisselt af: evalueren, dan verbeteren. Wat verandert er bij policy improvement, $V$ of de policy? Zie [Steeds beter: policy iteration](/course/open/rl-policy-iteration?lang=nl).",
            "random": "De policy improvement theorem zegt dat de nieuwe policy in elke state minstens zo goed is als de oude: kan een willekeurige action dat beloven? Zie [Steeds beter: policy iteration](/course/open/rl-policy-iteration?lang=nl).",
            "highest": "Een action kan naar meerdere states leiden, elk met een eigen kans: hoe weegt policy improvement deze states? Zie [Steeds beter: policy iteration](/course/open/rl-policy-iteration?lang=nl).",
        },
    )
    quick_improve
    return (quick_improve,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Policy iteration maakte een policy beter, maar had de tabel `T` nodig. Monte Carlo prediction had geen tabel nodig, maar zei alleen wat een vaste policy waard is. Samen geven deze twee een manier om alleen uit potjes goed te leren spelen: schat uit de potjes die je speelt wat elke action oplevert, en speel de action die meer opleverde. Zo kan een agent een kaartspel leren, een weg door het verkeer of een strategie tegen mensen, waar niemand de kansen kan opschrijven. Blackjack is het klassieke voorbeeld: de course kan de kansen uitrekenen, al ziet de agent deze kansen nooit, en de beste policy is al sinds de jaren 50 bekend, dus je kunt zien hoe dicht leren uit handen erbij komt.

    **Een voorbeeld uit het dagelijks leven.** Wanneer stop je met zoeken naar een parkeerplek dicht bij de deur, en neem je een plek verder weg? Niemand rekent de kansen van elke straat uit. Je probeert af en toe allebei, onthoudt hoe lang elke keuze uiteindelijk duurde, en doet vaker wat werkte. Als je altijd alleen zou doen wat tot nu toe het beste lijkt, zou je nooit merken dat de andere keuze toch beter was.
    """)
    return


@app.cell(hide_code=True)
def predict_worth():
    guess_worth = L.guess(
        "worth",
        question="Bij blackjack neem je kaarten om dichter bij 21 te komen dan de dealer, en boven 21 verlies je meteen. In de blackjack van de course levert een hand +1 op als je wint, 0 bij gelijkspel en −1 als je verliest. Monte Carlo control kan de beste policy vinden. Wat levert een hand met deze beste policy gemiddeld op?",
        options={
            "win": "Ongeveer +0,05: met de beste manier van spelen win je op den duur een beetje",
            "zero": "Ongeveer 0: met de beste manier van spelen win je even vaak als je verliest",
            "right": "Ongeveer −0,05",
            "start": "Ongeveer −0,35, net zoveel als een policy die een kaart neemt tot 20, omdat de dealer altijd in het voordeel is",
        },
        reveal="**Ongeveer −0,05.** De beste policy verliest 0,047 per hand: ongeveer 5 van elke 100 inzetten gaan naar de dealer. Het voordeel van de dealer komt voort uit één regel: jij speelt eerst, en als je boven 21 komt verlies je meteen, ook als de dealer daarna boven 21 was gekomen. De agent begint met de policy *een kaart nemen onder 20*, en deze policy verliest 0,354 per hand.",
    )
    guess_worth
    return (guess_worth,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **De blackjack van de course.** Je speelt tegen de dealer. De kaarten komen uit een stapel die nooit opraakt: een aas telt 1 of 11, een 2 tot en met 9 telt het getal op de kaart, en 10, boer, vrouw en heer tellen 10. Je krijgt twee kaarten, en de dealer één kaart die je ziet en één dichte kaart. Je mag zo vaak als je wilt een kaart nemen, **hit**; boven 21 verlies je meteen. Als je stopt, **stick**, draait de dealer de dichte kaart om en neemt kaarten tot de hand 17 of meer telt. Zit je dichter bij 21 dan de dealer, dan win je, +1; met hetzelfde totaal is het gelijkspel, 0; en als de dealer boven 21 komt win je ook. Een aas die als 11 kan tellen zonder dat de hand boven 21 komt is een **bruikbare aas**: daarmee kun je een kaart nemen zonder boven 21 te komen, omdat de aas terug kan naar 1. De course speelt zonder de extra regels van een casino, zoals verdubbelen en splitsen: een aas en een 10 als je eerste twee kaarten zijn een gewone 21, en bij 21 van de dealer is het gelijkspel.

        Een state is wat je weet als je beslist: je totaal van 12 tot en met 21, de kaart van de dealer, en of je een bruikbare aas hebt, dus 10 · 10 · 2 = 200 states. Onder 12 neem je altijd een kaart, omdat geen kaart je dan boven 21 kan brengen. De reward komt aan het einde, en met $\gamma$ = 1 is de return van elke beslissing in een hand deze ene reward.

        **Monte Carlo control.** Bij prediction schatte je de waarden van een vaste policy. Om een policy beter te maken koos policy iteration in elke state de action met de grootste $Q$, uitgerekend uit $V$ met de tabel `T`. Zonder `T` zegt $V$ niet welke action beter is: wat er komt nadat je een kaart neemt hangt af van kaarten die je niet kunt opzoeken. Dus schat **Monte Carlo control** $Q(s, a)$ zelf, als het gemiddelde van de returns na action $a$ in state $s$. Met $n$ het aantal returns van $(s, a)$ tot nu toe schuift de nieuwe return $G$ deze schatting op:
        """),
        L.formula(r"[[param:Q(s, a)]] \leftarrow [[param:Q(s, a)]] + \frac{1}{n} \left( [[loss:G]] - [[param:Q(s, a)]] \right)"),
        mo.md(r"""
        Dit heeft de vorm van de update van Q-learning, met de hele return $G$ als target in plaats van $r + \gamma \max_{a'} Q(s', a')$, en met $\eta = 1/n$ als learning rate. Daarmee blijft $Q$ precies het gemiddelde van de $n$ returns: na de eerste return is $Q = G_1$, na de tweede $G_1 + \frac{1}{2}(G_2 - G_1) = \frac{1}{2}(G_1 + G_2)$, enzovoort. Na elke hand kiest de policy in elke state van de hand de action met de grotere $Q$, en bij gelijke $Q$ stopt de policy. Evalueren en verbeteren wisselen elkaar hand voor hand af: generalized policy iteration, zonder de tabel. In een hand blackjack komt elke state hoogstens één keer voor, omdat het totaal alleen groeit of de aas niet meer bruikbaar wordt, dus first visit en every visit zijn hier hetzelfde.

        **Beide actions blijven proberen.** Een policy die altijd de action met de grotere $Q$ kiest komt nooit te weten wat de andere action oplevert: de $Q$ van een action die de policy niet kiest blijft staan waar deze stond. Er zijn twee uitwegen. Met **exploring starts** begint elke episode in een willekeurige state met een willekeurige eerste action, zodat alle states en actions steeds weer voorkomen; na deze eerste action speelt de policy. Een **ε-soft policy** geeft elke action een kans van minstens $\varepsilon$ gedeeld door het aantal actions. ε-greedy uit [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl) is zo'n policy: met kans $\varepsilon$ is de zet willekeurig, en een willekeurige zet kan ook de action van de policy zelf kiezen. Met twee actions krijgt elke action minstens $\varepsilon/2$: de helft van de willekeurige zetten kiest deze action. Monte Carlo control met ε-greedy leert de beste van de ε-soft policies: de beste manier van spelen als je willekeurige zetten blijft doen.

        ### Nieuwe woorden en tekens

        | Woord of teken | Zeg je als | Wat het betekent | In code |
        |---|---|---|---|
        | **Monte Carlo control** | | $Q(s, a)$ schatten uit episodes en de policy na elke episode de action met de grotere $Q$ laten kiezen | `update` |
        | **exploring starts** | | elke episode begint in een willekeurige state met een willekeurige eerste action | |
        | **ε-soft policy** | epsilon-soft policy | een policy die elke action een kans geeft van minstens $\varepsilon$ gedeeld door het aantal actions | `soft` |
        | **bruikbare aas** | | een aas die als 11 telt zonder dat de hand boven 21 komt; in het Engels *usable ace* | `s[2]` |
        | **hit**, **stick** | | een kaart nemen; stoppen en de dealer laten spelen | `"hit"`, `"stick"` |
        | $n$ | n | hoeveel returns van $(s, a)$ er tot nu toe waren | `count[(s, a)]` |
        """),
        L.watch_out("De blackjack van een casino heeft meer regels, zoals verdubbelen, splitsen en een uitbetaling van 3 tegen 2 voor blackjack, een aas en een 10 als je eerste twee kaarten. De beste policy van deze les is de beste policy voor de regels van de course, niet voor een echte tafel."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Monte Carlo control met exploring starts op de blackjack van de course, 500.000 handen. De agent begint met de policy *een kaart nemen onder 20, stoppen op 20 en 21*. De twee roosters laten de policy zien, voor handen zonder en met een bruikbare aas: je totaal van 21 bovenaan tot 12 onderaan, tegen de kaart van de dealer van de aas tot en met 10. Een <span class="c-param">oranje</span> vak met H betekent dat de policy daar een kaart neemt, een ring dat de policy daar van de beste policy verschilt. De grafiek eronder laat zien wat een hand waard is met de policy, uitgerekend door de course met de kansen van de kaarten: de agent zelf ziet deze getallen nooit, alleen de eigen handen.
    """)
    return


@app.cell(hide_code=True)
def explore_hands_cell():
    explore_hands = L.explore("play", "handen", lambda _v: C.play_frame(int(_v), L.lang), start=C.N_HANDS,
                              values=C.LOOKS, role="",
                              task="Ga naar 0, en dan stap voor stap naar 500.000: in hoeveel states verschilt de policy van de beste na 1.000, 100.000 en 500.000 handen? En waarom is een hand na 1.000 handen zoveel meer waard, terwijl de roosters nog bijna even veel ringen hebben als bij de start?")
    explore_hands
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Bij de start verschilt de policy in 70 van de 200 states van de beste, en is een hand −0,354 waard: je verliest ongeveer een derde van elke inzet. Na 1.000 handen verschilt de policy nog in 66 states, maar is een hand −0,118 waard: de agent stopt nu op 18 en 19 in 32 van de 40 states, waar een kaart nemen het meeste kostte, en bij de meeste states die nog fout zijn gaat het om twijfelgevallen, waar de verkeerde action weinig kost, of om states die bij echt gedeelde kaarten zelden voorkomen, zodat deze states weinig meetellen in wat een hand waard is. Tussen 1.000 en 2.000 handen zakt de waarde zelfs een beetje, naar −0,129: een gemiddelde van een paar handen kan de verkeerde kant op wijzen. Na 100.000 handen verschillen er nog 4 states, en na 500.000 handen geen enkele: de policy is de beste policy, −0,047 waard per hand. De waarde klimt eerst snel en dan langzaam, omdat de states die als laatste goed gaan de states zijn waar de twee actions bijna evenveel waard zijn.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Monte Carlo control met exploring starts:

    1. Zet $Q(s, a)$ en het aantal $n$ op 0 voor elke state en action, en neem een policy om mee te beginnen.
    2. Begin een hand in een willekeurige state met een willekeurige eerste action; speel daarna met de policy tot je stopt of boven 21 komt.
    3. De reward aan het einde is de return van elke beslissing in de hand, omdat $\gamma$ = 1.
    4. Voor elke state en action van de hand: tel 1 op bij het aantal $n$, en schuif $Q(s, a)$ met $1/n$ van de afstand op naar de return.
    5. Laat de policy in elke state van de hand de action met de grotere $Q$ kiezen, en stoppen bij gelijke $Q$. Ga dan voor de volgende hand terug naar stap 2.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Je hebt 16 zonder bruikbare aas, en de dealer laat een 10 zien: een kaart nemen of stoppen? De cel schat $Q$ van allebei, stap 2 tot en met 4 van het recept voor deze ene state: de cel speelt 100.000 handen waarin je stopt op 16, en 100.000 waarin je één kaart neemt en dan stopt, en laat de gemiddelde reward zien na 100, 1.000, 10.000 en 100.000 handen. `_card()` trekt een kaart, met 11, 12 en 13 als boer, vrouw en heer, elk 10 waard, en `_dealer(up)` speelt de hand van de dealer vanaf de kaart `up` en geeft terug wat deze hand aan het einde telt. Voer de cel uit.
    """)
    return


@app.cell
def example_sixteen():
    import numpy as np

    _rng = np.random.default_rng(0)


    def _card():
        # 1 is een aas; 11, 12 en 13 zijn boer, vrouw en heer, elk 10 waard
        return min(int(_rng.integers(1, 14)), 10)


    def _dealer(up):
        # de dichte kaart, en dan kaarten tot de hand 17 of meer telt; een aas telt 11 zolang het totaal dan 21 of minder blijft
        cards = [up, _card()]
        while True:
            total = sum(cards)
            if 1 in cards and total + 10 <= 21:
                total = total + 10
            if total >= 17:
                return total
            cards.append(_card())


    def _reward(player, dealer):
        if player > 21:
            return -1
        if dealer > 21 or player > dealer:
            return 1
        if player == dealer:
            return 0
        return -1


    _stick = []
    _hit = []
    for _i in range(100000):
        _stick.append(_reward(16, _dealer(10)))          # stoppen op 16
        _hit.append(_reward(16 + _card(), _dealer(10)))  # één kaart, dan stoppen: een aas telt hier 1, want 11 komt boven 21
        if _i + 1 in [100, 1000, 10000, 100000]:
            print("handen:", _i + 1, "stoppen:", round(sum(_stick) / len(_stick), 3), "kaart:", round(sum(_hit) / len(_hit), 3))
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Na 100 handen lijkt stoppen beter, −0,54 tegen −0,58; na 1.000 lijkt een kaart nemen beter, −0,565 tegen −0,574; na 100.000 staat stoppen op −0,574 en een kaart nemen op −0,573. Uitgerekend met de kansen is stoppen −0,576 waard en één kaart nemen −0,569: een kaart nemen is 0,0065 beter, terwijl het verschil tussen twee gemiddelden van 100.000 handen nog een spreiding van ongeveer 0,004 heeft. Beide keuzes verliezen meer dan een halve inzet: 16 tegen een 10 is een slechte hand, wat je ook doet. Een twijfelgeval als dit kost weinig als je het fout doet, en het kost veel handen om de goede keuze te vinden.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_update():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Leren van één hand

        `update(Q, count, policy, hand, reward)` doet stap 4 en 5 van het recept na één hand. `Q` en `count` zijn dictionaries met een key `(s, a)` voor elk van de 400 paren van state en action, en `policy` is een dictionary van elke state naar de action, `"stick"` of `"hit"`. Een state is een tuple `(totaal, kaart van de dealer, bruikbare aas)`, zoals `(13, 2, False)`. `hand` is de lijst met de beslissingen van de hand, elk een paar `(s, a)`, en `reward` is de reward aan het einde: de return van elke beslissing. De functie verandert `Q`, `count` en `policy` en geeft niets terug: verandert een functie een dictionary, dan is de dictionary ook veranderd voor de code die de functie aanriep. Vervang de twee `...`: de nieuwe `Q[(s, a)]`, stap 4, en wanneer de policy een kaart neemt, stap 5; bij gelijke `Q` stopt de policy.

        De check begint met `Q` en `count` op 0 en de startpolicy, roept `update` aan voor elke hand van `HANDS`, en kijkt naar `Q` en `policy` na de laatste hand. `HANDS` bevat vijf handen tegen een 2 van de dealer: een kaart op 13 en stoppen op 19, gewonnen; een kaart op 13 en boven 21; een kaart op 13 en stoppen op 17, gelijkspel; stoppen op 19, gelijkspel; stoppen op 13, verloren. Daarna vier andere gevallen: `ACE_HAND`, één hand tegen een 6, een kaart op 15 met een bruikbare aas, een 10 maakt er 15 zonder bruikbare aas van, nog een kaart tot 21 en stoppen, gewonnen; `HANDS + ACE_HAND`; `FIRST_HANDS`, de eerste 1.000 handen van het speelblok; en `[]`, helemaal geen handen.
        """),
        L.task("Laat `update` elke `Q[(s, a)]` het gemiddelde van de returns houden, en laat de policy de action met de grotere `Q` kiezen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_update():
    box_update = L.code_box("update", lines=10)
    box_update
    return (box_update,)


@app.cell(hide_code=True)
def check_update(box_update):
    result_update = L.check("update", box_update.value)
    result_update
    return (result_update,)


@app.cell(hide_code=True)
def hints_update():
    L.hints(
        "update",
        [
            "Stap 4 van het recept en de formule van *Het idee*: `Q[(s, a)]` schuift 1/`count[(s, a)]` van de afstand op naar `reward`, en `count[(s, a)]` is voor deze hand al met 1 opgehoogd. Stap 5: welke van `Q[(s, \"hit\")]` en `Q[(s, \"stick\")]` moet groter zijn om de policy een kaart te laten nemen, en wat doet de policy bij gelijke `Q`?",
            "In `HANDS` krijgt een kaart nemen op 13 de returns 1, −1 en 0: de `Q` is 1 na de eerste, 1 + (−1 − 1)/2 = 0 na de tweede, en 0 + (0 − 0)/3 = 0 na de derde. Stoppen op 19 krijgt 1 en dan 0: de `Q` is 1, en dan 1 + (0 − 1)/2 = 0,5.",
            "`        Q[(s, a)] = ___`",
        ],
        why="`Q[(s, a)] = Q[(s, a)] + (reward - Q[(s, a)]) / count[(s, a)]` houdt `Q` het gemiddelde van alle returns na `(s, a)`, en `if Q[(s, \"hit\")] > Q[(s, \"stick\")]:` laat de policy alleen een kaart nemen waar een kaart nemen gemiddeld meer opleverde.",
    )
    return


@app.cell(hide_code=True)
def text_after_update():
    mo.md(r"""
    Met de `update` van de course eindigen de 500.000 handen van het speelblok op de beste policy. In de afbeelding speelt je `update` de eerste 100.000 handen, dezelfde handen als in het speelblok zolang je functie dezelfde actions kiest; met de `update` van de course verschillen er daarna nog 4 states.

    ### De hele policy in één keer met NumPy

    De dictionaries van `update` zijn handig voor één hand tegelijk; voor alle 200 states tegelijk is een array korter. `C.play_q()` geeft $Q$ na de 500.000 handen van het speelblok als een array met vorm `(2, 10, 10, 2)`: bruikbare aas of niet, je totaal van 12 tot en met 21, de kaart van de dealer van de aas tot en met 10, en de action, 0 voor stoppen en 1 voor een kaart nemen. `np.argmax(_q, axis=3)` geeft voor elke state tegelijk de plek van het grootste getal langs de laatste as: de action met de grotere $Q$, en bij gelijke $Q$ de eerste, stoppen. `C.best_array()` is de beste policy in de vorm van `_policy`, `(2, 10, 10)`; `_policy != _best` is `True` waar de twee verschillen, en `.sum()` telt de keren `True`. Voer de cel uit.
    """)
    return


@app.cell
def example_numpy(np):
    _q = C.play_q()                      # Q na het speelblok, vorm (2, 10, 10, 2)
    _policy = np.argmax(_q, axis=3)      # 0 stoppen, 1 een kaart, voor alle 200 states tegelijk
    _best = C.best_array()
    print("vorm van Q:", _q.shape)
    print("states met een andere action dan de beste policy:", int((_policy != _best).sum()))
    print("zonder bruikbare aas, totaal 12 bovenaan, de aas van de dealer links:")
    print(_policy[0])
    return


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    Geen enkele state verschilt. Lees de geprinte array van boven, totaal 12, tegen de kaart van de dealer van de aas tot en met 10: op 12 een kaart nemen tegen een aas, een 2 en een 3, stoppen tegen een 4 tot en met 6, en een kaart nemen tegen een 7 of hoger; op 13 tot en met 16 stoppen tegen een 2 tot en met 6 en een kaart nemen tegen de rest; vanaf 17 altijd stoppen. De 2 tot en met 6 van de dealer zijn de zwakke kaarten: daarmee komt de dealer het vaakst boven 21, dus stop je vroeg en laat je het risico aan de dealer.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_soft():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een ε-soft policy

        `soft(policy, s, epsilon)` geeft de kansen op de twee actions in state `s` terug onder ε-greedy, een ε-soft policy, als een dictionary `chances` met een kans voor elke action `a`: `{"stick": ..., "hit": ...}`. Een willekeurige zet komt met kans `epsilon`, en kiest elke action met kans 1/2. De kans op de andere action staat er al: `epsilon / 2`. Vervang de `...` door de kans op de action van de policy zelf, `policy[s]`. De check roept `soft(START, (20, 10, False), 0.1)` aan en vijf andere gevallen; `START` is de startpolicy, die op 20 stopt.
        """),
        L.task("Laat `soft` de action van de policy zelf de kans geven die deze action onder ε-greedy heeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_soft():
    box_soft = L.code_box("soft", lines=8)
    box_soft
    return (box_soft,)


@app.cell(hide_code=True)
def check_soft(box_soft):
    result_soft = L.check("soft", box_soft.value)
    result_soft
    return (result_soft,)


@app.cell(hide_code=True)
def hints_soft():
    L.hints(
        "soft",
        [
            "*Het idee*: met kans `epsilon` is de zet willekeurig, en een willekeurige zet kiest de helft van de keren de action van de policy zelf; elke andere zet neemt `policy[s]`. De twee kansen in `chances` zijn samen 1, en `chances[a]` is de kans op de action van de policy.",
            "Met ε = 0,1 volgen 9 van de 10 zetten de policy, en 1 van de 10 is willekeurig, waarvan de helft weer de action van de policy kiest: 0,9 + 0,05 = 0,95.",
            "`            chances[a] = ___`",
        ],
        why="`chances[a] = 1 - epsilon + epsilon / 2`: de zetten die niet willekeurig zijn, $1 - \\varepsilon$, plus de helft van de willekeurige zetten, $\\varepsilon/2$. Korter geschreven: `1 - epsilon / 2`.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Leren van echt gedeelde kaarten

    De demo speelt 500.000 handen zonder exploring starts: elke hand begint met echt gedeelde kaarten, en elke beslissing volgt je `soft` met $\varepsilon$ = 0,1. Na elke hand werkt je `update` $Q$ en de policy bij. Waar je functie de check nog niet haalt, neemt de functie van de course de plek in. De grafiek laat zien wat een hand waard is met de policy, doorgetrokken, en zoals de agent speelt, met de willekeurige zetten, gestreept; ernaast de roosters van de policy zoals deze nu is.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    play_button = mo.ui.run_button(label="Speel 500.000 handen met mijn update en soft")
    play_button
    return (play_button,)


@app.cell(hide_code=True)
def demo_play(box_soft, box_update, play_button, result_soft, result_update):
    played = None
    if not play_button.value:
        _out = mo.md("*Druk op de knop om een agent te laten leren van echt gedeelde kaarten.*")
    else:
        _mine = [_n for _n, _r in (("update", result_update), ("soft", result_soft)) if _r.state == "correct"]
        _update = run_code(box_update.value).names["update"] if "update" in _mine else C.update
        _soft = run_code(box_soft.value).names["soft"] if "soft" in _mine else C.soft
        _w = C.DEMO_TEXT[L.lang]
        _now = {}

        def _progress(_n, _run):
            if "policy" not in _now:
                _now["chart"].watch(lambda: C.now_picture(_now["n"], _now["policy"], L.lang))
            _now["n"], _now["policy"] = _n, _run["policies"][_n]
            _now["chart"].reference([(_k, _run["played"][_k]) for _k in sorted(_run["played"]) if _k > 0], "", role="ink", replace=True)
            _now["chart"].extend([(_n, _run["worth"][_n])], f"**Hand {C.number(_n, 0, 'nl')} van {C.number(C.N_HANDS, 0, 'nl')}**")

        _live = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"], "alt": _w["alt"]}, C.N_HANDS, role="ink",
                       low=-0.4, high=0.0, window=1, xlog=True,
                       key=[("ink", False, _w["key_policy"]), ("ink", True, _w["key_played"]), ("ink-3", True, _w["best"])])
        _live.line("**Spelen** …")
        _live.reference([(100, C.BEST_VALUE), (C.N_HANDS, C.BEST_VALUE)], "", role="ink-3")
        _now["chart"] = _live
        _r = C.soft_run(_update, _soft, show=_progress)
        if _r == "update":
            _out = _live.stopped("*Je `update` gaf een fout tijdens de run, of liet een state achter zonder `\"stick\"` of `\"hit\"`. Haalt je functie de check nog?*")
        elif _r == "soft":
            _out = _live.stopped("*Je `soft` gaf een fout tijdens de run, of kansen die samen niet 1 zijn. Haalt je functie de check nog?*")
        elif _r == "slow":
            _out = _live.stopped("*De run duurde langer dan twee minuten. Doet je functie meer werk dan nodig is?*")
        else:
            played = _r
            _theirs = [_n for _n in ("update", "soft") if _n not in _mine]
            if len(_mine) == 2:
                _who = "je `update` en je `soft`"
            elif len(_mine) == 1:
                _who = f"je `{_mine[0]}` en de `{_theirs[0]}` van de course"
            else:
                _who = "de `update` en `soft` van de course, omdat de jouwe de checks nog niet halen"
            _end = _r["policies"][C.N_HANDS]
            _out = mo.vstack([
                L.explore("rewind", "handen", lambda _v: C.rewind_frame(_r, int(_v), L.lang), start=C.N_HANDS,
                          values=C.LOOKS, role="",
                          task="Ga terug naar 0, en dan weer vooruit: hoe ver liggen de doorgetrokken en de gestreepte lijn aan het einde uit elkaar, en welke states hebben nog een ring?"),
                mo.md(f"Gespeeld met {_who}. Na 500.000 handen verschilt de policy in {len(C.differ(_end))} van de 200 states van de beste policy; een hand is er {C.number(_r['worth'][C.N_HANDS], 3, 'nl')} waard mee, en {C.number(_r['played'][C.N_HANDS], 3, 'nl')} zoals gespeeld, met de willekeurige zetten."),
            ])
    _out
    return (played,)


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met de functies van de course verschilt de policy na 500.000 handen in 9 van de 200 states van de beste policy, en is een hand er −0,048 waard mee; zoals gespeeld, met de willekeurige zetten, −0,079. De willekeurige zetten kosten ongeveer 0,03 per hand: de prijs van steeds beide actions proberen. Van de 9 states hebben er 6 een bruikbare aas: bij echt gedeelde kaarten heb je in ongeveer 15 van de 100 beslissingen een bruikbare aas, en in sommige van deze states kom je bijna nooit, terwijl exploring starts in elke state even vaak een hand begonnen. Twee van de andere drie zijn states waar de beste ε-soft policy stopt: 12 tegen een 3 en 16 tegen een 10. Als één zet op de twintig de andere kant op gaat, levert een hand die doorgaat minder op, dus is stoppen daar beter. Monte Carlo control met ε-greedy leert de beste manier van spelen met willekeurige zetten, niet zonder.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_casino():
    mo.md(r"""
    ### Oefening 3. Exploring starts aan een echte tafel

    In het speelblok begon elke hand in een state die de course willekeurig koos, met een willekeurige eerste action. Aan een echte tafel, met echt gedeelde kaarten, kan een agent dat niet.
    """)
    return


@app.cell(hide_code=True)
def choice_casino_cell():
    choice_casino = L.choice(
        "casino",
        question="Waarom kan een agent die aan een echte tafel blackjack speelt geen exploring starts gebruiken?",
        options={
            "rules": "Een echte tafel heeft andere regels, dus de states van de course bestaan daar niet",
            "slow": "Een echte tafel deelt te langzaam voor de 500.000 handen die exploring starts nodig hebben, terwijl een ε-soft policy er veel minder nodig heeft",
            "right": "De kaarten bepalen waar een hand begint: je kunt niet kiezen om op 13 tegen een 2 te beginnen",
            "hidden": "Aan een echte tafel zie je de dichte kaart van de dealer niet, en exploring starts hebben deze kaart nodig",
        },
        correct="right",
        why={
            "right": "Exploring starts vragen een spel dat je overal kunt beginnen, zoals de eigen blackjack van de course. Aan een echte tafel bepalen de gedeelde kaarten de state waarin een hand begint. Daar houdt een ε-soft policy beide actions in het spel, met echt gedeelde kaarten, zoals in de demo, ten koste van de willekeurige zetten.",
            "rules": "Een state is wat je weet als je beslist: je totaal, de kaart van de dealer en of je een bruikbare aas hebt. Aan een echte tafel heb je deze dingen ook; wat kan een speler daar niet kiezen?",
            "slow": "De demo had met een ε-soft policy ook 500.000 handen nodig, allemaal met echt gedeelde kaarten: is het aantal handen het probleem, of waar elke hand begint?",
            "hidden": "De states van de course bevatten de dichte kaart ook niet: de agent beslist ook in het speelblok zonder deze kaart. Wat kiest een exploring start dat je bij echt gedeelde kaarten niet kunt kiezen?",
        },
    )
    choice_casino
    return (choice_casino,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Leren van hele potjes: Monte Carlo prediction](/course/open/rl-monte-carlo-prediction?lang=nl) schatte je $V$ van een vaste policy met het gemiddelde van returns. Hier schatten dezelfde gemiddelden $Q(s, a)$, en na elke hand kiest de policy de action met de grotere $Q$: het afwisselen van evalueren en verbeteren uit [Steeds beter: policy iteration](/course/open/rl-policy-iteration?lang=nl), zonder de tabel `T`. Exploring starts en ε-soft policies houden beide actions in het spel, zoals ε-greedy deed in [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl). De volgende les, *Leren na elke stap: TD(0)*, werkt waarden na elke stap bij in plaats van te wachten tot het einde van de episode. In het project aan het einde van dit deel leert Monte Carlo control in een grid world die je zelf ontwerpt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_casino, result_soft, result_update):
    L.wrap_up(result_update, result_soft, choice_casino)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
