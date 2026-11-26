# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-policy-evaluation. Route: course/route/lessen/reinforcement-learning.yaml. Checks, de kamer, de eigen policy evaluation van de course en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rl-policy-evaluation", "nl")
    C = L.checks   # de kamer en de tabel ervan, de eigen policy evaluation van de course, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Als stappen kunnen uitglijden: value iteration](/course/open/rl-value-iteration?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_q():
    quick_q = L.choice(
        "quick-q",
        question="Een action heeft twee uitkomsten, allebei met reward 0: met kans 0,8 kom je in een state met waarde 1,0, met kans 0,2 in een state met waarde 0,5. Wat is $Q(s, a)$ bij $\\gamma$ = 0,9?",
        options={
            "added": "1,35: de twee waarden opgeteld en dan keer γ",
            "right": "0,81: γ keer de verwachtingswaarde 0,9",
            "no_gamma": "0,9: elke waarde keer de kans erop",
            "likely": "1,0: de waarde van de uitkomst met de grootste kans",
        },
        correct="right",
        why={
            "right": "Elke uitkomst telt mee met de kans erop: 0,8 · (0 + 0,9 · 1,0) + 0,2 · (0 + 0,9 · 0,5) = 0,72 + 0,09 = 0,81.",
            "added": "De twee uitkomsten kunnen niet allebei tegelijk gebeuren: hoeveel telt elke uitkomst mee als de kans erop 0,8 of 0,2 is? Zie [Als stappen kunnen uitglijden: value iteration](/course/open/rl-value-iteration?lang=nl).",
            "no_gamma": "0,8 · 1,0 + 0,2 · 0,5 = 0,9 is de verwachtingswaarde van de volgende state, een stap later: wat doet $\\gamma$ daarmee? Zie [Als stappen kunnen uitglijden: value iteration](/course/open/rl-value-iteration?lang=nl).",
            "likely": "De uitkomst met de kleinere kans kan ook gebeuren, één keer op de vijf: hoeveel telt deze uitkomst mee in een verwachtingswaarde? Zie [Als stappen kunnen uitglijden: value iteration](/course/open/rl-value-iteration?lang=nl).",
        },
    )
    quick_q
    return (quick_q,)


@app.cell(hide_code=True)
def quick_check_max():
    quick_max = L.choice(
        "quick-max",
        question="Hoe geeft een ronde van value iteration een state $s$ een nieuwe waarde $V(s)$?",
        options={
            "mean": "Met het gemiddelde van de vier $Q(s, a)$, één voor elke action",
            "right": "Met de grootste $Q(s, a)$: de waarde van de beste action in $s$",
            "last": "Met de $Q$ van de action die de agent daar de laatste keer nam",
            "all": "Met alle $Q(s, a)$ opgeteld, omdat elke action genomen kan worden",
        },
        correct="right",
        why={
            "right": "Value iteration zoekt de beste policy, en de beste policy neemt in elke state de beste action, dus is $V(s)$ de grootste $Q(s, a)$. In deze les is de policy gegeven, en maakt de grootste $Q$ plaats voor iets anders.",
            "mean": "Value iteration vindt de waarden van de beste policy: neemt de beste policy elke action even vaak? Zie [Als stappen kunnen uitglijden: value iteration](/course/open/rl-value-iteration?lang=nl).",
            "last": "Value iteration werkt met de tabel `T`, en tijdens value iteration neemt geen agent actions: welke $Q$ gebruikt de beste policy? Zie [Als stappen kunnen uitglijden: value iteration](/course/open/rl-value-iteration?lang=nl).",
            "all": "In een state neemt de agent één action, niet alle vier: welke neemt de beste policy? Zie [Als stappen kunnen uitglijden: value iteration](/course/open/rl-value-iteration?lang=nl).",
        },
    )
    quick_max
    return (quick_max,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Als stappen kunnen uitglijden: value iteration](/course/open/rl-value-iteration?lang=nl) zeiden de waarden wat een state waard is als de agent zo goed mogelijk handelt. Vaak is de vraag een andere: wat is een state waard met de policy van de agent van nu? Een robotstofzuiger heeft een vaste regel voor de volgende tegel, een spel een strategie van iemand anders, een agent de policy van vorige week. Voor je zo'n policy verandert, wil je weten wat de policy vanuit elke state waard is, en of een andere policy beter zou doen. Policy evaluation geeft dat antwoord, en de volgende les bouwt erop voort om een policy steeds beter te maken.

    **Een voorbeeld uit het dagelijks leven.** Hoeveel beurten heeft een pion bij een bordspel met een dobbelsteen gemiddeld nodig om van een bepaald vakje bij de finish te komen? Je hoeft geen duizend potjes te spelen om dat te weten. Werk terug vanaf de finish: het getal van een vakje is één beurt plus het gemiddelde van de getallen van de zes vakjes waar een worp de pion kan brengen.
    """)
    return


@app.cell(hide_code=True)
def predict_steps():
    guess_steps = L.guess(
        "steps",
        question="Een robot staat op een grid van 5 bij 5 tegels, met het laadstation op de middelste tegel. Een plan is er niet: bij elke stap gaat de robot omhoog, omlaag, naar links of naar rechts, elk met kans 1/4, en na een stap tegen de rand blijft de robot op dezelfde tegel. De robot begint op een tegel naast het laadstation. Hoeveel stappen zijn er gemiddeld nodig tot het laadstation?",
        options={
            "four": "Ongeveer 4: één stap op de vier gaat naar het laadstation",
            "two": "Ongeveer 2",
            "right": "24",
            "ten": "Ongeveer 10, want op zo'n klein grid komt de robot nooit ver van het laadstation",
        },
        reveal="**24.** Vanaf een tegel naast het laadstation heeft de robot gemiddeld 24 stappen nodig; vanuit een hoek, langs de kortste weg 4 stappen van het laadstation, zijn het er 36. Eén stap op de vier gaat naar het laadstation, maar de andere drie brengen de robot er verder vandaan, en vanaf daar duurt het langer. *Het idee* laat zien hoe je deze getallen uitrekent zonder dat de robot een stap zet.",
    )
    guess_steps
    return (guess_steps,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De regel van de robot is een **policy**, zoals in [Leren van rewards](/course/open/rl-what-is-rl?lang=nl). Een policy kan één action per state geven, zoals de beste policy van value iteration deed, of een kans voor elke action: $\pi(a \mid s)$ is de kans dat de policy in state $s$ action $a$ neemt. De regel van de robot is de **random policy**: elke action met kans 1/4.

        Geef elke stap een reward van −1, en laat de taak eindigen op het laadstation, met de waarde 0. Met $\gamma = 1$ is de return van een wandeling min het aantal stappen, dus is de waarde van een tegel min het gemiddelde aantal stappen van daar tot het laadstation. Dat is de value function van deze policy, $V_\pi(s)$: de return die je vanuit $s$ mag verwachten als je $\pi$ volgt. De **Bellman equation van een policy** verbindt de waarden van buren, zoals de Bellman equation van value iteration dat deed:
        """),
        L.formula(r"[[pred:V_\pi(s)]] = \sum_a \pi(a \mid s) \sum [[data:p]] \left( [[data:r]] + [[hyper:\gamma]] \, [[pred:V_\pi(s')]] \right)"),
        mo.md(r"""
        Het binnenste deel is $Q(s, a)$, zoals bij value iteration: elke uitkomst van $a$ met de kans $p$ erop, de reward $r$ en de waarde van de state $s'$ waar de uitkomst landt. Waar value iteration de grootste $Q(s, a)$ nam, neemt deze policy elke action met een eigen kans, dus is de waarde het gemiddelde van de $Q$-waarden, elk meegeteld met $\pi(a \mid s)$. Op een tegel naast het laadstation komt één stap op de vier op het laadstation, waard −1 + 0, en komen drie op de vier op een tegel waarvandaan de robot er langer over doet.

        **Policy evaluation** vindt $V_\pi$ zoals value iteration de beste waarden vond: begin overal met $V = 0$, en gebruik de vergelijking als update voor elke state, met de oude waarden. Eén zo'n ronde langs alle states heet een **sweep**; de rondes van value iteration waren ook sweeps. Na sweep 1 staat elke tegel op −1: één stap, geteld. Na sweep 2 staat een tegel naast het laadstation op ¼ · (−1 + 0) + ¾ · (−1 − 1) = −1,75, elke andere tegel op −2. Na $k$ sweeps telt een waarde de eerste $k$ stappen van de wandelingen vanaf de tegel; langere wandelingen komen er in latere sweeps bij. Zodra de grootste verandering in een sweep kleiner is dan een klein getal $\theta$, veranderen de waarden bijna niet meer, en stopt policy evaluation.

        Dit werkt met $\gamma = 1$ omdat de random robot vanaf elke tegel vroeg of laat bij het laadstation komt. Value iteration en policy evaluation horen bij **dynamic programming**: methoden waarmee je waarden uitrekent uit een bekende tabel, door de Bellman equation steeds opnieuw op elke state toe te passen.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **policy evaluation** | | de waarde van elke state onder een gegeven policy uitrekenen, met sweeps van de Bellman equation van de policy | `evaluate` |
        | $\pi(a \mid s)$ | pi van a gegeven s | de kans dat de policy in state $s$ action $a$ neemt | `pi[s][a]` |
        | **random policy** | | een policy die elke action met dezelfde kans neemt | `RANDOM` |
        | $V_\pi(s)$ | V pi van s | de waarde van state $s$ als je policy $\pi$ volgt | `V[s]` |
        | **Bellman equation van een policy** | | de waarde van een state als gemiddelde van de $Q$-waarden, elk meegeteld met de kans op de action | |
        | **sweep** | | één update van elke state, met de oude waarden | `sweep` |
        | $\theta$ | theta | policy evaluation stopt zodra de grootste verandering in een sweep kleiner is dan $\theta$ | `theta` |
        | **dynamic programming** | | methoden waarmee je waarden uitrekent uit een bekende tabel, door de Bellman equation steeds opnieuw op elke state toe te passen | |
        """),
        L.watch_out("In robotica en daarbuiten betekent *een policy evalueren* meestal: meten hoe vaak de policy slaagt in tests. Hier betekent policy evaluation: de waarden uitrekenen uit de tabel."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Policy evaluation van de random policy op ROOM, het grid van de robot, sweep voor sweep. Een getal is de <span class="c-pred">waarde</span> van de tegel na de sweep; het laadstation in het midden heeft de waarde 0. De grafiek ernaast toont na elke sweep de waarden van twee tegels: een tegel naast het laadstation en een hoek.
    """)
    return


@app.cell(hide_code=True)
def explore_sweeps_cell():
    explore_sweeps = L.explore("sweeps", "sweep", lambda _v: C.play_frame(int(_v), L.lang), start=C.PLAY_SWEEPS,
                               values=C.LOOKS, role="",
                               task="Ga naar sweep 3: waarom staat de hoek op precies −3, en de tegel naast het laadstation op −2,50? Ga dan door naar sweep 100: hoe ver zit de hoek nog van −36 af?")
    explore_sweeps
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Na 3 sweeps telt een waarde de eerste 3 stappen van elke wandeling. Vanuit de hoek ligt het laadstation op 4 stappen, dus is nog geen enkele wandeling vanaf daar afgelopen: drie stappen van −1 maken −3. Vanaf de tegel naast het laadstation eindigt een kwart van de wandelingen met de eerste stap, dus worden er gemiddeld minder stappen geteld: −2,50. Sweep na sweep zakken de waarden naar −24 naast het laadstation en −36 in een hoek, met −30 schuin naast het laadstation en −32 en −34 langs de randen. Na 100 sweeps staat de hoek op −34,45, nog 1,55 van −36 af: elke sweep haalt slechts ongeveer 3% van de rest eraf, dus het laatste stuk kost veel sweeps.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Policy evaluation van een policy $\pi$, met een tabel `T`:

    1. Begin met $V(s) = 0$ voor elke state.
    2. Sweep: reken voor elke state $s$ en elke action $a$ de $Q(s, a)$ uit met de oude waarden, zoals bij value iteration.
    3. De nieuwe waarde van $s$ is het gemiddelde van deze $Q$-waarden, elk meegeteld met $\pi(a \mid s)$. Houd de nieuwe waarden apart van de oude tot de sweep klaar is.
    4. Houd bij hoe ver elke waarde in deze sweep verschoof, en onthoud de grootste verandering.
    5. Vervang de oude waarden door de nieuwe. Is de grootste verandering $\theta$ of meer, ga dan verder bij stap 2.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een gang van vier tegels: het laadstation op tegel 0 aan de linkerkant, en een muur na tegel 3. De robot stapt naar links of naar rechts, elk met kans 1/2; tegen de muur blijft de robot op tegel 3. Elke stap kost −1, en $\gamma$ = 1. De cel houdt de vier waarden bij in een lijst en doet 200 sweeps, veel meer dan deze waarden nodig hebben; met een vast aantal sweeps slaat de cel stap 4 van het recept over, en oefening 2 voegt stap 4 toe. De cel toont de waarden van tegel 1, 2 en 3 na de eerste drie sweeps en na de laatste. Voer de cel uit.
    """)
    return


@app.cell
def example_corridor():
    # 1. de waarden van de vier tegels; tegel 0 is het laadstation en houdt 0
    _V = [0, 0, 0, 0]
    for _k in range(200):
        # 2. en 3. één sweep: elke nieuwe waarde uit de oude waarden, elke stap met kans 1/2
        _new = [0, 0, 0, 0]
        for _s in [1, 2, 3]:
            _left = _s - 1
            _right = min(_s + 1, 3)   # tegen de muur blijft de robot op tegel 3
            _new[_s] = 0.5 * (-1 + _V[_left]) + 0.5 * (-1 + _V[_right])
        # 5. de nieuwe waarden vervangen de oude
        _V = _new
        if _k < 3 or _k == 199:
            print("na sweep", _k + 1, ":", round(_V[1], 2), round(_V[2], 2), round(_V[3], 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Sweep 1 geeft elke tegel −1: één stap, geteld. In sweep 2 krijgt tegel 1 ½ · (−1 + 0) + ½ · (−1 − 1) = −1,5: de helft van de stappen vanaf tegel 1 komt op het laadstation. Tegel 3 staat na sweep 2 op −2 en na sweep 3 op −3: het laadstation ligt drie tegels verderop, dus binnen drie stappen is geen enkele wandeling vanaf daar afgelopen. Na 200 sweeps liggen de waarden vast op −6, −10 en −12: vanaf het verre eind van de gang heeft de robot gemiddeld 12 stappen nodig, vier keer de 3 stappen van de kortste weg.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_evaluate():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Eén policy, veel sweeps

        `evaluate(T, pi, gamma, sweeps)` doet `sweeps` sweeps vanaf $V = 0$ en geeft de waarden terug: een dictionary met een waarde voor elke state. `T` is een tabel zoals bij value iteration, en `pi[s][a]` is de kans dat de policy in state `s` action `a` neemt. `k` telt de sweeps, en `new` bewaart de nieuwe waarden tot een sweep klaar is. De loop over `p, nxt, r` in `T[s][a]` rekent `q` al uit, de $Q$ van action `a` in state `s`, en `v` verzamelt de nieuwe waarde van `s`. Vervang de `...` door het aandeel van action `a` in de nieuwe waarde, stap 3 van het recept. De check roept `evaluate(T_ROOM, RANDOM, 1.0, 2)` aan, twee sweeps van de random policy op ROOM, en nog vijf andere gevallen, waaronder de gang en een policy die vaker omhooggaat dan de andere kanten op.
        """),
        L.task("Laat `evaluate` het aandeel van elke action optellen bij de nieuwe waarde van een state."),
    ])
    return


@app.cell(hide_code=True)
def exercise_evaluate():
    box_evaluate = L.code_box("evaluate", lines=16)
    box_evaluate
    return (box_evaluate,)


@app.cell(hide_code=True)
def check_evaluate(box_evaluate):
    result_evaluate = L.check("evaluate", box_evaluate.value)
    result_evaluate
    return (result_evaluate,)


@app.cell(hide_code=True)
def hints_evaluate():
    L.hints(
        "evaluate",
        [
            "Stap 3 van het recept: `v` verzamelt de nieuwe waarde van `s`, en elke action `a` voegt een deel toe. `pi[s][a]` zegt hoeveel de policy een action gebruikt.",
            "Neem op ROOM een tegel naast het laadstation na sweep 1: één action heeft een $Q$ van −1 + 0 = −1, de drie andere −1 + (−1) = −2. De random policy telt elke $Q$ mee met 1/4: 0,25 · (−1) + 0,75 · (−2) = −1,75.",
            "`                v = v + ___`",
        ],
        why="`v = v + pi[s][a] * q`: elke action voegt de eigen $Q$ toe, meegeteld met de kans dat de policy de action neemt, dus eindigt `v` op het gemiddelde van de $Q$-waarden. Value iteration nam op deze plek de grootste `q`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_stable():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Stop als de waarden bijna niet meer veranderen

        Hoeveel sweeps zijn genoeg? `until_stable(T, pi, gamma, theta)` doet sweeps tot de grootste verandering van een state in een sweep kleiner is dan `theta`, en geeft de waarden terug en het aantal sweeps. `sweep(T, pi, gamma, V)` doet één sweep, zoals de binnenkant van je `evaluate`, en geeft de nieuwe waarden terug, `new`. `sweeps` telt de sweeps, en `change` begint in elke sweep op 0; met `max` moet `change` uitkomen op de grootste afstand tussen de nieuwe en de oude waarde van één state `s` in deze sweep. Vervang de twee `...`, stap 4 en 5 van het recept. De check roept `until_stable(T_ROOM, RANDOM, 1.0, 0.01)` aan, dat na 151 sweeps hoort te stoppen, en nog vijf andere gevallen, waaronder `T_PRIZE`: de gang van het uitgewerkte voorbeeld, waar de stap naar het laadstation +10 oplevert, zodat waarden ook stijgen.
        """),
        L.task("Laat `until_stable` de grootste verandering van een sweep bijhouden in `change`, en stoppen zodra `change` kleiner is dan `theta`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_stable():
    box_stable = L.code_box("stable", lines=14)
    box_stable
    return (box_stable,)


@app.cell(hide_code=True)
def check_stable(box_stable):
    result_stable = L.check("stable", box_stable.value)
    result_stable
    return (result_stable,)


@app.cell(hide_code=True)
def hints_stable():
    L.hints(
        "stable",
        [
            "Stap 4: `change` begint in elke sweep op 0, en `max` houdt het grootste van twee getallen, dus na de loop over `s` staat de grootste verandering erin. Een verandering telt hoe ver een waarde verschoof, welke kant de waarde ook op ging. Stap 5: de loop mag stoppen zodra `change` klein is.",
            "In de eerste sweep op ROOM gaat elke waarde van 0 naar −1: de waarde verschoof 1, al is `new[s] - V[s]` gelijk aan −1, en `abs(-1)` is 1. Zo eindigt `change` op 1. Na sweep 151 is de grootste verandering voor het eerst kleiner dan 0,01.",
            "`            change = max(change, ___)`",
        ],
        why="`change = max(change, abs(new[s] - V[s]))` houdt de grootste afstand tussen een nieuwe en een oude waarde bij, en `if change < theta:` stopt de loop zodra de afstand klein is.",
    )
    return


@app.cell(hide_code=True)
def text_after_stable():
    mo.md(r"""
    Met $\theta$ = 0,01 stopt de loop na 151 sweeps, terwijl de hoek nog op −35,69 staat, 0,31 van −36 af. Een verandering van 0,01 per sweep betekent niet dat de waarden binnen 0,01 van de eindwaarden liggen: elke sweep haalt ongeveer 3% van de rest eraf, dus er komt nog ongeveer 30 keer de laatste verandering bij. Met $\theta$ = 0,001 eindigt de hoek op −35,97, na 224 sweeps.

    ### Dezelfde sweeps met NumPy

    Voor één vaste policy is een sweep een matrixvermenigvuldiging. Nummer de 25 tegels van ROOM rij voor rij. `P[i, j]` is de kans dat de robot in één stap van de policy van tegel `i` naar tegel `j` gaat, en `R[i]` de reward die de robot op tegel `i` mag verwachten. Dan is een sweep `V = R + gamma * P @ V`: rij `i` van `P @ V` is de gemiddelde waarde van waar tegel `i` heen leidt. De cel bouwt `P` en `R` uit `T_ROOM` en `RANDOM` van de course, doet 224 sweeps en toont de waarden, rij voor rij. Voer de cel uit.
    """)
    return


@app.cell
def example_numpy():
    import numpy as np

    _P = np.zeros((25, 25))
    _R = np.zeros(25)
    _at = {}   # het nummer van elke tegel, rij voor rij
    for _s in C.T_ROOM:
        _at[_s] = len(_at)
    for _s in C.T_ROOM:
        for _a in C.T_ROOM[_s]:
            for _p, _nxt, _r in C.T_ROOM[_s][_a]:
                _P[_at[_s], _at[_nxt]] = _P[_at[_s], _at[_nxt]] + C.RANDOM[_s][_a] * _p
                _R[_at[_s]] = _R[_at[_s]] + C.RANDOM[_s][_a] * _p * _r
    _V = np.zeros(25)
    for _k in range(224):
        _V = _R + 1.0 * _P @ _V
    for _row in range(5):
        print(np.round(_V[5 * _row:5 * _row + 5], 2))
    return


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    Dit zijn de waarden die je `until_stable` geeft met $\theta$ = 0,001: −35,97 in de hoeken, −23,98 naast het laadstation. Een matrix van 25 bij 25 is klein; een tabel met een miljoen states zou er een maken van een miljoen bij een miljoen, bijna helemaal nullen, en dan doet een loop over de states minder werk, zolang de loop voor elke state alleen de stappen neemt die de state echt heeft.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_forever():
    mo.md(r"""
    ### Oefening 3. Een policy die nooit aankomt

    De policy `UP` gaat altijd omhoog. Twee tegels onder het laadstation komen er wel: de tegel direct eronder in één stap, de tegel daaronder in twee. Elke andere tegel gaat omhoog naar de bovenste rij en botst daar steeds tegen de rand.
    """)
    return


@app.cell(hide_code=True)
def choice_forever_cell():
    choice_forever = L.choice(
        "forever",
        question="Wat doet je `until_stable` met `UP` op ROOM, met $\\gamma$ = 1 en $\\theta$ = 0,01?",
        options={
            "zero": "Na één sweep geeft de functie 0 terug voor elke tegel zonder weg naar het laadstation",
            "right": "De functie stopt nooit: elke sweep haalt 1 af van elke tegel zonder aankomst, dus blijft de grootste verandering 1",
            "minus": "De functie stopt na een paar sweeps met −1 voor elke tegel in de bovenste rij, want een stap tegen de rand laat de robot staan en kost slechts één keer",
            "five": "De functie stopt na 5 sweeps, zodra elke tegel de bovenste rij heeft bereikt",
        },
        correct="right",
        why={
            "right": "Met `UP` botst een robot in de bovenste rij bij elke stap tegen de rand en komt nooit bij het laadstation: −1 per stap, zonder eind. Na $k$ sweeps tellen de waarden van deze tegels $k$ stappen, dus haalt elke sweep er 1 af, komt de grootste verandering nooit onder $\\theta$, en komt de loop nooit bij `return`. Alleen de twee tegels onder het laadstation komen tot rust, op −1 en −2. Met $\\gamma$ = 0,9 zouden de waarden tot rust komen op −10, want 1 + 0,9 + 0,81 + … = 10. Met $\\gamma$ = 1 heeft policy evaluation een policy nodig waarmee de robot vanuit elke state een eind bereikt, zoals de random robot doet.",
            "zero": "Een tegel zonder weg naar het laadstation betaalt −1 bij elke stap: blijft de waarde op 0 na de eerste sweep, die één stap telt?",
            "minus": "Een stap tegen de rand kost ook −1, en in de bovenste rij zet de robot elke keer zo'n stap: hoeveel stappen telt sweep $k$?",
            "five": "Eenmaal in de bovenste rij zet de robot nog steeds stappen, en elke stap kost −1: komt de grootste verandering ooit onder $\\theta$?",
        },
    )
    choice_forever
    return (choice_forever,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Als stappen kunnen uitglijden: value iteration](/course/open/rl-value-iteration?lang=nl) hoorden de waarden bij de beste policy, gevonden met de grootste $Q$. Je `evaluate` rekent de waarden uit van elke policy die je de functie geeft, met een gemiddelde over de actions, en je `until_stable` beslist wanneer de sweeps genoeg gedaan hebben. De waarden van de random robot zijn min het gemiddelde aantal stappen tot het laadstation, van −24 ernaast tot −36 in een hoek. Neem je in elke tegel de action met de grootste $Q$ onder deze waarden, dan gaat elke tegel al naar het laadstation: de volgende les, *Steeds beter: policy iteration*, maakt er een methode van om een policy stap voor stap beter te maken. Verderop in dit deel schatten Monte Carlo en TD learning dezelfde $V_\pi$ zonder de tabel `T`, uit wandelingen die de robot echt maakt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_forever, result_evaluate, result_stable):
    L.wrap_up(result_evaluate, result_stable, choice_forever)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
