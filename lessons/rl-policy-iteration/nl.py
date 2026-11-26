# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-policy-iteration. Route: course/route/lessen/reinforcement-learning.yaml. Checks, de kamer en de hal, de policy iteration van de course en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rl-policy-iteration", "nl")
    C = L.checks   # de kamer en de hal met de tabellen, de policy iteration van de course, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Wat is een policy waard? Policy evaluation](/course/open/rl-policy-evaluation?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_v():
    quick_v = L.choice(
        "quick-v",
        question="Op ROOM, het grid van de robot, heeft een tegel naast het laadstation vier buren: het laadstation (waarde 0), een tegel midden aan een rand (−32) en twee tegels diagonaal naast het laadstation (elk −30). Elke stap kost −1, en $\\gamma$ = 1. Wat is de waarde van de tegel onder de random policy?",
        options={
            "best": "−1, want een van de vier stappen gaat meteen naar het laadstation",
            "right": "−24, het gemiddelde van de vier $Q$-waarden",
            "values": "−23, het gemiddelde van de waarden van de buren",
            "added": "−96, de vier $Q$-waarden opgeteld",
        },
        correct="right",
        why={
            "right": "Elke stap kost −1, dus zijn de vier $Q$-waarden −1 + 0, −1 − 32, −1 − 30 en −1 − 30. De random policy telt elke $Q$-waarde mee met 1/4: (−1 − 33 − 31 − 31) / 4 = −24.",
            "best": "−1 is de $Q$ van de stap naar het laadstation, maar de random policy kiest deze stap slechts één keer op de vier: hoe telt de policy de andere drie mee? Zie [Wat is een policy waard? Policy evaluation](/course/open/rl-policy-evaluation?lang=nl).",
            "values": "De buren liggen een stap verderop, en een stap kost iets: wat komt er bij elke waarde bij voor je het gemiddelde neemt? Zie [Wat is een policy waard? Policy evaluation](/course/open/rl-policy-evaluation?lang=nl).",
            "added": "De robot zet één van de vier stappen, niet alle vier tegelijk: hoeveel telt elke stap mee bij de random policy? Zie [Wat is een policy waard? Policy evaluation](/course/open/rl-policy-evaluation?lang=nl).",
        },
    )
    quick_v
    return (quick_v,)


@app.cell(hide_code=True)
def quick_check_theta():
    quick_theta = L.choice(
        "quick-theta",
        question="Policy evaluation van de random policy op ROOM stopt na 151 sweeps met $\\theta$ = 0,01. Wat zegt deze stop je?",
        options={
            "within": "Elke waarde ligt nu binnen 0,01 van de eindwaarde, die je pas na veel meer sweeps zou bereiken",
            "right": "In de laatste sweep bewoog geen waarde 0,01 of meer, maar het eind kan verder weg liggen",
            "final": "Meer sweeps zouden geen enkele waarde veranderen",
            "exact": "Elke waarde veranderde in de laatste sweep precies 0,01",
        },
        correct="right",
        why={
            "right": "De stop kijkt alleen naar de laatste verandering. Op ROOM haalt elke sweep ongeveer 3% van de rest eraf, dus staat de hoek na 151 sweeps nog 0,31 van −36 af, ongeveer 30 keer de laatste verandering.",
            "within": "De stop vergelijkt twee sweeps, niet een sweep met het eind: als elke sweep slechts ongeveer 3% van de rest eraf haalt, hoeveel is er dan nog over zodra een sweep een waarde 0,01 verschuift? Zie [Wat is een policy waard? Policy evaluation](/course/open/rl-policy-evaluation?lang=nl).",
            "final": "Een verandering onder 0,01 is nog steeds een verandering: verschuift de volgende sweep de waarden helemaal niet, of slechts een beetje? Zie [Wat is een policy waard? Policy evaluation](/course/open/rl-policy-evaluation?lang=nl).",
            "exact": "$\\theta$ is een grens voor de grootste verandering, niet de grootte van elke verandering: wat zegt `change < theta` over de grootste verandering van de laatste sweep? Zie [Wat is een policy waard? Policy evaluation](/course/open/rl-policy-evaluation?lang=nl).",
        },
    )
    quick_theta
    return (quick_theta,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Wat is een policy waard? Policy evaluation](/course/open/rl-policy-evaluation?lang=nl) rekende je uit wat een policy vanuit elke state waard is. Dat is de helft van het werk: een agent wil een goede policy, en niet alleen weten hoe goed de policy is. Deze les maakt van de waarden van een policy een betere policy, en door dat steeds opnieuw te doen vind je na een paar rondes een beste policy. Hetzelfde heen en weer tussen waarden en een policy komt terug in bijna elke methode van reinforcement learning, ook als de agent geen tabel heeft en leert uit eigen ervaring.

    **Een voorbeeld uit het dagelijks leven.** Je fietst langs een vaste route naar je werk, en je weet hoe lang de rest van de rit duurt vanaf elk kruispunt onderweg. Bij elk kruispunt kun je je afvragen: als ik hier een andere kant op ga en daarna fiets zoals altijd, ben ik dan eerder? Waar het antwoord ja is, pas je de route aan. Daarna meet je de nieuwe route, en stel je dezelfde vraag opnieuw, tot geen enkel kruispunt een snellere afslag biedt.
    """)
    return


@app.cell(hide_code=True)
def predict_steps():
    guess_steps = L.guess(
        "steps",
        question="Onder de random policy heeft een hoek van ROOM de waarde −36: de robot heeft gemiddeld 36 stappen nodig tot het laadstation. Nu verandert de policy één keer: op elke tegel neemt de robot de action met de grootste $Q$ onder deze waarden van de random policy, en vanaf dan blijft de robot bij deze action. Hoeveel stappen zijn er nu nodig vanuit een hoek?",
        options={
            "same": "Ongeveer 36, net als eerst, want de waarden horen bij de random policy en de policy veranderde slechts één keer",
            "half": "Ongeveer 18: één verandering maakt de policy beter, maar niet de beste",
            "right": "4",
            "eight": "Ongeveer 8, want de robot maakt hier en daar nog een omweg",
        },
        reveal="**4.** Eén verandering is hier genoeg: vanaf elke tegel gaat de robot nu recht naar het laadstation, dus vanuit een hoek zet de robot de 4 stappen van de kortste weg, waar de random robot er gemiddeld 36 nodig had. De waarden van de random policy zeggen weinig over het aantal stappen van de nieuwe policy, maar wijzen wel de goede kant op: een tegel dichter bij het laadstation heeft een hogere waarde. *Het idee* laat zien waarom de action met de grootste $Q$ een policy nooit slechter maakt.",
    )
    guess_steps
    return (guess_steps,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De waarden $V_\pi$ van een policy $\pi$ zeggen wat elke state waard is als je $\pi$ volgt. Daarmee kun je voor elke action uitrekenen wat het waard is om de action één keer te nemen en daarna $\pi$ te volgen: dat is $Q_\pi(s, a)$, het binnenste deel van de Bellman equation van een policy. **Policy improvement** maakt een nieuwe policy $\pi'$ die in elke state de action met de grootste $Q_\pi(s, a)$ neemt:
        """),
        L.formula(r"\pi'(s) = \arg\max_a \sum [[data:p]] \left( [[data:r]] + [[hyper:\gamma]] \, [[pred:V_\pi(s')]] \right)"),
        mo.md(r"""
        $\arg\max_a$ is de action $a$ waarvoor wat erachter staat het grootst is. Waarom is $\pi'$ minstens zo goed als $\pi$? In state $s$ is $V_\pi(s)$ het gemiddelde van de $Q_\pi(s, a)$, elk meegeteld met $\pi(a \mid s)$, en de grootste $Q$ is minstens zo groot als dat gemiddelde. Dus is het minstens $V_\pi(s)$ waard om in $s$ één keer de beste action te nemen en daarna $\pi$ te volgen. Dat geldt in elke state, dus ook in de state waar de eerste stap landt, en in de state daarna: neem bij elke stap de beste action, en de waarde van $\pi'$ is in elke state minstens zo groot als de waarde van $\pi$. Dit is de **policy improvement theorem**. En als de verbetering niets verandert, is in elke state de waarde van $\pi$ al gelijk aan de grootste $Q$: $V_\pi$ past dan bij de Bellman equation van value iteration, en $\pi$ is een beste policy.

        **Policy iteration** zet de twee stappen achter elkaar: evalueer de policy, verbeter de policy, evalueer de verbeterde policy, verbeter opnieuw, tot de verbetering de policy ongewijzigd teruggeeft. Een **ronde** van policy iteration is één hele policy evaluation plus één verbetering; de rondes van value iteration waren losse sweeps. Elke ronde geeft een policy die minstens zo goed is als de vorige, en ergens beter, tenzij de waarden al de beste waren. Zijn de waarden de beste, dan geeft een vaste keuze tussen even goede actions, zoals steeds de eerste, een ronde later dezelfde policy terug; oefening 3 laat zien waarom de vaste keuze ertoe doet. Er bestaat slechts een beperkt aantal policies met één action per state, dus houden de rondes op, bij een beste policy.

        [Als stappen kunnen uitglijden: value iteration](/course/open/rl-value-iteration?lang=nl) is een korte vorm van hetzelfde idee: elke sweep neemt meteen de grootste $Q$, alsof elke policy evaluation één sweep duurde. Daartussen liggen veel mengvormen, zoals een paar sweeps evaluatie en dan een verbetering. **Generalized policy iteration** is de naam voor elke methode waarin waarden en een policy elkaar om de beurt beter maken: de waarden gaan passen bij de policy, en de policy gaat de beste action nemen onder de waarden.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **policy improvement** | | een nieuwe policy die in elke state de action met de grootste $Q$ neemt onder de waarden van de oude policy | `improve` |
        | $\pi'$ | pi accent | de verbeterde policy | `new` |
        | $\arg\max_a$ | arg max over a | de action $a$ waarvoor wat erachter staat het grootst is | `np.argmax` |
        | **policy improvement theorem** | | de verbeterde policy is in elke state minstens zo goed als de oude | |
        | **policy iteration** | | om de beurt evalueren en verbeteren, tot de verbetering de policy ongewijzigd teruggeeft | `policy_iteration` |
        | **ronde** | | bij policy iteration: één hele policy evaluation plus één verbetering | `rounds` |
        | **generalized policy iteration** | | elke methode waarin waarden en een policy elkaar om de beurt beter maken | |
        """),
        L.watch_out("Policy improvement maakt een waarde nooit slechter; wel kan een ronde veel waarden ongewijzigd laten: de policy improvement theorem zegt alleen dat de nieuwe policy in elke state minstens zo goed is."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Policy iteration op HALL, een hal van 5 bij 6 tegels met het laadstation linksboven en het startpunt S van de robot linksonder. Acht tegels zijn plassen: een stap die in een plas eindigt kost −10 in plaats van −1. Ronde 1 evalueert de random policy; elke latere ronde toont de policy na nog een verbetering, de <span class="c-pred">waarden</span> en de weg vanaf S. De staven ernaast tellen de tegels die aan het eind van elke ronde een nieuwe action krijgen.
    """)
    return


@app.cell(hide_code=True)
def explore_rounds_cell():
    explore_rounds = L.explore("rounds", "ronde", lambda _v: C.play_frame(int(_v), L.lang), start=1,
                               values=C.PLAY_ROUNDS, role="",
                               task="Ga de rondes langs. In welke ronde gaat de robot vanaf S voor het eerst om de plassen heen, en wat levert dat op? Volg rij 2 van ronde 5 tot ronde 8: in welke volgorde krijgen de tegels een pijl naar rechts?")
    explore_rounds
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Ronde 1 kost 972 sweeps: vanaf S heeft de random robot gemiddeld zo'n 114 stappen nodig, waarvan zo'n 28 in een plas, met elk 9 extra, dus begint S op −365,64. Na één verbetering gaat de robot vanaf S recht omhoog, door de plas aan de linkerkant: 4 stappen, −13. De verbeteringen aan het eind van ronde 2 tot en met 8 veranderen daarna 9, 5, 3, 1, 1, 4 en 3 tegels. De droge weg over rij 2 verspreidt zich met één tegel per ronde, vanaf het laadstation terug naar S: elke verbetering kijkt slechts één stap vooruit, met de waarden van de oude policy. In ronde 8 gaat de robot vanaf S om de plassen heen, 12 stappen zonder plas: −12, één beter dan −13. Ronde 9 geeft geen enkele tegel een nieuwe action, dus is de policy stabiel. De latere evaluaties kostten elk 10 tot 16 sweeps: de waarden van een policy zonder omwegen komen snel tot rust.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Policy iteration, met een tabel `T`:

    1. Begin met een policy, zoals de random policy.
    2. Evalueer de policy: sweeps van policy evaluation tot de grootste verandering kleiner is dan $\theta$.
    3. Verbeter de policy: reken in elke state de $Q$ van elke action uit met deze waarden, en neem de action met de grootste $Q$; van even goede actions altijd de eerste.
    4. Is de verbeterde policy gelijk aan de oude, stop dan: dat is een beste policy. Ga anders verder bij stap 2 met de verbeterde policy.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eén verbetering op ROOM, met de waarden van de random policy bij $\theta$ = 0,001. De cel rekent met deze waarden de vier $Q$-waarden van de hoek `(0, 0)` uit, stap 3 van het recept voor één state. Daarna verbetert de cel elke tegel met de `improve` van de course en evalueert de verbeterde policy: de cel toont de policy in de hoek, de waarde van de hoek en het aantal sweeps, en de waarde van een tegel naast het laadstation. Voer de cel uit.
    """)
    return


@app.cell
def example_corner():
    # 2. de waarden van de random policy
    _V, _n = C.until_stable(C.T_ROOM, C.RANDOM, 1.0, 0.001)
    # 3. de Q van elke action in de hoek, uit deze waarden
    for _a in C.T_ROOM[(0, 0)]:
        _q = 0
        for _p, _nxt, _r in C.T_ROOM[(0, 0)][_a]:
            _q = _q + _p * (_r + 1.0 * _V[_nxt])
        print(_a, round(_q, 2))
    # 3. de verbeterde policy op elke tegel, en de waarden ervan
    _better = C.improve(C.T_ROOM, _V, 1.0)
    _W, _m = C.until_stable(C.T_ROOM, _better, 1.0, 0.001)
    print("hoek:", _better[(0, 0)], round(_W[(0, 0)], 2), "na", _m, "sweeps")
    print("naast het laadstation:", round(_W[(1, 2)], 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Omhoog en links botsen tegen de rand: de robot blijft in de hoek, dus is de $Q$ van beide −1 − 35,97 = −36,97. Omlaag en rechts gaan naar tegels die onder de random policy −33,97 waard zijn, dus is de $Q$ daar −34,97. Omlaag en rechts zijn even goed, en `improve` neemt de eerste van de twee, omlaag. Na de verbetering gaat elke tegel recht op het laadstation af. De nieuwe policy evalueren kost slechts 5 sweeps: na 4 sweeps is elke weg naar het laadstation helemaal geteld, en de vijfde sweep verandert niets. De hoek staat op −4, de tegel naast het laadstation op −1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_improve():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De action met de grootste Q

        `improve(T, V, gamma)` maakt de verbeterde policy uit de waarden `V`: in elke state `s` kans `1.0` voor de action met de grootste $Q$, en `0.0` voor elke andere action. `actions` bevat de actions van `s`, in de volgorde van `T[s]`. Voor elke action `a` rekent de loop over `p, nxt, r` de waarde `q` uit, zoals bij value iteration, en `qs.append(q)` zet `q` in de lijst `qs`, in dezelfde volgorde als `actions`. Vervang de `...` zodat `best` de action met de grootste $Q$ is; neem van even goede actions de eerste. De loop aan het eind geeft `best` dan kans `1.0` in `pi[s]`. De check roept `improve(T_ROOM, V_ROOM, 1.0)` aan, met `V_ROOM` de waarden van de random policy, en nog vijf andere gevallen, waaronder HALL en de gang uit de les over policy evaluation.
        """),
        L.task("Laat `improve` elke state kans 1 geven voor de action met de grootste Q."),
    ])
    return


@app.cell(hide_code=True)
def exercise_improve():
    box_improve = L.code_box("improve", lines=18)
    box_improve
    return (box_improve,)


@app.cell(hide_code=True)
def check_improve(box_improve):
    result_improve = L.check("improve", box_improve.value)
    result_improve
    return (result_improve,)


@app.cell(hide_code=True)
def hints_improve():
    L.hints(
        "improve",
        [
            "Stap 3 van het recept: `qs` bevat de $Q$ van elke action, in de volgorde van `actions`. `best` moet een action zijn, zoals `\"down\"`, geen getal: zoek de plek van de grootste $Q$ in `qs`, en dan de action op deze plek in `actions`.",
            "Bij value iteration vond je de plek van de grootste $Q$ met NumPy, de eerste plek als er meer even groot zijn. In de hoek van ROOM is `qs` ongeveer [−36,97; −34,97; −36,97; −34,97], dus is de plek 1, en de action op plek 1 is omlaag.",
            "`        best = ___`",
        ],
        why="`best = actions[np.argmax(qs)]`: `np.argmax(qs)` is de plek van de grootste $Q$, de eerste van even grote, en `actions` bevat de action op elke plek. De verbeterde policy neemt deze action met kans 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_iteration():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Evalueren, verbeteren, herhalen

        `policy_iteration(T, gamma, theta)` voert policy iteration uit vanaf de random policy en geeft de stabiele policy terug, met de waarden en het aantal rondes. De eerste regels maken de random policy `pi`: in elke state `s` krijgt elke action `a` kans `1 / len(T[s])`. `rounds` telt de rondes. Elke ronde evalueert `pi` met `until_stable(T, pi, gamma, theta)`, zoals in de les over policy evaluation; de functie geeft de waarden `V` en het aantal `sweeps` terug. Vervang de twee `...`: `new` moet de verbeterde policy zijn, uit `improve`, de course-versie van oefening 1, en de loop moet stoppen zodra `new` dezelfde policy is als `pi`, stap 4 van het recept. De check roept `policy_iteration(T_ROOM, 1.0, 0.01)` aan, en nog vijf andere gevallen, waaronder HALL, de hal van het speelblok.
        """),
        L.task("Laat `policy_iteration` de policy elke ronde verbeteren, en stoppen zodra de verbetering dezelfde policy teruggeeft."),
    ])
    return


@app.cell(hide_code=True)
def exercise_iteration():
    box_iteration = L.code_box("iteration", lines=14)
    box_iteration
    return (box_iteration,)


@app.cell(hide_code=True)
def check_iteration(box_iteration):
    result_iteration = L.check("iteration", box_iteration.value)
    result_iteration
    return (result_iteration,)


@app.cell(hide_code=True)
def hints_iteration():
    L.hints(
        "iteration",
        [
            "Stap 3: `new` komt uit `improve`, en `improve` heeft de tabel, de waarden van `pi` en de discount factor nodig. Stap 4: de loop mag stoppen zodra `new` en `pi` dezelfde policy zijn, in elke state en voor elke action; `if` kijkt daarnaar.",
            "Twee dictionaries zijn gelijk bij dezelfde keys met dezelfde waarden, en Python vergelijkt dictionaries met hetzelfde teken als twee getallen. Op ROOM geeft de verbetering aan het eind van ronde 3 dezelfde policy terug als de policy waarmee ronde 3 begon, dus stopt je functie na 3 rondes.",
            "`        if ___:`",
        ],
        why="`new = improve(T, V, gamma)` verbetert de policy met de waarden van deze ronde, en `if new == pi:` stopt zodra de verbetering de policy ongewijzigd teruggeeft: dan is `pi` een beste policy, en `V` de waarden ervan.",
    )
    return


@app.cell(hide_code=True)
def text_after_iteration():
    mo.md(r"""
    Op ROOM stopt je functie na 3 rondes, terwijl ronde 2 de beste waarden al heeft: −4 in een hoek, −1 naast het laadstation. De verbetering aan het eind van ronde 2 verandert toch nog vier tegels, zoals tegel `(1, 0)`. Met de waarden van de random policy ging deze tegel naar rechts; nu hebben omlaag en rechts allebei een $Q$ van −3, en `improve` neemt de eerste, omlaag. Ronde 3 evalueert deze policy, vindt dezelfde waarden, en de verbetering verandert niets. Op HALL stopt je functie na 9 rondes, met in totaal 1.068 sweeps, waarvan 972 voor de random policy. Value iteration heeft op HALL met dezelfde $\theta$ 16 sweeps nodig: policy iteration heeft weinig rondes nodig, maar elke ronde bevat een hele policy evaluation.

    ### Dezelfde verbetering met NumPy

    Voor elke action kun je de tabel als matrix schrijven, zoals in de les over policy evaluation. Nummer de 30 tegels van HALL rij voor rij. `P[i]` bevat de kans om met action nummer `i` van de ene tegel naar de andere te gaan, en `R[i]` de reward die je mag verwachten. Dan geeft `R[i] + gamma * P[i] @ v` de $Q$ van deze action in elke state tegelijk, en de array `Q`, met een rij voor elk van de vier actions, bevat alle $Q$-waarden. `np.argmax(Q, axis=0)` geeft voor elke kolom, elke state, de plek van de grootste $Q$. De cel bouwt `P` en `R` uit de `T_HALL` van de course, waarbij `enumerate` elke action van een tegel met het nummer `i` geeft, 0 tot en met 3. Daarna verbetert de cel zo de random policy van HALL, `RANDOM_HALL`, met de waarden van deze policy, en print een pijl voor de action van elke tegel, rij voor rij. Voer de cel uit.
    """)
    return


@app.cell
def example_numpy():
    import numpy as np

    _at = {}   # het nummer van elke tegel, rij voor rij
    for _s in C.T_HALL:
        _at[_s] = len(_at)
    _P = np.zeros((4, 30, 30))
    _R = np.zeros((4, 30))
    for _s in C.T_HALL:
        for _i, _a in enumerate(C.T_HALL[_s]):
            for _p, _nxt, _r in C.T_HALL[_s][_a]:
                _P[_i, _at[_s], _at[_nxt]] = _P[_i, _at[_s], _at[_nxt]] + _p
                _R[_i, _at[_s]] = _R[_i, _at[_s]] + _p * _r
    _V, _n = C.until_stable(C.T_HALL, C.RANDOM_HALL, 1.0, 0.001)
    _v = np.array([_V[_s] for _s in C.T_HALL])
    _Q = np.array([_R[_i] + 1.0 * _P[_i] @ _v for _i in range(4)])
    _best = np.argmax(_Q, axis=0)
    _signs = "↑↓←→"   # omhoog, omlaag, links, rechts: de volgorde van de actions
    for _row in range(5):
        _line = ""
        for _c in range(6):
            if C.HALL[_row][_c] == "G":
                _line = _line + "G "
            else:
                _line = _line + _signs[_best[6 * _row + _c]] + " "
        print(_line)
    return


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    Dit zijn de pijlen van ronde 2 in het speelblok. Onder de waarden van de random policy lijkt een tegel dichter bij het laadstation beter, dus gaat elke tegel langs de kortste weg naar het laadstation, met of zonder plas: kolom 0, en kolom 1 boven de onderste rij, gaan recht omhoog, door de bovenste rij plassen. Pas als ronde 2 deze policy evalueert, blijkt wat een plas kost.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ties():
    mo.md(r"""
    ### Oefening 3. Even goede actions

    Op ROOM had ronde 2 de beste waarden al, en toch veranderde de verbetering vier tegels, elk van de ene beste action naar een andere, net zo goede action. `improve` neemt de eerste van even goede actions, dus verandert de policy na één zo'n wissel niet meer.
    """)
    return


@app.cell(hide_code=True)
def choice_ties_cell():
    choice_ties = L.choice(
        "ties",
        question="Stel dat `improve` een van de even goede actions willekeurig koos, met een nieuwe trekking in elke ronde. Wat kan dat met je `policy_iteration` doen?",
        options={
            "worse": "Je functie eindigt dan met een slechtere policy, want een willekeurige action is soms een slechte",
            "sweeps": "Elke evaluatie kost dan veel meer sweeps, want een policy met willekeurige keuzes geeft elke state waarden die rommelig zijn en traag tot rust komen",
            "right": "Je functie kan heel lang doorgaan: de policy kan blijven wisselen tussen even goede actions, dus blijft de nieuwe policy verschillen van de oude",
            "same": "Niets: op ROOM stopt je functie nog steeds na 3 rondes",
        },
        correct="right",
        why={
            "right": "Met een nieuwe trekking in elke ronde kan tegel `(1, 0)` in de ene ronde omlaag krijgen en in de volgende rechts. Beide zijn de beste, dus blijven de waarden gelijk, maar de policy blijft veranderen, en de vergelijking van `new` en `pi` vindt de twee zelden gelijk. Zijn de waarden de beste, dan hebben 16 tegels van ROOM twee beste actions, en een nieuwe trekking herhaalt alle 16 keuzes slechts één ronde op de 65.536 (2 tot de macht 16), dus duurt de loop gemiddeld zo'n 65.536 rondes. Een vaste keuze tussen even goede actions, zoals `np.argmax` maakt, voorkomt dit; een andere manier is stoppen zodra de waarden niet meer veranderen.",
            "worse": "Elke action in de trekking heeft de grootste $Q$: kan een trekking tussen beste actions een slechte action kiezen?",
            "sweeps": "Elke ronde evalueert nog steeds één vaste policy: gebeurt de trekking tijdens de evaluatie, of alleen bij de verbetering?",
            "same": "In ronde 3 op ROOM heeft tegel `(1, 0)` weer twee beste actions: geeft een nieuwe trekking altijd de action van ronde 2?",
        },
    )
    choice_ties
    return (choice_ties,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Wat is een policy waard? Policy evaluation](/course/open/rl-policy-evaluation?lang=nl) rekende je de waarden van een gegeven policy uit. Je `improve` maakt van deze waarden een policy die minstens zo goed is, door in elke state de action met de grootste $Q$ te nemen, en je `policy_iteration` herhaalt evaluatie en verbetering tot de policy niet meer verandert: dan is de policy een beste policy, op ROOM na 3 rondes en op HALL na 9. Waarden die bij een policy passen, en een policy die de beste action neemt onder de waarden: dat heen en weer, generalized policy iteration, drijft ook de komende lessen, waarin de tabel `T` er niet meer is. De volgende les, *Leren van hele potjes: Monte Carlo prediction*, schat de waarden van een policy uit wandelingen die de robot echt maakt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_ties, result_improve, result_iteration):
    L.wrap_up(result_improve, result_iteration, choice_ties)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
