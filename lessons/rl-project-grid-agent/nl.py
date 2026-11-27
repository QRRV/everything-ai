# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-project-grid-agent. Route: course/route/lessen/reinforcement-learning.yaml. Checks, de wereld en de agent van de course, de runs, de demo en de afbeeldingen: checks.py; de loop die de course in je project zet: project/agent/episode.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("rl-project-grid-agent", "nl")
    C = L.checks   # de wereld, de agent en de runs van de course, de controle van je wereld, de demo en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een kaart van RL-methoden](/course/open/rl-kinds-of-rl?lang=nl) en [Project: je eigen bandit-agent](/course/open/rl-project-bandit?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kinds():
    quick_kinds = L.choice(
        "quick-kinds",
        question="Monte Carlo control en SARSA leren allebei de $Q$ van de policy die de agent volgt, zonder model. Welke van de vier vragen van de kaart onderscheidt de twee?",
        options={
            "model": "Monte Carlo control heeft een model van de environment nodig, met de kans op elke volgende state, en SARSA niet",
            "right": "Monte Carlo leert van hele episodes, SARSA van de eigen schattingen",
            "off": "SARSA is off-policy",
        },
        correct="right",
        why={
            "right": "Monte Carlo schuift $Q(S, A)$ op naar de return die echt volgde, dus wacht Monte Carlo op het einde van de episode. SARSA schuift $Q(S, A)$ na elke stap op naar $R + \\gamma Q(S', A')$, met $Q(S', A')$ de eigen schatting. Beide methoden zijn model-free en on-policy.",
            "model": "Beide methoden leren alleen van stappen die de agent echt gezet heeft, zonder de kans op elke volgende state: waar schuift elke methode $Q(S, A)$ naartoe? Zie [Een kaart van RL-methoden](/course/open/rl-kinds-of-rl?lang=nl).",
            "off": "SARSA leert van de action $A'$ die de agent echt als volgende neemt, met de policy die de agent volgt: is dat on-policy of off-policy? Zie [Een kaart van RL-methoden](/course/open/rl-kinds-of-rl?lang=nl).",
        },
    )
    quick_kinds
    return (quick_kinds,)


@app.cell(hide_code=True)
def quick_check_fair():
    quick_fair = L.choice(
        "quick-fair",
        question="In je bandit-project speelde elke agent op dezelfde 200 bandits, gemaakt met dezelfde seeds. Waarom?",
        options={
            "speed": "Zodat de run minder tijd kost",
            "right": "Zodat een verschil tussen de agents aan de agents ligt, niet aan makkelijkere bandits",
            "luck": "Zodat elke agent dezelfde rewards krijgt, aan welke automaat de agent ook trekt, en elke run op hetzelfde totaal eindigt",
        },
        correct="right",
        why={
            "right": "Met dezelfde bandits geeft dezelfde automaat bij dezelfde trek voor elke agent hetzelfde. Blijft er een verschil over, dan ligt het verschil aan wat de agents kiezen.",
            "speed": "Het maken van 200 bandits duurt even lang, met welke seeds ook: wat zou er veranderen als elke agent eigen bandits kreeg? Zie [Project: je eigen bandit-agent](/course/open/rl-project-bandit?lang=nl).",
            "luck": "De agents trekken aan verschillende automaten, dus krijgen de agents verschillende rewards en verschillende totalen: wat houden dezelfde bandits gelijk? Zie [Project: je eigen bandit-agent](/course/open/rl-project-bandit?lang=nl).",
        },
    )
    quick_fair
    return (quick_fair,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        In [Voorzichtig of gedurfd: SARSA en Q-learning](/course/open/rl-sarsa?lang=nl) en [Beter spelen door ervaring: Monte Carlo control](/course/open/rl-monte-carlo-control?lang=nl) maakte de course de werelden, en leerde de code van de course deze werelden. In dit project maak je beide delen zelf: een grid world in je project, met muren, plassen en putten op de plekken die jij kiest, en één agent die de wereld op drie manieren kan leren, met de targets van Monte Carlo control, SARSA en Q-learning. Dan zoek je uit welke van de drie het best werkt op je wereld, met de zuivere vergelijking van je bandit-project: dezelfde wereld, veel runs, en een learning curve voor elke methode.

        De beste methode hangt af van de wereld. Op de ene wereld wint een methode, op de volgende kan dezelfde methode verliezen, en één run kan de slechtere methode beter laten lijken. Pas met een wereld die je kunt veranderen en een vergelijking die je kunt herhalen zie je waarom.

        **Een voorbeeld uit het dagelijks leven.** Een bakker wil weten welk van drie recepten het lekkerste brood geeft. Eén brood van elk recept, gebakken op verschillende dagen, zegt weinig: de ene dag is de oven te heet, de andere dag is het meel vochtig. Bakt de bakker elk recept tien keer, in dezelfde oven met hetzelfde meel, dan laten de recepten zien wat elk recept waard is. En in een andere oven kan een ander recept winnen, dus probeert de bakker ook een tweede oven.
        """),
        draw.project_steps(["Je bandit-agent", "Je grid-agent"], 1,
                           here_label="← vandaag", title="Waar je project staat: agent"),
    ])
    return


@app.cell(hide_code=True)
def predict_method():
    guess_method = L.guess(
        "method",
        question="In deze les leert je agent de wereld van de course, op de afbeelding, met elke methode: 10 runs van 300 episodes, allemaal met ε = 0,1, $\\eta$ = 0,2 en $\\gamma$ = 1. Elke stap kost −1, een stap op een plas −10, en een stap in een put −100, waarna de agent opnieuw begint op S. De kortste weg, 7 stappen, loopt tussen de plassen en de putten door; een weg van 11 stappen gaat over de bovenste rij. Welke methode heeft over de laatste 50 episodes de hoogste gemiddelde return, denk je?",
        options={
            "mc": "Monte Carlo",
            "q": "Q-learning",
            "right": "SARSA",
            "same": "Alle drie ongeveer gelijk",
        },
        reveal="**SARSA.** Over de laatste 50 episodes heeft SARSA een gemiddelde return van −16,7, Q-learning −30,4 en Monte Carlo −42,0. Q-learning leert de kortste weg, langs de putten, en dat kost −100, elke keer dat een willekeurige stap de agent in een put laat vallen. SARSA leert de weg over de bovenste rij. De demo laat je eigen agent dezelfde wereld leren.",
    )
    mo.vstack([mo.Html(C.world_picture(L.lang)), guess_method])
    return (guess_method,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een eigen wereld.** Je wereld is een lijst met strings, één string per rij en één teken per vakje. In elk vakje kan de agent omhoog, omlaag, naar links of naar rechts. Het effect van een stap hangt af van het teken van het vakje waar de stap heen leidt:

        | Teken | Vakje | Een stap erheen |
        |---|---|---|
        | `.` of `S` | vrij, of de start | −1 |
        | `G` | het doel | −1, en de episode is voorbij |
        | `W` | een plas | −10 |
        | `X` | een put | −100, en de agent gaat terug naar S; de episode gaat verder |
        | `#` | een muur | de agent blijft staan, en de stap kost evenveel als een stap op het vakje waar de agent staat |

        Een stap tegen de rand van de wereld werkt als een stap tegen een muur. Elke stap kost iets, dus krijgt de agent die G in zo weinig mogelijk stappen haalt, zonder plassen en putten, de hoogste return, net als op het klif van [Voorzichtig of gedurfd: SARSA en Q-learning](/course/open/rl-sarsa?lang=nl). De agent ziet alleen het vakje waar de agent staat, als (rij, kolom), en de reward: de agent kent de tekens en de regels niet. Alle drie de methoden zijn model-free.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        Op de afbeelding een wandeling van 13 stappen door de wereld van de course. Vanaf S gaat de agent naar links, tegen de rand, en blijft staan; dan omhoog, naar rechts op een plas, omlaag, naar rechts, en omlaag in een put, die de agent terugzet op S. Zeven stappen naar rechts brengen de agent dan tussen de plassen en de putten door naar G. De rewards zijn −1, −1, −10, −1, −1, −100 en zeven keer −1: een return van −121.

        **Eén loop.** Net als in je bandit-project wisselen een agent en een environment elkaar af in een loop. De course zet deze loop in je project, als `work/projects/agent/episode.py`:

        ```python
        def episode(agent, world, limit):
            state = world.reset()
            action = agent.act(state)
            total = 0
            for t in range(limit):
                nxt, reward, done = world.step(action)
                nxt_action = agent.act(nxt)
                agent.learn(state, action, reward, nxt, nxt_action)
                total = total + reward
                if done:
                    break
                state = nxt
                action = nxt_action
            agent.end()
            return total
        ```

        De agent kiest de volgende action eerst en leert pas daarna, omdat SARSA van deze action leert. Na de laatste stap laat `agent.end()` Monte Carlo leren van de hele episode. Een episode eindigt op G, of stopt na `limit` stappen, 200 in deze les: anders kan een agent die weinig geleerd heeft eindeloos rondjes lopen. `episode` geeft de return terug, het totaal van de rewards, omdat $\gamma$ = 1.

        **Eén agent, drie targets.** Je `GridAgent` bewaart $Q$ in een dict: voor elke state die de agent al tegenkwam, een dict met de $Q$ van elke action. De agent kiest ε-greedy, en schuift $Q(S, A)$ een deel $\eta$ van de weg op naar een target, zoals in [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl):
        """),
        L.formula(r"[[param:Q(S, A)]] \leftarrow [[param:Q(S, A)]] + [[hyper:\eta]] \left( \text{target} - [[param:Q(S, A)]] \right)"),
        mo.md(r"""
        Alleen het target verschilt per methode:

        - SARSA: $R + \gamma Q(S', A')$, met $A'$ de action die de agent als volgende neemt.
        - Q-learning: $R + \gamma \max_{a'} Q(S', a')$, de grootste $Q$ in de volgende state.
        - Monte Carlo: de return na $S$ en $A$, uitgerekend na de episode, van de laatste stap terug naar de eerste: de reward van een stap plus $\gamma$ keer de return na deze stap.

        In [Beter spelen door ervaring: Monte Carlo control](/course/open/rl-monte-carlo-control?lang=nl) was $Q$ het gemiddelde van de returns, met $\eta = 1/n$. Hier schuift Monte Carlo met dezelfde vaste $\eta$ als de andere twee methoden, zoals de schattingen van de verschuivende automaten in je bandit-project: de policy verandert terwijl de agent leert, dus zeggen oude returns er minder over. En Monte Carlo leert van elk bezoek aan een state en een action in een episode, niet alleen van het eerste. De agent heeft geen regel voor G nodig: geen enkele stap begint op G, dus blijft elke $Q$ van G 0, en is het target van de laatste stap de reward van deze stap.

        **Een zuivere vergelijking.** Elke methode krijgt dezelfde wereld en dezelfde instellingen: ε = 0,1, $\eta$ = 0,2 en $\gamma$ = 1. In de wereld zit geen toeval: dezelfde action in hetzelfde vakje geeft altijd hetzelfde. Twee runs verschillen dus alleen in de willekeurige keuzes van de agents. Een run is één nieuwe agent op één nieuwe wereld, 300 episodes lang; run $k$ geeft de agent `np.random.default_rng(k)`, met $k$ van 0 tot en met 9, voor elke methode. Je `learning_curve` middelt de returns van de 10 runs bij elke episode, met je `average` uit je bandit-project.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **muur** | | een vakje waar de agent niet in kan: na een stap ertegen staat de agent nog op dezelfde plek | `"#"` |
        | **plas** | | een vakje waarop een stap −10 kost | `"W"` |
        | **put** | | een vakje: een stap erin kost −100 en zet de agent terug op S | `"X"` |
        | **limiet** | | het maximale aantal stappen van een episode: daarna stopt de episode, ook als de agent G nog niet gehaald heeft | `limit` |
        | **learning curve** | | de gemiddelde return over de runs bij elke episode | `learning_curve` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een eigen agent op een eigen wereld:

    1. Teken de wereld als een lijst met strings, en zorg dat er een weg van S naar G loopt.
    2. Schrijf de wereld als class: `reset` zet de agent op S, en `step(action)` verplaatst de agent volgens de regels en geeft terug waar de agent staat, de reward, en of de episode voorbij is.
    3. Schrijf de agent als class: een dict met $Q$-waarden, een ε-greedy `act`, en een `learn` die $Q(S, A)$ opschuift naar het target van de methode. Voor Monte Carlo bewaart `learn` de stappen, en leert `end` er na de episode van.
    4. Laat de agent en de wereld elkaar afwisselen met `episode`, tot de limiet.
    5. Maak voor elke methode en elke seed een nieuwe wereld en een nieuwe agent, speel alle episodes, en middel de returns bij elke episode over de seeds.
    6. Vergelijk de learning curves, de gemiddelde return over de laatste episodes, en waar de beste actions vanaf S heen leiden. Verander dan de wereld, en vergelijk opnieuw.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een agent met $\eta$ = 0,5 en $\gamma$ = 0,9 heeft deze $Q$-waarden in het vakje (0, 1): up −4, down −6, left −8 en right −2. Elke andere $Q$ is nog 0. De agent zet twee stappen: van (0, 0) een stap naar rechts, naar (0, 1), met reward −1, en kiest daar down; dan omlaag op een plas in (1, 1), met reward −10, en kiest daar left. Stel dat de episode daar stopt.

    - SARSA, stap 1: het target is $-1 + 0{,}9 \cdot Q$((0, 1), down) $= -1 + 0{,}9 \cdot (-6) = -6{,}4$, dus $Q$((0, 0), right) $= 0 + 0{,}5 \cdot (-6{,}4 - 0) = -3{,}2$.
    - Q-learning, stap 1: de grootste $Q$ in (0, 1) is −2, van right, dus het target is $-1 + 0{,}9 \cdot (-2) = -2{,}8$, en $Q$((0, 0), right) $= 0 + 0{,}5 \cdot (-2{,}8 - 0) = -1{,}4$.
    - Stap 2: elke $Q$ van (1, 1) is nog 0, dus beide targets zijn −10, en $Q$((0, 1), down) $= -6 + 0{,}5 \cdot (-10 - (-6)) = -8$.
    - Monte Carlo verandert niets tijdens de twee stappen. Na de episode gaat Monte Carlo terug: de return na stap 2 is $-10 + 0{,}9 \cdot 0 = -10$, dus wordt $Q$((0, 1), down) ook −8. De return na stap 1 is $-1 + 0{,}9 \cdot (-10) = -10$, dus $Q$((0, 0), right) $= 0 + 0{,}5 \cdot (-10 - 0) = -5$.

    Voor de eerste stap zijn de drie targets −6,4, −2,8 en −10. Q-learning verwacht het meest: Q-learning gaat uit van de beste action in (0, 1), right. Monte Carlo verwacht het minst: Monte Carlo zag wat er echt op volgde: de plas.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_world():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Je GridWorld

        `GridWorld(rows)` is de environment. `__init__` is gegeven: de class bewaart `rows`, de hoogte `height` en de breedte `width` van de wereld, het vakje van S als `self.start`, en zet de agent daar neer, in `self.state`. `reset` zet de agent terug op S. In `step(action)` zijn de eerste regels gegeven: `MOVES` zegt hoe de rij en de kolom veranderen bij elke action, en `r` en `c` worden de rij en de kolom van het vakje waar de action heen leidt, of van het vakje waar de agent staat als de action tegen de rand of een muur leidt. `sign` is het teken van dat vakje.

        Vervang de `...` door de regels. In een put gaat de agent terug naar S, met −100, en de episode gaat verder. Anders gaat de agent naar het vakje van `r` en `c`, en krijgt dan −1 en het einde van de episode op G, −10 op een plas, en −1 op elk ander vakje. Elke regel geeft drie waarden terug: waar de agent nu staat, de reward, en of de episode voorbij is.

        De check maakt `GridWorld(["S#G", ".W.", "X.."])` en zet acht stappen: right, up, down, down, down, right, right en up. De resultaten horen `[((0, 0), -1, False), ((0, 0), -1, False), ((1, 0), -1, False), ((0, 0), -100, False), ((1, 0), -1, False), ((1, 1), -10, False), ((1, 2), -1, False), ((0, 2), -1, True)]` te zijn: een muur, de rand, een put, een plas en het doel. Daarna probeert de check twee eigen gevallen: `reset`, en een wereld waarin S niet in een hoek staat. Is het goed, dan bewaart de check je class in je project, als `work/projects/agent/gridworld.py`.
        """),
        L.task("Schrijf de regels van `step`: vanuit een put terug naar S, anders naar het nieuwe vakje, steeds met de reward van het teken en of de episode voorbij is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_world():
    box_world = L.code_box("world", lines=32)
    box_world
    return (box_world,)


@app.cell(hide_code=True)
def check_world(box_world):
    result_world = L.check("world", box_world.value)
    result_world
    return (result_world,)


@app.cell(hide_code=True)
def hints_world():
    L.hints(
        "world",
        [
            "Eerst de put: als `sign` gelijk is aan `\"X\"`, wordt `self.state` gelijk aan `self.start`, en geeft `step` `self.state`, `-100` en `False` terug, nog voordat de agent zich verplaatst. Dan wordt `self.state` het vakje van `r` en `c`. Dan één regel voor `\"G\"`, één voor `\"W\"`, en een laatste `return` voor elk ander vakje.",
            "In het voorbeeld gaat de vierde stap van (1, 0) omlaag, de put op (2, 0) in: `step` geeft `((0, 0), -100, False)` terug, en de agent staat weer op S, dus gaat de volgende stap omlaag naar (1, 0). Een stap tegen de muur op (0, 1) laat `r` en `c` op het vakje waar de agent staat, (0, 0), en de regels gaan dan verder als bij een vrij vakje. Onder het blok voor de put verplaatst één regel de agent naar het nieuwe vakje.",
            "`        self.state = ___`",
        ],
        why="`if sign == \"X\":` met daaronder `self.state = self.start` en `return self.state, -100, False`; dan `self.state = (r, c)`; dan `if sign == \"G\":` met `return self.state, -1, True`, `if sign == \"W\":` met `return self.state, -10, False`, en als laatste `return self.state, -1, False`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_agent(result_world):
    result_world  # opnieuw gemaakt na de check van oefening 1, die je wereld bewaart
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Je GridAgent

        `GridAgent(method, epsilon, eta, gamma, rng)` is één agent voor alle drie de methoden: `method` is `"mc"`, `"sarsa"` of `"q"`, en `ACTIONS` bevat de vier actions. Gegeven zijn `__init__`, dat de instellingen bewaart, een dict `q` en een lijst `steps`; `qs(state)`, dat de dict met de $Q$ van elke action in `state` geeft, en er eerst een maakt met vier nullen als de agent `state` nog niet gezien heeft; `act(state)`, ε-greedy; en `move(state, action, target)`, dat de $Q$ van `state` en `action` een deel `eta` van de weg opschuift naar `target`. `learn(state, action, reward, nxt, nxt_action)` krijgt één stap: voor Monte Carlo bewaart `learn` de stap alleen in `steps`. `end()` komt na de episode: `reversed(self.steps)` loopt de stappen langs van de laatste naar de eerste, `g` houdt de return bij, en elke $Q$ schuift ernaar op.

        Vervang de drie `...` door de targets: in `learn` het target van SARSA en het target van Q-learning, in `end` de nieuwe `g`, de return vanaf deze stap. `.values()` van een dict geeft de waarden van de dict, hier de vier $Q$-waarden van een state, in de volgorde van de actions; `act` gebruikt het al.

        De check maakt voor elke methode een agent met $\eta$ = 0,5 en $\gamma$ = 0,9, geeft het vakje (0, 1) de $Q$-waarden uit het uitgewerkte voorbeeld, laat de agent de twee stappen uit het uitgewerkte voorbeeld leren, en roept `end()` aan. $Q$((0, 0), right) en $Q$((0, 1), down) horen `[(-3.2, -8.0), (-1.4, -8.0), (-5.0, -8.0)]` te zijn, voor SARSA, Q-learning en Monte Carlo. Daarna probeert de check drie eigen gevallen: Monte Carlo met een vakje dat terugkomt in de episode, Q-learning met $\eta$ = 1, en SARSA na een put. Is het goed, dan bewaart de check je class als `work/projects/agent/gridagent.py`.
        """),
        L.task("Schrijf de drie targets: het target van SARSA en het target van Q-learning in `learn`, en de return `g` van Monte Carlo in `end`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_agent():
    box_agent = L.code_box("agent", lines=46)
    box_agent
    return (box_agent,)


@app.cell(hide_code=True)
def check_agent(box_agent):
    result_agent = L.check("agent", box_agent.value)
    result_agent
    return (result_agent,)


@app.cell(hide_code=True)
def hints_agent():
    L.hints(
        "agent",
        [
            "Elk target is `reward` plus `self.gamma` keer wat er na de stap komt. SARSA: de $Q$ van `nxt_action` in `nxt`, uit `self.qs`. Q-learning: `max` van de vier $Q$-waarden in `nxt`. Monte Carlo: `g`, de return na deze stap. `self.move(state, action, ...)` schuift dan de $Q$ van `state` en `action` op naar het target.",
            "In het uitgewerkte voorbeeld is het target van Q-learning bij de eerste stap −1 + 0,9 · (−2): −2 is de grootste van de vier $Q$-waarden in (0, 1). Het target van SARSA is −1 + 0,9 · (−6): −6 is de $Q$ van down, de action die de agent als volgende neemt. De `g` van Monte Carlo begint bij 0 en wordt vanaf de laatste stap terug opgebouwd: eerst −10 + 0,9 · 0, dan −1 + 0,9 · (−10).",
            "`            self.move(state, action, reward + self.gamma * max(___))`",
        ],
        why="SARSA: `self.move(state, action, reward + self.gamma * self.qs(nxt)[nxt_action])`. Q-learning: `self.move(state, action, reward + self.gamma * max(self.qs(nxt).values()))`. Monte Carlo, in `end`: `g = reward + self.gamma * g`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_curves(result_agent, result_world):
    result_world, result_agent  # opnieuw gemaakt na de checks van oefening 1 en 2, die je bestanden bewaren
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Learning curves

        `learning_curve(method, rows, seeds, episodes)` meet één methode op één wereld. Het begin is gegeven: voor elke seed een nieuwe `GridWorld(rows)` en een nieuwe `GridAgent` met ε = 0,1, $\eta$ = 0,2, $\gamma$ = 1 en `np.random.default_rng(seed)`. Vervang de `...` door de run van deze seed: een nieuwe lijst `returns`; voor elke episode één aanroep van de `episode` van de course, met hoogstens `LIMIT` stappen, waarvan de return in `returns` gaat; en dan deze lijst in `runs`. De laatste regel is gegeven en middelt de runs bij elke episode met je `average` uit je bandit-project.

        `small_world`, de `SMALL` van de course, is `["S..", "WX.", "..G"]`. `learning_curve("q", small_world, 3, 4)` hoort `[-139.667, -272.667, -48.0, -62.0]` te geven, op drie decimalen. De eerste twee episodes zijn lang, met putten onderweg; vanaf de derde halen de agents G sneller. Daarna probeert de check vier eigen gevallen: andere methoden, andere werelden, en andere aantallen seeds en episodes. Is het goed, dan bewaart de check je functie als `work/projects/agent/curves.py`.
        """),
        L.course_versions(),
        L.task("Laat `learning_curve` alle episodes van elke seed met één agent spelen, en zet de lijst met returns van elke seed in `runs`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_curves():
    box_curves = L.code_box("curves", lines=22)
    box_curves
    return (box_curves,)


@app.cell(hide_code=True)
def check_curves(box_curves, result_agent, result_world):
    result_world, result_agent  # opnieuw checken als oefening 1 en 2 je bestanden bewaren
    result_curves = L.check("curves", box_curves.value)
    result_curves
    return (result_curves,)


@app.cell(hide_code=True)
def hints_curves():
    L.hints(
        "curves",
        [
            "Voor één seed: `returns = []`, een `for`-loop over `range(episodes)`, en daarin de return van één episode van de `episode` van de course, toegevoegd met `returns.append`. Na deze loop `runs.append(returns)`.",
            "Met 3 seeds en 4 episodes eindigt `runs` als drie lijsten van vier returns, één lijst per seed, en geeft `average` vier getallen. Een nieuwe lijst voor elke seed houdt de runs uit elkaar. Maak per seed één agent, voordat de episodes beginnen: dan blijft de agent leren van episode tot episode.",
            "`            returns.append(___)`",
        ],
        why="`returns = []`, dan `for e in range(episodes):` met daaronder `returns.append(episode(agent, world, LIMIT))`, en na deze loop `runs.append(returns)`.",
    )
    return


@app.cell(hide_code=True)
def text_your_world():
    mo.md(r"""
    ### Je eigen wereld

    Teken nu een eigen wereld in de cel hieronder, en voer de cel uit. Gebruik de tekens uit de tabel van het idee: `.`, `S`, `G`, `#`, `W` en `X`. Een wereld heeft 1 tot en met 8 rijen van 2 tot en met 12 tekens nodig, alle rijen even lang, één S, minstens één G, en een weg van S naar G zonder muren of putten. Een paar ideeën: een doolhof van muren, een korte weg door een plas naast een langere droge weg, een put vlak naast de kortste weg, twee doelen. De afbeelding onder de cel toont je wereld zoals de demo de wereld ziet. De cel begint met de wereld van de course.
    """)
    return


@app.cell
def your_world():
    # je wereld: één string per rij; . vrij, S start, G doel, # muur, W plas, X put
    my_world = [
        "........",
        ".WWWWWW.",
        "S......G",
        ".XXXXXX.",
    ]
    return (my_world,)


@app.cell(hide_code=True)
def your_world_view(my_world):
    mo.Html(C.world_view(my_world, L.lang))
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Laat je agent je wereld leren

    De knop laat je `GridAgent` `my_world` leren met elk van de drie methoden, in je `GridWorld`, met de `episode` van de course: 10 runs van 300 episodes per methode, zoals in de vergelijking van het idee. Je `average` maakt de curves. Waar een deel van jou nog niet in je project staat, of de check nog niet haalt, neemt het deel van de course het over. De grafiek toont de gemiddelde return van de laatste 10 episodes over de 10 runs: SARSA doorgetrokken, Q-learning gestreept, en Monte Carlo lichter en gestreept; een return onder −150 blijft op de onderrand. Ernaast zie je hoe de run met seed 0 van elke methode er nu bij staat: de beste actions als kleine pijlen, en in <span class="c-param">oranje</span> de weg waar de pijlen vanaf S heen leiden, zodra de pijlen naar G leiden. Als de run klaar is, kun je teruggaan door de episodes.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    run_button = mo.ui.run_button(label="Laat mijn agent mijn wereld leren")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def demo_run(my_world, result_agent, result_world, run_button):
    result_world, result_agent   # opnieuw leren met je bestanden zodra een check je bestanden bewaart
    learned = None
    if not run_button.value:
        _out = mo.md("*Druk op de knop om je agent je wereld te laten leren.*")
    elif C.problems(my_world):
        _out = mo.md("*De demo kan je wereld nog niet gebruiken: de afbeelding onder je wereld zegt waarom.*")
    else:
        _w = C.DEMO_TEXT[L.lang]
        _now = {"shown": 0}

        def _progress(_n, _r):
            if "r" not in _now:   # vanaf de eerste episode de beste actions van seed 0 naast de grafiek
                _now["chart"].watch(lambda: C.now_picture(_now["r"], _now["n"], L.lang))
            _now["r"], _now["n"] = _r, _n
            _now["chart"].reference(C.chart_points(_r, "q", 1, _n), "", role="ink", replace=True)
            _now["chart"].reference(C.chart_points(_r, "mc", 1, _n), "", role="ink-3", replace=True)
            _now["chart"].extend(C.chart_points(_r, "sarsa", _now["shown"] + 1, _n),
                                 _w["progress"].format(n=_n, total=C.EPISODES))
            _now["shown"] = _n

        _live = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"], "alt": _w["alt"]}, C.EPISODES, role="ink",
                       low=C.LOW, high=0.0, window=1, key=C.curve_key(L.lang))
        _live.line("**Bezig met leren** …")
        _now["chart"] = _live
        _r = C.demo(my_world, show=_progress)
        if _r == "agent":
            _out = _live.stopped("*Je `GridWorld` of je `GridAgent` stopte tijdens de run met een fout, of gaf een return die geen getal is. Halen beide de check nog?*")
        elif _r == "average":
            _out = _live.stopped("*Je `average` stopte tijdens de run met een fout, of gaf iets anders dan een getal voor elke episode. Haalt je `average` de check nog in [Project: je eigen bandit-agent](/course/open/rl-project-bandit?lang=nl)?*")
        elif _r == "slow":
            _out = _live.stopped("*De run duurde langer dan twee minuten. Doen je `step` en je agent meer werk dan nodig?*")
        else:
            learned = _r
            _out = mo.vstack([
                L.explore("rewind", "episodes", lambda _v: C.rewind_frame(_r, int(_v), L.lang), start=C.EPISODES,
                          values=C.LOOKS, role="",
                          task="Met de wereld van de course: ga terug naar 50, en dan weer vooruit. Wanneer leiden de pijlen van elke methode voor het eerst van S naar G, en welke weg nemen de pijlen van SARSA bij 100, en bij 300?"),
                mo.md(C.summary(_r, L.lang)),
            ])
    _out
    return (learned,)


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met de wereld en de delen van de course heeft SARSA over de laatste 50 episodes een gemiddelde return van −16,7, Q-learning −30,4 en Monte Carlo −42,0. Aan het eind leiden de pijlen van SARSA over de bovenste rij, in 11 stappen, in alle 10 runs; de pijlen van Q-learning leiden tussen de plassen en de putten door, in 7 stappen. De pijlen van Q-learning tonen de kortste weg, maar de agent volgt de pijlen met ε = 0,1: één op de tien stappen is willekeurig, en naast de putten kost een willekeurige stap omlaag −100. SARSA leert wat de eigen willekeurige stappen kosten, en leert zo uit de buurt van de putten te blijven, net als op het klif. Monte Carlo leert het traagst: over alle 300 episodes is de gemiddelde return −179,9, tegen −29,9 voor SARSA en −37,3 voor Q-learning. Van de 3.000 episodes van Monte Carlo stoppen er 719 na 200 stappen zonder G te halen; SARSA haalt G in 2.997 episodes, Q-learning in 2.995. Bij het teruggaan door de episodes leiden de pijlen van Monte Carlo met seed 0 wel als eerste naar G, bij 50 episodes: seed 0 is een van de beste runs van Monte Carlo, zoals de NumPy-cel hieronder laat zien.
    """)
    return


@app.cell(hide_code=True)
def text_numpy():
    mo.md(r"""
    ### Alle runs tegelijk met NumPy

    Het gemiddelde over 10 runs verbergt hoe ver de runs uit elkaar liggen. `C.course_run()` geeft de run van de course op de eigen wereld; onder `["returns"]` staan, voor elke methode, de returns van elke episode van elke run. `np.array` maakt van de returns van Monte Carlo en van Q-learning arrays met vorm `(10, 300)`: een rij per run, een kolom per episode. `_mc[:, 250:]` neemt de laatste 50 episodes van elke run, en `np.mean(..., axis=1)` middelt elke rij: één getal per run. `_last_mc > _last_q` is `True` voor de runs waarin Monte Carlo voorstaat, en `.sum()` telt deze runs. Voer de cel uit.
    """)
    return


@app.cell
def example_numpy():
    import numpy as np

    _r = C.course_run()                      # de run van de course op de eigen wereld
    _mc = np.array(_r["returns"]["mc"])      # vorm (10, 300): een rij per run, een kolom per episode
    _q = np.array(_r["returns"]["q"])
    _last_mc = np.mean(_mc[:, 250:], axis=1)   # de gemiddelde return van de laatste 50 episodes, één getal per run
    _last_q = np.mean(_q[:, 250:], axis=1)
    print("vorm:", _mc.shape)
    print("Monte Carlo, per run:", np.round(_last_mc, 1))
    print("Q-learning, per run: ", np.round(_last_q, 1))
    print("runs waarin Monte Carlo voorstaat:", int((_last_mc > _last_q).sum()))
    print("gemiddelde over de runs, Monte Carlo:", round(float(np.mean(_last_mc)), 1), "Q-learning:", round(float(np.mean(_last_q)), 1))
    return


@app.cell(hide_code=True)
def text_after_numpy():
    mo.vstack([
        mo.md(r"""
        De runs van Monte Carlo liggen ver uit elkaar: van −20,0 tot −89,6 over de laatste 50 episodes. De runs van Q-learning liggen tussen −19,8 en −38,8. Over de 10 runs ligt Monte Carlo ver achter, −42,0 tegen −30,4, maar in 4 van de 10 runs staat Monte Carlo voor.
        """),
        L.watch_out("Eén run kan een ander verhaal vertellen dan tien. Met alleen seed 0 krijgt Monte Carlo −20,9 over de laatste 50 episodes en Q-learning −29,4: Monte Carlo lijkt beter. Vergelijk methoden over veel runs, en kijk hoe ver de runs uit elkaar liggen."),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_maze():
    mo.md(r"""
    ### Oefening 4. Een doolhof

    Zet dit doolhof in `my_world`, voer de cel uit, en laat je agent het doolhof leren met de knop:

    ```python
    my_world = [
        "S..#.....",
        ".#.#.###.",
        ".#...#...",
        ".####.#X.",
        "......#.G",
    ]
    ```

    De kortste weg van S naar G kost 16 stappen. Met de delen van de course heeft Q-learning over de laatste 50 episodes een gemiddelde return van −21,3 en SARSA −53,9, en de pijlen van Q-learning leiden in alle 10 runs langs de kortste weg. Q-learning haalt G in 2.889 van de 3.000 episodes, SARSA in 2.730; aan het eind leiden de pijlen van SARSA in 3 van de 10 runs naar G. Monte Carlo haalt G in 9 van de 3.000 episodes: bijna elke episode van Monte Carlo stopt na 200 stappen, en de gemiddelde return over de laatste 50 is −200,1. In de wereld van de course leerde Monte Carlo wel.
    """)
    return


@app.cell(hide_code=True)
def choice_maze_cell():
    choice_maze = L.choice(
        "maze",
        question="Waarom haalt Monte Carlo in het doolhof bijna nooit G, terwijl SARSA en Q-learning in de meeste episodes G halen?",
        options={
            "luck": "Pech: met andere seeds zou Monte Carlo het doolhof ook leren",
            "model": "Monte Carlo heeft een model van de environment nodig om te leren waar de muren staan, en SARSA en Q-learning niet",
            "right": "Monte Carlo verandert tijdens een episode geen enkele $Q$, dus kan Monte Carlo in dezelfde vakjes steeds dezelfde stappen zetten, tot de limiet; SARSA en Q-learning verlagen de $Q$ van elke stap meteen, en stappen zo over op andere actions",
            "long": "Met $\\gamma$ = 1 telt de return van een stap elke reward erna op, tot 200 stuks; zulke lange totalen lopen te ver uiteen om een doolhof van te leren, terwijl de targets van SARSA en Q-learning slechts één stap vooruitkijken en veel minder uiteenlopen",
        },
        correct="right",
        why={
            "right": "In de eerste episode is elke $Q$ 0, en bij gelijke waarden neemt de agent de eerste action, up. Vanaf S is dat de rand, dus loopt Monte Carlo in de eerste episode steeds tegen de rand, met één op de tien keer een willekeurige stap, tot de limiet. Ook later krijgt een vakje waar de agent in een episode terugkomt opnieuw dezelfde action, dus loopt Monte Carlo rondjes. SARSA en Q-learning verlagen $Q(S, A)$ na elke stap, dus lijkt een stap die de agent net zette slechter dan een stap die de agent nog niet probeerde: binnen één episode gaan SARSA en Q-learning verder, en vinden G. De wereld van de course heeft geen muren en een korte weg naar G, dus daar halen de willekeurige stappen van Monte Carlo G vaak genoeg.",
            "luck": "Geen van de 10 runs leert het doolhof: samen halen de runs G in 9 van de 3.000 episodes. Wat verandert er in de $Q$-waarden van Monte Carlo terwijl een episode nog bezig is?",
            "model": "Monte Carlo control is model-free, net als SARSA en Q-learning: alle drie leren alleen van stappen die de agent echt gezet heeft. Wat doet elke methode met een stap op het moment dat de agent de stap zet?",
            "long": "De returns lopen alleen uiteen door de eigen willekeurige stappen van de agent, en in de wereld van de course leert Monte Carlo ook van returns van wel 200 rewards. Wat verandert er in de $Q$-waarden van Monte Carlo terwijl een episode nog bezig is?",
        },
    )
    choice_maze
    return (choice_maze,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Voorzichtig of gedurfd: SARSA en Q-learning](/course/open/rl-sarsa?lang=nl) en [Beter spelen door ervaring: Monte Carlo control](/course/open/rl-monte-carlo-control?lang=nl) leerde de code van de course de werelden van de course. Nu heeft je project `agent` een wereld die je zelf ontwerpt, één agent die de wereld leert met elk van de drie targets, en learning curves over veel runs, naast je bandit-agent en je `average`. Je vergelijking liet zien dat SARSA naast putten het best werkt tijdens het leren, dat één run een ander verhaal kan vertellen dan tien, en dat Monte Carlo in een doolhof kan verdwalen, omdat Monte Carlo tijdens een episode niets verandert. Later geeft *Eén interface voor elke environment: Gymnasium* veel environments dezelfde vorm als je `GridWorld`, met `reset` en `step`. De volgende les, *Een expert nadoen: behavior cloning*, begint het deel *Andere settings*: een agent die leert van de stappen van een expert, zonder rewards.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_maze, result_agent, result_curves, result_world):
    L.wrap_up(result_world, result_agent, result_curves, choice_maze)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
