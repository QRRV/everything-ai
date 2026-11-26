# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-project-bandit. Route: course/route/lessen/reinforcement-learning.yaml. Checks, de agent van de course, de demo en de afbeeldingen: checks.py; het testbed dat de course in je project zet: project/agent/testbed.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("rl-project-bandit", "nl")
    C = L.checks   # de agent, de loop en het gemiddelde van de course, de demo en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Gemiddelden bijhouden: verschuivende automaten en optimistische starts](/course/open/rl-bandit-updates?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_drift():
    quick_drift = L.choice(
        "quick-drift",
        question="De gemiddelden van een bandit verschuiven na elke trek een beetje. Welke learning rate houdt een schatting dicht bij wat de automaat nu geeft?",
        options={
            "average": "1/n, het steekproefgemiddelde, dat elke reward gebruikt",
            "right": "Een vaste learning rate, zoals η = 0,1",
            "zero": "η = 0: de schatting blijft staan",
        },
        correct="right",
        why={
            "right": "Met een vaste η vermenigvuldigt elke update het aandeel van elke oudere reward met 1 − η, dus tellen recente rewards zwaarder, en volgt de schatting een gemiddelde dat verschuift.",
            "average": "Na 1.000 rewards van een automaat schuift 1/n de schatting slechts een duizendste van de weg op naar een nieuwe reward: hoe snel volgt de schatting zo een gemiddelde dat verschuift? Zie [Gemiddelden bijhouden: verschuivende automaten en optimistische starts](/course/open/rl-bandit-updates?lang=nl).",
            "zero": "Met η = 0 schuift elke update de schatting een deel 0 van de weg op naar de reward: kan de schatting dan leren wat de automaat nu geeft? Zie [Gemiddelden bijhouden: verschuivende automaten en optimistische starts](/course/open/rl-bandit-updates?lang=nl).",
        },
    )
    quick_drift
    return (quick_drift,)


@app.cell(hide_code=True)
def quick_check_optimistic():
    quick_optimistic = L.choice(
        "quick-optimistic",
        question="Een greedy agent met η = 0,1 laat de schattingen van de tien automaten beginnen bij 5, ver boven elke reward. Wat doet de agent bij de eerste tien keer trekken?",
        options={
            "same": "Tien keer automaat 0, want alle schattingen zijn gelijk",
            "right": "De agent probeert elke automaat één keer, om de beurt",
            "random": "Tien willekeurige automaten, zoals ε-greedy",
        },
        correct="right",
        why={
            "right": "Elke reward valt tegen: een eerste reward $r$ zet de schatting van de automaat op 5 + 0,1 · ($r$ − 5) = 4,5 + 0,1$r$, onder de automaten die nog op 5 staan. Dus gaat de greedy agent door naar de volgende automaat, tot alle automaten geprobeerd zijn.",
            "same": "Automaat 0 is de eerste, want de schattingen zijn gelijk. Waar staat de schatting van automaat 0 na de eerste reward, tegenover de negen automaten die nog op 5 staan? Zie [Gemiddelden bijhouden: verschuivende automaten en optimistische starts](/course/open/rl-bandit-updates?lang=nl).",
            "random": "Een greedy agent trekt nooit aan een willekeurige automaat: welke automaat heeft na de eerste reward de grootste schatting? Zie [Gemiddelden bijhouden: verschuivende automaten en optimistische starts](/course/open/rl-bandit-updates?lang=nl).",
        },
    )
    quick_optimistic
    return (quick_optimistic,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        In [Welke automaat geeft het meest? De multi-armed bandit](/course/open/rl-bandits?lang=nl) en [Gemiddelden bijhouden: verschuivende automaten en optimistische starts](/course/open/rl-bandit-updates?lang=nl) schreef de course de agents en liet de course de agents voor je spelen. In dit project bouw je een eigen agent, in je eigen projectmap, in de vorm waarin agents in de praktijk gebouwd worden: een object dat kiest en leert, een omgeving die antwoordt, en een loop tussen de twee. Dezelfde drie delen komen terug in elke les van deze course waarin een agent leert van ervaring, ook als de omgeving een grid world of een spel is.

        Een agent bouwen is de helft van het werk. De andere helft is uitzoeken of de agent iets waard is: beter dan een andere agent, bij welke problemen, en volgens welke maat. Een vergelijking op verschillende problemen, of op te weinig runs, kan de slechtere agent beter laten lijken. Dus krijgt je project ook code om mee te meten.

        **Een voorbeeld uit het dagelijks leven.** Drie fietsers willen weten wie van de drie het snelst rijdt. Elke fietser rijdt één keer, op een andere dag, op een andere weg, bij andere wind. Zegt de snelste tijd wie het snelst rijdt? Op een fietstrainer, met dezelfde route en dezelfde weerstand voor alle drie, leren de fietsers meer, en nog meer als elke fietser de route vaak rijdt.
        """),
        draw.project_steps(["Je bandit-agent", "Je grid-agent"], 0,
                           here_label="← vandaag", title="Waar je project staat: agent"),
    ])
    return


@app.cell(hide_code=True)
def predict_reward():
    guess_reward = L.guess(
        "reward",
        question="In deze les spelen drie versies van je agent op dezelfde 200 verschuivende bandits, elk 10.000 keer trekken, allemaal met learning rate 0,1: greedy met elke schatting vanaf 0, ε-greedy met $\\varepsilon$ = 0,1 vanaf 0, en greedy met elke schatting vanaf 5. Welke versie krijgt de meeste reward over alle 10.000 keer trekken, denk je?",
        options={
            "greedy": "Greedy, vanaf 0",
            "epsilon": "ε-greedy, vanaf 0",
            "right": "Optimistisch, vanaf 5",
            "same": "Alle drie ongeveer evenveel",
        },
        reveal="**Optimistisch, vanaf 5.** Over alle 10.000 keer trekken krijgt de optimistische agent 1,77 per trek, ε-greedy 1,66 en greedy 1,55. Over de eerste 100 keer trekken is de optimistische agent veruit de laatste, met 0,35 tegen 0,92 en 0,97. De demo laat je eigen agents dezelfde run spelen.",
    )
    guess_reward
    return (guess_reward,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Twee objecten.** In dit project zijn de bandit en de agent elk een object van een class, in een bestand van je project, zoals in [Je eigen module: projectcode in een bestand bewaren](/course/open/py-own-modules?lang=nl).

        - De omgeving is een `DriftingBandit`, uit `work/projects/agent/testbed.py`, dat de course in je project zette. `DriftingBandit(k, drift, rng)` maakt een bandit van `k` automaten met gemiddelden rond 0 aan het begin, met een standaarddeviatie van 1. `pull(action)` geeft de reward van één trek aan automaat `action`, het gemiddelde plus ruis, en laat daarna elk gemiddelde een willekeurige stap zetten met standaarddeviatie `drift`, zoals in [Gemiddelden bijhouden: verschuivende automaten en optimistische starts](/course/open/rl-bandit-updates?lang=nl). `best()` geeft de automaat met het hoogste gemiddelde van dat moment. Alleen het meten gebruikt `best()`: de agent ziet de gemiddelden nooit.
        - De agent is je `BanditAgent`, in `work/projects/agent/bandit.py`, dat je in oefening 1 schrijft. `act()` kiest een automaat, ε-greedy; `learn(action, reward)` telt de reward en schuift de schatting van de automaat ernaar op, met de update van [Gemiddelden bijhouden: verschuivende automaten en optimistische starts](/course/open/rl-bandit-updates?lang=nl). Wat de agent weet, een lijst `q` met schattingen en een lijst `counts`, houdt de agent zelf bij, van de ene trek naar de volgende.

        **Eén loop.** Tussen de twee draait een loop, je `run` uit oefening 2: de agent kiest een automaat, de bandit antwoordt met een reward, de agent leert ervan, en zo verder, trek na trek. Het is de loop van agent en omgeving uit [Leren van rewards](/course/open/rl-what-is-rl?lang=nl). Een bandit heeft één state, dus is de reward alles wat de agent waarneemt. De afbeelding toont één beurt van de loop.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        De agent trok al 53 keer aan deze bandit en probeerde 9 van de 10 automaten. Bij trek 54 kiest de agent automaat 7, met de grootste schatting: 0,97, na 40 keer trekken. De beste automaat op dat moment is automaat 6, met een gemiddelde van 0,87 tegen 0,80 voor automaat 7, maar de twee keer trekken aan automaat 6 gaven tot nu toe weinig, dus schat de agent automaat 6 op −0,02. Automaat 7 geeft deze keer −1,97, en `learn` schuift de schatting een tiende van de weg omlaag, naar 0,68.

        **Een zuivere vergelijking.** Eén trek zegt weinig, en één run op één bandit niet veel meer: in het uitgewerkte voorbeeld hieronder geeft een automaat met een gemiddelde van 0,13 bij één trek 1,49. Dus laat je project elke agent op 200 bandits spelen en neemt het project bij elke trek het gemiddelde over de runs, zoals in de learning curves van [Welke automaat geeft het meest? De multi-armed bandit](/course/open/rl-bandits?lang=nl). En het project geeft elke agent dezelfde 200 bandits. Bandit `b` wordt voor elke agent gemaakt met seed `b`, dus begint de bandit voor elke agent met dezelfde gemiddelden. Een bandit haalt de ruis en de verschuiving uit een eigen random number generator, in dezelfde volgorde, aan welke automaat de agent ook trekt, dus verschuiven de gemiddelden voor elke agent op dezelfde manier, en geeft dezelfde automaat bij dezelfde trek hetzelfde. Een verschil tussen de agents komt dan van de agents, en niet van bandits die toevallig makkelijker waren.

        **Twee maten.** Je project meet elke agent op twee manieren, bij elke trek: de **gemiddelde reward**, over de bandits, en het **aandeel beste keuzes**: bij welk deel van de bandits de agent de beste automaat van dat moment koos. Vaak zijn de twee het eens. Oefening 4 gaat over een geval waarin dat niet zo is.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **testbed** | | een vaste set problemen om methoden op te vergelijken, zoals de verschuivende bandits van de course | `testbed.py` |
        | **run** | | één agent die op één bandit speelt, van de eerste trek tot de laatste | `run` |
        | **gemiddelde reward** | | de reward bij één trek, gemiddeld over de bandits | `rewards`, gemiddeld over de runs |
        | **aandeel beste keuzes** | | bij welk deel van de bandits de agent de beste automaat van dat moment koos | `best`, gemiddeld over de runs |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een eigen agent op een testbed:

    1. Schrijf de agent als class: in `__init__` wat de agent weet, een schatting en een telling voor elke automaat; in `act` de keuze; in `learn` wat de agent met een reward doet.
    2. Schrijf de loop: bij elke trek kiest de agent, noteer je of de agent de beste automaat koos, betaalt de bandit, leert de agent, en bewaar je de reward.
    3. Maak veel bandits, dezelfde voor elke agent die je vergelijkt, en geef elke agent een eigen kopie van elke bandit.
    4. Laat elke agent met je loop op elke bandit spelen, en bewaar de rewards en de beste keuzes.
    5. Middel bij elke trek over de runs: de gemiddelde reward en het aandeel beste keuzes.
    6. Vergelijk de agents op beide maten, over de hele run en over delen ervan.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel importeert `DriftingBandit` uit je project, maakt een bandit met seed 0, en toont de gemiddelden en de beste automaat. Dan trekt de cel om de beurt aan de beste automaat en aan automaat 0, elk drie keer, en trekt nog 10.000 keer voordat de cel de gemiddelden opnieuw toont. Voer de cel uit.
    """)
    return


@app.cell
def example_testbed():
    from agent.testbed import DriftingBandit

    # een bandit van 10 automaten waarvan de gemiddelden bij elke trek verschuiven met standaarddeviatie 0.01, gemaakt met seed 0
    _bandit = DriftingBandit(10, 0.01, np.random.default_rng(0))
    print("gemiddelden:", np.round(_bandit.means, 2))
    _b = _bandit.best()
    print("beste:", _b)
    # drie beurten: een trek aan de beste automaat, dan een aan automaat 0
    for _t in range(3):
        print("automaat", _b, "geeft", round(_bandit.pull(_b), 2), "en automaat 0 geeft", round(_bandit.pull(0), 2))
    # 10.000 keer trekken later is elk gemiddelde verschoven
    for _t in range(10_000):
        _bandit.pull(0)
    print("gemiddelden:", np.round(_bandit.means, 2))
    print("beste:", _bandit.best())
    return (DriftingBandit,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Aan het begin heeft automaat 6 het hoogste gemiddelde, 1,30, en automaat 7 komt daarna met 0,95. Eén trek zegt weinig: automaat 0, met een gemiddelde van 0,13, geeft bij de eerste beurt 1,49, meer dan twee keer wat automaat 6 dan geeft, 0,68. Na nog 10.000 keer trekken is elk gemiddelde verschoven: automaat 7 ging van 0,95 naar 0,21, automaat 4 van −0,54 naar −1,17. Hier is automaat 6 nog steeds de beste, nu met 1,57, maar bij 115 van de 200 bandits van de course is de beste automaat bij trek 10.000 een andere dan aan het begin.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_agent():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Je BanditAgent

        `BanditAgent(k, epsilon, eta, start, rng)` is een agent voor een bandit van `k` automaten; bovenaan staat `import numpy as np`. `__init__` is gegeven en bewaart `epsilon`, de learning rate `eta`, de random number generator `rng`, een lijst `q` met de schatting `start` voor elke automaat, en een lijst `counts` met nullen. Als `eta` `None` is, neemt de agent de learning rate 1/n, het steekproefgemiddelde. In `act` is de willekeurige keuze met kans `epsilon` gegeven, met `self.rng.random()` en `self.rng.integers`.

        Vervang de twee `...`. In `act`: de automaat met de grootste schatting, de eerste als er meerdere gelijk zijn, als heel getal. In `learn(action, reward)`: tel 1 op bij de telling van `action`, neem de learning rate, en schuif de schatting van `action` op naar `reward`.

        De check maakt `BanditAgent(3, 0.0, 0.5, 0.0, np.random.default_rng(0))`, geeft de agent de rewards 2,0 en 4,0 voor automaat 1 en 1,0 voor automaat 2, en kijkt naar `agent.q`, `agent.counts` en `agent.act()`: dat moeten `[0.0, 2.5, 0.5]`, `[0, 2, 1]` en `1` zijn. Daarna probeert de check drie eigen gevallen: `eta` None, een optimistische start, en een agent die de helft van de tijd verkent. Als je class klopt, slaat de check de class op in je project, als `work/projects/agent/bandit.py`.
        """),
        L.task("Laat `act` de automaat met de grootste schatting kiezen, en `learn` de reward tellen en de schatting ernaar opschuiven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_agent():
    box_agent = L.code_box("agent", lines=26)
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
            "`act` geeft de plaats van de grootste waarde in `self.q` terug, niet de waarde zelf, als gewoon heel getal. `learn` telt 1 op bij `self.counts[action]`, geeft een naam `eta` de learning rate, `1 / self.counts[action]` of `self.eta`, en schuift `self.q[action]` een deel `eta` van de weg op naar `reward`.",
            "`np.argmax` vindt de plaats, de eerste als er meerdere gelijk zijn, en `int` maakt er een gewoon heel getal van. In het voorbeeld schuift `learn(1, 2.0)` met learning rate 0,5 de schatting van automaat 1 halverwege van 0,0 naar 2,0, dus naar 1,0; `learn(1, 4.0)` schuift de schatting halverwege van 1,0 naar 4,0, dus naar 2,5. Tel voordat je 1/n neemt: de eerste reward van een automaat heeft n = 1.",
            "`        self.q[action] = self.q[action] + eta * ___`",
        ],
        why="In `act`: `return int(np.argmax(self.q))`. In `learn`: `self.counts[action] = self.counts[action] + 1`; dan `eta = 1 / self.counts[action]` als `self.eta is None`, anders `eta = self.eta`; en `self.q[action] = self.q[action] + eta * (reward - self.q[action])`, de update van [Gemiddelden bijhouden: verschuivende automaten en optimistische starts](/course/open/rl-bandit-updates?lang=nl), nu bewaard in het object.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_run(result_agent):
    result_agent  # opnieuw gemaakt na de check van oefening 1: de check slaat je agent op
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De loop

        `run(agent, bandit, pulls)` laat `agent` en `bandit` `pulls` keer om de beurt aan zet zijn; `for t in range(pulls)` telt de beurten. De functie geeft twee lijsten terug met een getal voor elke trek: `rewards`, de reward van de trek, en `best`, 1 als de agent de beste automaat van dat moment koos en anders 0. Vervang de `...` door één beurt: de agent kiest een automaat, `best` krijgt 1 of 0, de bandit betaalt, de agent leert van de reward, en `rewards` bewaart de reward. Vraag de bandit naar de beste automaat voordat de trek komt: `pull` laat elk gemiddelde verschuiven.

        De check importeert je `BanditAgent` en de `DriftingBandit` van de course uit je project, en laat een greedy agent met learning rate 0,5 vier keer trekken aan een bandit van 3 automaten zonder verschuiving. De rewards, op drie decimalen, `best` en de schattingen moeten `[-2.252, 0.251, -0.833, -0.512]`, `[1, 0, 0, 0]` en `[-1.126, -0.354, -0.256]` zijn: automaat 0 was de beste, maar de eerste reward, −2,252, liet de greedy agent de automaat verlaten. Daarna probeert de check drie eigen gevallen: een tweede aanroep die verdergaat waar de eerste stopte, gemiddelden die sterk verschuiven, en 0 keer trekken. Als je `run` klopt, slaat de check de functie op als `work/projects/agent/run.py`.
        """),
        L.course_versions(),
        L.task("Laat in `run` de agent en de bandit om de beurt spelen, en bewaar de reward en of de agent de beste automaat koos."),
    ])
    return


@app.cell(hide_code=True)
def exercise_run():
    box_run = L.code_box("run", lines=16)
    box_run
    return (box_run,)


@app.cell(hide_code=True)
def check_run(box_run, result_agent):
    result_agent  # check opnieuw als oefening 1 je agent opslaat
    result_run = L.check("run", box_run.value)
    result_run
    return (result_run,)


@app.cell(hide_code=True)
def hints_run():
    L.hints(
        "run",
        [
            "Eén beurt, in de volgorde van stap 2 van het recept: `agent.act()` geeft de automaat, `bandit.best()` zegt of het de beste was, `bandit.pull` geeft de `reward`, `agent.learn` neemt de reward aan, en `append` voegt een getal toe aan `best` en aan `rewards`.",
            "In het voorbeeld kiest de eerste beurt automaat 0, de beste, dus krijgt `best` een 1. De bandit geeft −2,252, en de agent schuift de schatting van automaat 0 halverwege op, naar −1,126. Bij de tweede beurt delen automaat 1 en 2 de grootste schatting, 0, en `act` neemt de eerste van de twee.",
            "`        if ___:`",
        ],
        why="`action = agent.act()`, dan `if action == bandit.best():` met `best.append(1)` en `else:` met `best.append(0)`, dan `reward = bandit.pull(action)`, `agent.learn(action, reward)` en `rewards.append(reward)`. `best.append(int(action == bandit.best()))` doet hetzelfde in één regel.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Laat drie van je agents spelen

    De knop laat drie versies van je `BanditAgent` met je `run` op dezelfde 200 verschuivende bandits spelen, elk 10.000 keer trekken, allemaal met learning rate 0,1: greedy vanaf 0, ε-greedy met $\varepsilon$ = 0,1 vanaf 0, en greedy met elke schatting vanaf 5, de optimistische agent. Zolang je `bandit.py` of `run.py` er nog niet is, of je `bandit.py` niet geeft wat de course verwacht, springt de versie van de course in. De demo gaat zo:

    ```python
    for name in ["greedy", "epsilon", "optimistic"]:
        for b in range(200):   # bandit b krijgt seed b, voor elke agent
            agents[name][b] = BanditAgent(10, epsilon[name], 0.1, start[name], np.random.default_rng(1000 + b))
            bandits[name][b] = DriftingBandit(10, 0.01, np.random.default_rng(b))
    for t in range(0, 10000, 100):   # 100 keer trekken tegelijk, zodat de grafiek kan groeien
        for name in ["greedy", "epsilon", "optimistic"]:
            for b in range(200):
                rewards, best = run(agents[name][b], bandits[name][b], 100)
    ```

    De grafiek toont de gemiddelde reward van de ε-greedy agent over de 200 bandits: dun over elke 10 keer trekken, dik het gemiddelde over de laatste 100 keer trekken; gestreept hetzelfde gemiddelde voor de optimistische agent en, lichter, voor de greedy agent. Ernaast zie je wat de drie agents weten van één van de 200 bandits: de gemiddelden van dat moment als korte streepjes, de schattingen in <span class="c-param">oranje</span>, open bij een automaat waaraan nooit getrokken is, en hoe vaak aan elke automaat getrokken is. Het duurt ongeveer een halve minuut; daarna kun je door de run teruggaan.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    run_button = mo.ui.run_button(label="Laat mijn drie agents spelen")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def demo_run(result_agent, result_run, run_button):
    result_agent, result_run   # speel opnieuw met je bestanden zodra een check de bestanden opslaat
    trained = None
    if not run_button.value:
        _out = mo.md("*Druk op de knop om de drie agents te laten spelen.*")
    else:
        _w = C.live_words(L.lang)
        _now = {}   # de live grafiek, wat de agents weten van één bandit na de laatste 100 keer trekken, en bij welke trek
        _raw = {_a: [] for _a in C.AGENTS}   # de gemiddelde reward over de bandits van elke agent bij elke trek

        def _progress(_t, _means, _seen):
            if "seen" not in _now:   # vanaf de eerste 100 keer trekken: wat de agents weten, naast de grafiek
                _now["chart"].watch(lambda: C.now_picture(_now["seen"], _now["t"], L.lang))
            _now["seen"], _now["t"] = _seen, _t
            for _a in C.AGENTS:
                _raw[_a].extend(_means[_a])
            if _t % 500 == 0:
                _now["chart"].reference(C.smoothed(_raw["optimistic"]), "", role="ink", replace=True)
                _now["chart"].reference(C.smoothed(_raw["greedy"]), "", role="ink-3", replace=True)
            _now["chart"].extend([(_u / 1000, C.mean_of(_raw["epsilon"], _u - 9, _u)) for _u in range(_t - 90, _t + 1, 10)],
                                 C.progress(_t, L.lang))

        _now["chart"] = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"], "alt": _w["alt"]},
                               C.PULLS / 1000, role="ink", low=-0.5, high=2.5, window=10, key=C.live_key(L.lang))
        _now["chart"].line("**Spelen** …")
        _run = C.demo(show=_progress)
        if _run == "run":
            _out = _now["chart"].stopped("*Je `work/projects/agent/run.py` stopt met een fout, of heeft geen `run`. Komt je `run` nog door de check?*")
        elif _run == "stopped":
            _out = _now["chart"].stopped("*Je `BanditAgent` of je `run` stopte met een fout tijdens de run. Komen de twee nog door de check?*")
        elif _run == "wrong":
            _out = _now["chart"].stopped("*Je `run` gaf iets anders terug dan twee lijsten met een getal voor elke trek. Komt je `run` nog door de check?*")
        elif _run == "slow":
            _out = _now["chart"].stopped("*De run duurde langer dan tweeënhalve minuut. Doen je `act`, `learn` en `run` meer werk dan nodig?*")
        else:
            trained = _run
            _who = {("bandit", "run"): "je `BanditAgent` en je `run`",
                    ("bandit",): "je `BanditAgent` en de `run` van de course, want je `run.py` is er nog niet",
                    ("run",): "je `run` en de `BanditAgent` van de course, want je `bandit.py` is er nog niet of geeft niet wat de course verwacht",
                    (): "de `BanditAgent` en de `run` van de course, want de jouwe staan nog niet in je project"}[tuple(_run["yours"])]
            _all = {_a: float(_run["rewards"][_a].mean()) for _a in C.AGENTS}
            _out = mo.vstack([
                L.explore("rewind", "trek", lambda _v: (C.rewind_frame(_run, int(_v), L.lang), C.rewind_text(_run, int(_v), L.lang)),
                          start=C.PULLS, values=C.LOOKS, role="",
                          task="Ga terug naar trek 100, en dan weer vooruit: vanaf welke trek is automaat 7 de beste automaat van deze bandit, en welke agent trekt er vanaf dan aan?"),
                mo.md(f"Gespeeld met {_who}. Over alle 10.000 keer trekken was de gemiddelde reward per trek {C.number(_all['greedy'], 2, L.lang)} bij greedy, {C.number(_all['epsilon'], 2, L.lang)} bij ε-greedy en {C.number(_all['optimistic'], 2, L.lang)} bij de optimistische agent."),
            ])
    _out
    return (trained,)


@app.cell(hide_code=True)
def text_after_demo():
    mo.vstack([
        mo.md(r"""
        Met de agents van de course verandert de volgorde tijdens de run. Over de eerste 100 keer trekken is de optimistische agent veruit de laatste, met 0,35 per trek tegen 0,92 voor ε-greedy en 0,97 voor greedy: de agent besteedt de eerste keren aan het proberen van elke automaat, ook de slechte. Vanaf ongeveer trek 240 ligt de optimistische agent voor op allebei, en blijft de agent voor tot het eind. Over alle 10.000 keer trekken krijgt de optimistische agent 1,77 per trek, ε-greedy 1,66 en greedy 1,55.

        De bandit naast de grafiek, één van de 200, laat zien wat de optimistische agent mist. Bij trek 100 heeft automaat 7 een gemiddelde van −0,30. De optimistische agent probeert de automaat 15 keer in de eerste 223 keer trekken, terwijl het gemiddelde onder 0 blijft, en daarna nog slechts 6 keer, de laatste keer bij trek 2.792. Dan schuift automaat 7 omhoog: bij trek 8.000 is automaat 7 de beste automaat van de bandit, en aan het eind is het gemiddelde 2,40. De ε-greedy agent blijft af en toe aan een willekeurige automaat trekken en ontdekt dat: aan het eind heeft de agent in totaal 2.263 keer aan automaat 7 getrokken, met een schatting van 2,46. De optimistische agent schat automaat 7 nog steeds op 0,53, en de greedy agent trok er nooit aan; de twee eindigen allebei op automaat 6, met een gemiddelde van 0,96. Een hoge start laat een agent alleen aan het begin verkennen: zodra elke schatting gezakt is, is de optimistische agent net zo greedy als de greedy agent, en blijft een automaat die later beter wordt onopgemerkt.
        """),
        L.watch_out("Op een paar bandits beslist geluk de volgorde. Op de eerste van de 200 bandits alleen krijgt greedy meer dan ε-greedy over de 10.000 keer trekken, 1,82 per trek tegen 1,65, en op de eerste drie samen nog 1,80 tegen 1,66; over alle 200 is het andersom. Per bandit krijgt de optimistische agent het meest op 114 van de 200, greedy op 64 en ε-greedy op slechts 22: greedy ligt op veel bandits voor, maar op andere ver achter."),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_average():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Het gemiddelde over veel runs

        `average(runs)` krijgt een lijst met een lijst voor elke run, allemaal even lang, met een getal voor elke trek. De functie geeft een lijst `means` terug met het gemiddelde over de runs bij elke trek: een learning curve. Je project gebruikt de functie voor beide maten: op de rewards geeft de functie de gemiddelde reward, op de enen en nullen van `best` het aandeel beste keuzes. Vervang de `...` door het gemiddelde bij trek `t`, toegevoegd aan `means`.

        `small_runs`, de `SMALL_RUNS` van de course, bevat drie runs van vier keer trekken, met een 1 waar de agent de beste automaat koos: `[[0, 1, 1, 1], [0, 0, 1, 1], [1, 0, 0, 1]]`. `average(small_runs)` geeft `[0.333, 0.333, 0.667, 1.0]`: bij de eerste trek koos één run van de drie de beste automaat. De check roept dat aan, dan vier eigen gevallen, met rewards, één enkele run en vier runs. Als je functie klopt, slaat de check de functie op als `work/projects/agent/measure.py`.
        """),
        L.task("Laat `average` het gemiddelde over de runs bij elke trek teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_average():
    box_average = L.code_box("average", lines=10)
    box_average
    return (box_average,)


@app.cell(hide_code=True)
def check_average(box_average):
    result_average = L.check("average", box_average.value)
    result_average
    return (result_average,)


@app.cell(hide_code=True)
def hints_average():
    L.hints(
        "average",
        [
            "Voor elke trek `t`: een `total` dat bij 0 begint, een `for`-loop over de runs, waarin je trek `t` van elke run bij het totaal optelt, en dan het gemiddelde over de runs, toegevoegd aan de lijst met `means.append`.",
            "In het voorbeeld hebben de drie runs bij de eerste trek 0, 0 en 1: een totaal van 1, gedeeld door de 3 runs, geeft 0,333. Bij de laatste trek hebben alle drie een 1, dus is het gemiddelde 3 / 3 = 1,0.",
            "`        means.append(___)`",
        ],
        why="`total = 0`, dan `for one in runs:` met `total = total + one[t]` eronder, en `means.append(total / len(runs))`. NumPy doet het in één regel: `np.mean(runs, axis=0)`, het gemiddelde langs as 0, over de runs.",
    )
    return


@app.cell(hide_code=True)
def text_project_demo():
    mo.md(r"""
    ### Wat je project nu kan

    Je project heeft nu alle delen: een agent, een loop en een maat. De slider schuift een venster van 1.000 keer trekken over de run van de demo hierboven, van de eerste 1.000 keer trekken tot de laatste 1.000. De twee grafieken tonen beide maten voor alle drie de agents, gemaakt met je `average` zodra je functie door de check komt, elk als het gemiddelde over de laatste 100 keer trekken. De band markeert het venster, en de zin geeft beide maten over het venster.
    """)
    return


@app.cell(hide_code=True)
def project_demo(box_average, result_average, trained):
    if trained is None:
        _out = mo.md("*Laat eerst de drie agents spelen, met de knop hierboven: de slider meet de run van de agents.*")
    else:
        _yours = C.mine(box_average.value, result_average.state == "correct")
        _curves = C.curves(trained, _yours or C.average)
        if _curves == "stopped":
            _out = mo.md("*Je `average` stopte met een fout op de runs van de demo. Komt je functie nog door de check?*")
        elif _curves == "wrong":
            _out = mo.md("*Je `average` gaf iets anders terug dan een lijst met een getal voor elke trek. Komt je functie nog door de check?*")
        elif _curves == "slow":
            _out = mo.md("*Je `average` deed langer dan 30 seconden over de zes curves. Doet je functie meer werk dan nodig?*")
        else:
            _who = "je `average`" if _yours else "de `average` van de course, want de jouwe komt nog niet door de check"
            _out = mo.vstack([
                L.explore("window", "laatste trek", lambda _v: C.window_frame(trained, _curves, int(_v), L.lang),
                          start=1000, values=C.WINDOWS, role="",
                          task="Schuif het venster van de eerste 1.000 keer trekken naar de laatste: vanaf welk venster kiest ε-greedy vaker de beste automaat dan de optimistische agent, en welke agent krijgt in elk venster de meeste reward?"),
                mo.md(f"Gemeten met {_who}."),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_project_demo():
    mo.md(r"""
    Over trek 1 tot en met 1.000 zetten beide maten de optimistische agent op één: 1,34 per trek en de beste automaat bij 68,1% van de bandits, tegen 1,27 en 61,0% voor ε-greedy en 1,19 en 43,6% voor greedy. Vanaf het venster van trek 4.001 tot en met 5.000 zijn de maten het oneens. De optimistische agent krijgt nog steeds in elk venster de meeste reward, maar ε-greedy kiest vaker de beste automaat: over de laatste 1.000 keer trekken bij 80,4% van de bandits, tegen 69,7%. Greedy is in elk venster de laatste op beide maten.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_measures():
    mo.md(r"""
    ### Oefening 4. Twee maten, twee volgordes

    Over de laatste 1.000 keer trekken van de demo kiest de ε-greedy agent de beste automaat bij 80,4% van de bandits, de optimistische agent bij 69,7%. Toch krijgt de optimistische agent meer per trek: 2,03 tegen 1,95. De course liet dezelfde drie agents op 2.000 andere verschuivende bandits spelen en vond hetzelfde: over de laatste 1.000 keer trekken kiest ε-greedy de beste automaat bij 80,4% van de bandits tegen 73,6%, en krijgt ε-greedy 1,91 per trek tegen 2,02.
    """)
    return


@app.cell(hide_code=True)
def choice_measures_cell():
    choice_measures = L.choice(
        "measures",
        question="Hoe krijgt de optimistische agent meer reward, terwijl de agent minder vaak de beste automaat kiest?",
        options={
            "easier": "De optimistische agent speelde toevallig op makkelijkere bandits",
            "right": "Als ε-greedy mist, is het vaak een willekeurige trek die veel kost; als de optimistische agent mist, trekt de agent meestal aan een automaat die bijna net zoveel geeft",
            "start": "De startwaarde 5 telt aan het eind nog mee: de schattingen van de optimistische agent blijven daardoor hoog, en hoge schattingen tellen bij elke trek op bij de rewards van de automaten",
            "luck": "Toeval: over 1.000 keer trekken beslist de ruis van de rewards",
        },
        correct="right",
        why={
            "right": "Het aandeel beste keuzes telt elke misser even zwaar, de reward niet. Op de 2.000 andere bandits kostte een misser van ε-greedy over de laatste 1.000 keer trekken gemiddeld 1,19 tegenover de beste automaat van dat moment, een misser van de optimistische agent 0,46. Van wat ε-greedy misloopt, komt negen tiende van de willekeurige keuzes: één keer op de tien neemt de agent een willekeurige automaat, ook de slechtste. De optimistische agent mist vaker, maar meestal met een automaat die bijna net zoveel geeft als de beste.",
            "easier": "De demo geeft elke agent dezelfde 200 bandits, met dezelfde gemiddelden, verschuiving en ruis: kunnen de bandits het verschil verklaren? Wat kost een misser van elke agent?",
            "start": "Een schatting is wat de agent verwacht, niet wat de agent krijgt: de reward komt van het gemiddelde van de automaat, wat de agent ook schat. En hoeveel is er na duizenden keren trekken met η = 0,1 nog over van de startwaarde?",
            "luck": "Elk aandeel en elke gemiddelde reward hier gaat over 200 bandits en 1.000 keer trekken, en 2.000 andere bandits geven dezelfde volgorde. Wat kost een misser van elke agent?",
        },
    )
    choice_measures
    return (choice_measures,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Welke automaat geeft het meest? De multi-armed bandit](/course/open/rl-bandits?lang=nl) en [Gemiddelden bijhouden: verschuivende automaten en optimistische starts](/course/open/rl-bandit-updates?lang=nl) speelde de code van de course op de bandits. Nu heeft je project `agent` een `BanditAgent`, een loop die de agent laat spelen en een maat, naast het testbed van de course. Je vergelijking liet zien dat de optimistische start op deze bandits loont, dat ε-greedy blijft ontdekken welke automaten beter worden, en dat een zuivere vergelijking dezelfde bandits voor elke agent nodig heeft, veel runs en meer dan één maat. Je project komt terug in *Project: een agent voor je eigen grid world*, waar de omgeving ook states heeft. De volgende les, *Wat is een policy waard? Policy evaluation*, begint het deel over RL met een tabel: problemen met veel states, waarbij de agent weet hoe de omgeving werkt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_measures, result_agent, result_average, result_run):
    L.wrap_up(result_agent, result_run, result_average, choice_measures)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
