# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-q-learning. Route: course/route/lessen/overige-takken.yaml. Checks, de environment, de run en het terugspoelen: checks.py.
# termen:allow epsilon-greedy

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-q-learning", "nl")
    C = L.checks   # GRID, de environment die zijn dobbelstenen gooit, de Q-learning van de course en zijn opgenomen run


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Als stappen kunnen uitglijden: value iteration](/course/open/rl-value-iteration?lang=nl) en [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_table():
    quick_table = L.choice(
        "quick-table",
        question="Wat heeft value iteration nodig om $Q(s, a)$ uit te rekenen?",
        options={
            "rewards": "Alleen de rewards die de agent tot nu toe kreeg",
            "right": "Voor elke state en action zijn uitkomsten: kans, volgende state en reward",
            "nothing": "Niets: het probeert actions tot het G vindt",
        },
        correct="right",
        why={
            "right": "De tabel `T`: de Bellman equation telt elke uitkomst van een action op, elk met zijn kans.",
            "rewards": "Value iteration probeert niets, maar telt elke uitkomst op uit de tabel `T`. Zie [Als stappen kunnen uitglijden: value iteration](/course/open/rl-value-iteration?lang=nl).",
            "nothing": "Value iteration zet nooit een stap, maar heeft de tabel `T` nodig. Zie [Als stappen kunnen uitglijden: value iteration](/course/open/rl-value-iteration?lang=nl).",
        },
    )
    quick_table
    return (quick_table,)


@app.cell(hide_code=True)
def quick_check_random():
    quick_random = L.choice(
        "quick-random",
        question="`rng = np.random.default_rng(0)`, en dan `rng.random(size=1000) < 0.1`. Ongeveer hoeveel van de 1000 zijn `True`?",
        options={
            "right": "Ongeveer 100",
            "most": "Ongeveer 900",
            "exact": "Precies 100",
        },
        correct="right",
        why={
            "right": "`rng.random` geeft floats van 0 tot 1, gelijk verdeeld, dus ongeveer één op de tien ligt onder 0,1.",
            "most": "`< 0.1` is alleen `True` voor de getallen onder 0,1: ongeveer één op de tien. Zie [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).",
            "exact": "Ongeveer één op de tien ligt onder 0,1, maar de getallen zijn willekeurig: met deze seed zijn het er 89. Zie [Willekeurige getallen en seeds](/course/open/np-random?lang=nl).",
        },
    )
    quick_random
    return (quick_random,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Value iteration had de tabel `T` nodig: elke kans en elke reward. Een robot in een nieuw gebouw, of een spel waarvan niemand de regels opschreef, heeft zo'n tabel niet. Hij kan alleen proberen: een action nemen, en zien waar hij landt en wat hij krijgt.

    **Een voorbeeld uit het dagelijks leven.** In een nieuwe stad zoek je de snelste fietsroute naar je werk. Zonder kaart met reistijden probeer je routes, onthoud je hoe lang elke duurde, en neem je meestal de beste tot nu toe; af en toe probeer je een andere, voor het geval die sneller is.
    """)
    return


@app.cell(hide_code=True)
def predict_same():
    guess_same = L.guess(
        "same",
        question="Een agent in GRID ziet `T` nooit: na elke stap hoort hij alleen waar hij landde en zijn reward. Kan hij toch leren over de bovenste rij te gaan, zoals value iteration deed?",
        options={
            "yes": "Ja: de uitglijders die hij meemaakt, leren hem het risico",
            "straight": "Nee: hij neemt de weg rechtdoor, want hij kent het uitglijden niet",
            "never": "Nee: zonder `T` kan hij geen policy leren",
        },
        reveal="**Ja.** In de run van de course gaan zijn beste actions na een tijd van S over de bovenste rij. Een uitglijder een kuil in kost hem elke keer −1, dus telt het risico even vaak mee als het gebeurt.",
    )
    guess_same
    return (guess_same,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        De agent leert $Q(s, a)$ van zijn eigen stappen, en houdt die bij in een **Q-table**: een getal voor elke state en action, allemaal 0 bij de start. Na een stap weet hij zijn state $s$, zijn action $a$, de reward $r$ en de state $s'$ waar hij landde. Value iteration telde alle uitkomsten op, elk met zijn kans; de agent heeft alleen de uitkomst die gebeurde. Daarom schuift hij $Q(s, a)$ een stap op naar die uitkomst, het **target** $r + \gamma \max_{a'} Q(s', a')$: de reward plus de verdisconteerde grootste Q waar hij landde. Dat is **Q-learning**:
        """),
        L.formula(r"[[param:Q(s, a)]] \leftarrow [[param:Q(s, a)]] + [[hyper:\eta]] \left( [[data:r]] + [[hyper:\gamma]] \max_{a'} [[param:Q(s', a')]] - [[param:Q(s, a)]] \right)"),
        mo.md(r"""
        $\eta$ is de learning rate, zoals bij gradient descent; boeken over Q-learning schrijven α. Een uitkomst die in 8 van de 10 stappen komt, trekt 8 keer zo vaak als een die in 1 van de 10 komt, dus schuift $Q(s, a)$ over veel stappen op naar de verwachtingswaarde: de $Q$ van value iteration.

        Welke action neemt de agent? Altijd die met de grootste $Q$ nemen, **exploitation**, is niet genoeg: dan komt hij nooit te weten wat de andere doen. **ε-greedy** neemt met kans $\varepsilon$ een willekeurige action, **exploration**, en anders de beste. Eén doorloop van S tot G, een kuil of 100 stappen is een **episode**; daarna begint de agent weer bij S, met de Q-table die hij heeft.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **Q-learning** | | $Q(s, a)$ leren van stappen, door hem op te schuiven naar het target | `update` |
        | **Q-table** | | een getal $Q(s, a)$ voor elke state en action | `Q[s][a]` |
        | **exploitation** | | de action nemen die nu het beste lijkt | |
        | **exploration** | | een andere action proberen, om te weten te komen wat die doet | |
        | **ε-greedy** | epsilon-greedy | met kans $\varepsilon$ een willekeurige action, anders die met de grootste $Q$ | `choose` |
        | **episode** | | één doorloop van de start tot de taak voorbij is, of tot een maximum aantal stappen | |
        | $\varepsilon$ | epsilon | de kans op een willekeurige action | `epsilon` |
        | $\leftarrow$ | wordt | de linkerkant krijgt de waarde van de rechterkant | `=` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_replay():
    mo.md(rf"""
    ### De run, episode voor episode

    De course liet Q-learning {C.EPISODES} episodes lopen op GRID, met $\varepsilon$ = {C.number(C.EPSILON, 1, 'nl')}; zodra je in *Zelf doen* je eigen agent traint, zie je hier de jouwe. Een lijn is de weg van die episode, een stip een stap tegen de rand; een getal is de grootste <span class="c-param">Q</span> van een state na die episode, en een driehoekje zijn beste action. De *greedy weg* is waar de agent vanaf S heen zou gaan als hij altijd zijn beste action nam en niets uitgleed.
    """)
    return


@app.cell(hide_code=True)
def replay_cell(trained):
    _run = trained if trained is not None else C.RUN
    _labels = {"title": "Return van elke episode", "x": "episode", "y": "return",
               "alt": "De return van elke episode tot de gekozen, dun, en het gemiddelde over 20 episodes, dik, met een stip bij de gekozen.",
               "grid": "Na episode {k}",
               "grid_alt": "GRID na episode {k}: de weg van die episode als lijn, de grootste Q van elke state, en een driehoekje voor zijn beste action.",
               "start": "Episode 0: elke Q is 0, en nog geen action is beter dan een andere.",
               "episode": "Episode {k}: na {n} stappen {end}, met een return van {g}. De greedy weg vanaf S {way}.",
               "goal": "op G", "pit": "in een kuil", "limit": "gestopt",
               "way_top": "gaat omhoog en over de bovenste rij, zoals die van value iteration",
               "way_goal": "komt langs een andere weg op G", "way_not": "komt nog niet op G"}
    _out = L.explore("replay", "episode", lambda _v: C.replay_frame(_run, int(_v), _labels, L.lang), start=C.EPISODES,
                     values=C.stops(), role="",
                     task="Ga terug naar episode 0, en dan vooruit. Vanaf welke episode gaat de greedy weg vanaf S over de bovenste rij?")
    _out
    return


@app.cell(hide_code=True)
def text_replay_after():
    _last = sum(C.RUN["returns"][-100:]) / 100
    mo.md(rf"""
    In de run van de course vanaf episode {C.top_from(C.RUN)}. Zodra de agent G gevonden heeft, reist de reward een stap per keer terug. De gemiddelde return van de laatste 100 episodes, {C.number(_last, 2, "nl")}, blijft onder de {C.number(C.Q_STAR[C.START]["up"], 2, "nl")} van value iteration: één stap op de tien is willekeurig, en kan een kuil in gaan. Na {C.EPISODES} episodes is hij het in {C.agree(C.RUN["final"])} van de {len(C.STATES)} states eens met value iteration. In `(1, 4)`, waar hij zelden komt, gaat hij omhoog in plaats van naar rechts; value iteration waardeert die twee bijna even hoog, {C.number(C.Q_STAR[(1, 4)]["up"], 2, "nl")} en {C.number(C.Q_STAR[(1, 4)]["right"], 2, "nl")}.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Q-learning:

    1. Maak een Q-table: 0 voor elke state en action.
    2. Begin een episode bij S.
    3. Kies een action, ε-greedy: met kans $\varepsilon$ een willekeurige, anders die met de grootste $Q$.
    4. Neem hem: de environment geeft terug waar je landde en de reward.
    5. Reken het target uit, de reward plus $\gamma$ keer de grootste $Q$ waar je landde, en schuif $Q$ van je state en action een stap $\eta$ op naar het target.
    6. Ga verder vanaf waar je landde, bij stap 3, tot de episode voorbij is: op G, in een kuil, of na 100 stappen. Begin dan de volgende episode, bij stap 2.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    SMALL uit de vorige les, met $\eta$ = 0,5 en $\gamma$ = 0,9. Drie stappen die een agent daar zette, gaan door stap 5; de `print` laat elke stap zien, zijn target en de nieuwe $Q$. Dan stap 3, acht keer in `(0, 0)`, met $\varepsilon$ = 0,25. Voer de cel uit.
    """)
    return


@app.cell
def example_q_learning():
    import numpy as np

    # 1. de Q-table van SMALL: een Q voor elke action in elke state, allemaal 0 bij de start
    _Q = {}
    for _s in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]:
        _Q[_s] = {"up": 0, "down": 0, "left": 0, "right": 0}

    # drie stappen die een agent zette: (state, action, reward, waar hij landde); in de derde gleed de action omhoog uit naar rechts
    _steps = [((0, 1), "right", 1, (0, 2)), ((0, 0), "right", 0, (0, 1)), ((1, 1), "up", -1, (1, 2))]
    for _s, _a, _r, _nxt in _steps:
        # 5. het target: de reward plus gamma keer de grootste Q waar hij landde
        _qs = []
        for _b in _Q[_nxt]:
            _qs.append(_Q[_nxt][_b])
        _target = _r + 0.9 * max(_qs)
        # 5. Q schuift een stap eta = 0.5 op naar het target
        _Q[_s][_a] = _Q[_s][_a] + 0.5 * (_target - _Q[_s][_a])
        print(_s, _a, _r, _nxt, round(_target, 3), round(_Q[_s][_a], 3))

    # 3. epsilon-greedy in (0, 0), acht keer: met epsilon = 0.25 een willekeurige action, anders die met de grootste Q
    _rng = np.random.default_rng(4)
    _actions = ["up", "down", "left", "right"]
    _qs = []
    for _b in _actions:
        _qs.append(_Q[(0, 0)][_b])
    for _i in range(8):
        if _rng.random() < 0.25:
            print("willekeurig:", _actions[_rng.integers(0, 4)])
        else:
            print("beste:", _actions[np.argmax(_qs)])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste stap landt op G: het target is de reward, 1, want de $Q$ van G blijft 0, en $Q$ schuift halverwege op, naar 0,5. De tweede landt op `(0, 1)`, waarvan de grootste $Q$ nu 0,5 is: het target is 0,9 · 0,5 = 0,45, en $Q$ wordt 0,225: de reward van G reist terug. In de derde gleed de action omhoog uit, de kuil in: de agent weet niet dat hij uitgleed; zijn target is de reward, −1, en $Q$ wordt −0,5. Twee van de acht keuzes zijn willekeurig, omlaag en links; de andere zes nemen rechts, de grootste $Q$ in `(0, 0)`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_update():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Eén stap Q-learning

        `update(Q, state, action, reward, nxt, eta, gamma)` geeft de nieuwe $Q$ van `state` en `action` terug na een stap die op `nxt` landde: stap 5 van het uitgewerkte voorbeeld. Vervang de twee `...`: het target en de nieuwe $Q$. De check roept `update(Q, (1, 0), "up", 0, (0, 0), 0.2, 0.9)` aan, met `Q` de Q-table aan het eind van de run van de course, en nog vijf andere stappen.
        """),
        L.task("Laat `update` het target r + γ max Q(s′, a′) uitrekenen, en Q(s, a) teruggeven, een stap η naar het target opgeschoven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_update():
    box_update = L.code_box("update", lines=7)
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
            "Stap 5 van het recept, en de formule van *Het idee*: eerst het target, dan de stap ernaartoe.",
            "De loop zet elke $Q$ van `nxt` in `qs`, dus `max(qs)` is de grootste. De stap is `eta` keer het verschil tussen het target en de oude $Q$.",
            "`target = reward + gamma * ___(qs)` en `return Q[state][action] + eta * (___ - Q[state][action])`.",
        ],
        why="Het target is wat deze ene stap zegt dat $Q(s, a)$ moet zijn. Omdat $Q$ maar een deel `eta` van de weg opschuift, maakt één uitglijder niet veel stappen ongedaan.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_choose():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. ε-greedy

        `choose(Q, state, epsilon, rng)` geeft de action terug die de agent in `state` neemt: stap 3 van het uitgewerkte voorbeeld. Vervang de twee `...`: wanneer je een willekeurige action neemt, en welke action anders. De check roept `choose(Q, (1, 0), 0.1, rng)` 4000 keer aan en telt elke action: omhoog, de beste, hoort ongeveer 92,5 keer op 100 te komen, 90 plus een kwart van de 10 willekeurige.
        """),
        L.task("Laat `choose` met kans ε een willekeurige action nemen, en anders de action met de grootste Q."),
    ])
    return


@app.cell(hide_code=True)
def exercise_choose():
    box_choose = L.code_box("choose", lines=12)
    box_choose
    return (box_choose,)


@app.cell(hide_code=True)
def check_choose(box_choose):
    result_choose = L.check("choose", box_choose.value)
    result_choose
    return (result_choose,)


@app.cell(hide_code=True)
def hints_choose():
    L.hints(
        "choose",
        [
            "Stap 3 van het recept, en het uitgewerkte voorbeeld: een test met `rng.random()`, dan de beste action.",
            "`rng.random()` geeft een float van 0 tot 1, onder `epsilon` in een deel `epsilon` van de aanroepen. `np.argmax(qs)` is de plek van de grootste $Q$; `actions` heeft op elke plek de action.",
            "`if rng.random() < ___:` en `return actions[np.___(qs)]`.",
        ],
        why="In een deel `epsilon` van de aanroepen exploreert de agent, met elk van de vier actions, ook de beste; in de andere kiest hij voor exploitation.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(rf"""
    ### Een agent trainen

    De knop traint een agent op GRID, {C.EPISODES} episodes lang, met jouw `update` en `choose`, met $\eta$ = {C.number(C.ETA, 1, 'nl')}, $\varepsilon$ = {C.number(C.EPSILON, 1, 'nl')} en $\gamma$ = {C.number(C.GAMMA, 1, 'nl')}, zoals in het recept:

    ```python
    for episode in range({C.EPISODES}):
        state = (1, 0)
        for t in range({C.MAX_STEPS}):
            action = choose(Q, state, epsilon, rng)
            nxt, reward = step(state, action, rng)   # de environment: alleen die kent T
            Q[state][action] = update(Q, state, action, reward, nxt, eta, gamma)
            state = nxt
            if state in ends:   # G of een kuil: de episode is voorbij
                break
    ```

    De grafiek laat de return van elke episode zien, dun, en het gemiddelde over de laatste 20, dik; gestippeld hetzelfde, gemiddeld over 20 runs van de course. De demo wacht na elke episode 10 ms, zodat je kunt meekijken.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    train_button = mo.ui.run_button(label="Train een agent met mijn update en choose")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_train(box_choose, box_update, result_choose, result_update, train_button):
    trained = None
    if not train_button.value:
        _out = mo.md("*Druk op de knop om een agent te trainen.*")
    else:
        _mine = [_n for _n, _r in (("update", result_update), ("choose", result_choose)) if _r.state == "correct"]
        _update = run_code(box_update.value).names["update"] if "update" in _mine else C.update
        _choose = run_code(box_choose.value).names["choose"] if "choose" in _mine else C.choose
        _live = L.live({"title": "Jouw run: de return van elke episode", "x": "episode", "y": "return",
                        "alt": "De return van elke episode van je run tot nu toe, dun, en het gemiddelde over de laatste 20, dik; gestippeld hetzelfde, gemiddeld over 20 runs van de course."},
                       C.EPISODES, role="ink", low=-1.05, high=1.05, window=20)
        _live.line("**Trainen** …")
        _live.reference(list(enumerate(C.mean_line(), 1)), "gemiddelde van 20 runs", role="ink")
        _r = C.train(_update, _choose, seed=C.YOUR_SEED, pause=0.01,
                     show=lambda _k, _g: _live.add(_k, _g, f"**Episode {_k} van {C.EPISODES}**"))
        if _r == "update":
            _out = _live.stopped("*Je `update` gaf tijdens de run een fout of geen getal. Haalt hij zijn check nog?*")
        elif _r == "choose":
            _out = _live.stopped("*Je `choose` gaf tijdens de run een fout of geen action. Haalt hij zijn check nog?*")
        else:
            trained = _r
            _theirs = [_n for _n in ("update", "choose") if _n not in _mine]
            if len(_mine) == 2:
                _who = "jouw `update` en jouw `choose`"
            elif len(_mine) == 1:
                _who = f"jouw `{_mine[0]}` en de `{_theirs[0]}` van de course"
            else:
                _who = "de `update` en `choose` van de course, omdat de jouwe hun checks nog niet halen"
            _off = C.differ(_r["final"])
            _where = "" if not _off else " Waar niet: " + ", ".join(f'in `{_s}` neemt hij `"{_a}"`' for _s, _a in _off) + "."
            _out = mo.vstack([
                mo.Html(_live.picture()),
                mo.md(f"Getraind met {_who}. De gemiddelde return van de laatste 100 episodes is {C.number(sum(_r['returns'][-100:]) / 100, 2, 'nl')}. In {C.agree(_r['final'])} van de {len(C.STATES)} states waar de taak doorgaat, neemt je agent een action die value iteration ook neemt.{_where} *De run, episode voor episode*, in *Het idee*, laat nu jouw run zien."),
                C.policies(_r["final"], ("Jouw agent: de grootste Q", "Value iteration: de grootste Q"),
                           f"GRID twee keer: eerst de grootste Q die je agent in elke state leerde, met een driehoekje voor zijn beste action, dan hetzelfde voor value iteration. Ze komen in {C.agree(_r['final'])} van de {len(C.STATES)} states overeen.", decimal=","),
            ])
    _out
    return (trained,)


@app.cell(hide_code=True)
def text_exercise_zero():
    mo.md(r"""
    ### Oefening 3. Nooit exploreren
    """)
    return


@app.cell(hide_code=True)
def choice_zero_cell():
    choice_zero = L.choice(
        "no-exploring",
        question="Met $\\varepsilon$ = 0 neemt de agent altijd de action met de grootste $Q$. Wat gebeurt er in GRID, met elke $Q$ op 0 bij de start?",
        options={
            "faster": "Hij leert sneller: hij verspilt geen stappen aan willekeurige actions",
            "stuck": "Hij komt niet op G: met elke Q op 0 gaat hij altijd omhoog",
            "later": "Hij leert dezelfde policy, alleen later",
        },
        correct="stuck",
        why={
            "stuck": "In 20 runs van 300 episodes kwam hij geen enkele keer op G: bij gelijke $Q$ neemt `np.argmax` de eerste action, omhoog, en een uitglijder brengt hem alleen opzij. Zonder exploration komt hij nooit te weten wat de andere actions doen.",
            "faster": "Zonder willekeurige stappen schiet hij niets op: met elke $Q$ op 0 gaat hij altijd omhoog, en vanuit de bovenste rij leidt omhoog nooit naar G. Zie exploration in *Het idee*.",
            "later": "Hij leert helemaal niets: hij komt niet op G, dus verandert geen reward ooit een $Q$. Zie exploration in *Het idee*.",
        },
    )
    choice_zero
    return (choice_zero,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Als stappen kunnen uitglijden: value iteration](/course/open/rl-value-iteration?lang=nl) telde de Bellman equation elke uitkomst op uit de tabel `T`. Q-learning gebruikt dezelfde vergelijking zonder `T`: elke stap is één uitkomst, en kleine stappen $\eta$ ernaartoe middelen de uitkomsten. Reinforcement learning komt terug in een tak die nog komt, met als project je eigen agent. In de volgende les, *Een language model voorspelt het volgende token*, leert een model van alleen tekst welk token er volgt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_zero, result_choose, result_update):
    L.wrap_up(result_update, result_choose, choice_zero)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
