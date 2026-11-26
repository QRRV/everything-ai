# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-bandits. Route: course/route/lessen/reinforcement-learning.yaml. Checks, de bandits, de runs en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rl-bandits", "nl")
    C = L.checks   # de bandits, de agents en runs van de course, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl) en [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_greedy():
    quick_greedy = L.choice(
        "quick-greedy",
        question="Een agent kiest ε-greedy uit vier actions, met $\\varepsilon$ = 0,1. Hoe vaak neemt de agent ongeveer de action met de grootste $Q$?",
        options={
            "ten": "Ongeveer 10 van de 100 keer",
            "right": "Ongeveer 92,5 van de 100 keer: een willekeurige action kan ook de beste zijn",
            "ninety": "Precies 90 van de 100 keer, want de andere 10 zijn willekeurige actions en nooit de beste",
        },
        correct="right",
        why={
            "right": "In 90 van de 100 keuzes neemt de agent de beste action. In de andere 10 neemt de agent een willekeurige van de vier, in een kwart daarvan de beste: 90 + 2,5 = 92,5.",
            "ten": "$\\varepsilon$ = 0,1 is de kans op een willekeurige action: hoe vaak neemt de agent bij de andere keuzes de beste? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
            "ninety": "Een willekeurige action is een van alle vier de actions, dus kan het ook de beste zijn? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
        },
    )
    quick_greedy
    return (quick_greedy,)


@app.cell(hide_code=True)
def quick_check_spread():
    quick_spread = L.choice(
        "quick-spread",
        question="De rewards van een automaat hebben een gemiddelde van 1 en een standaardafwijking van 1. Wat zegt dat over de rewards?",
        options={
            "all": "Elke reward is precies 1",
            "right": "Rond 1, gewoonlijk ongeveer 1 ervan af, sommige dichterbij, sommige verder",
            "exact": "Alle rewards liggen tussen 0 en 2, want de standaardafwijking zegt hoe ver de rewards hooguit gaan",
        },
        correct="right",
        why={
            "right": "De standaardafwijking zegt hoe ver de waarden gewoonlijk van het gemiddelde liggen: hier ongeveer 1, sommige dichterbij en sommige verder.",
            "all": "Een standaardafwijking van 1 betekent dat de waarden rond het gemiddelde verspreid liggen: zouden de waarden verspreid liggen als alles 1 was? Zie [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).",
            "exact": "De standaardafwijking is hoe ver waarden gewoonlijk van het gemiddelde liggen, geen grens: kan een waarde verder weg liggen? Zie [Data beschrijven: gemiddelde, mediaan en spreiding](/course/open/stats-describe?lang=nl).",
        },
    )
    quick_spread
    return (quick_spread,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een nieuwssite heeft vier koppen voor één bericht en wil de kop met de meeste klikken. Een webwinkel probeert twee knoppen. Elke poging leert de site iets, en elke bezoeker met een slechte kop is een klik minder. Het is niet genoeg om alleen de kop te tonen die tot nu toe het best deed: een kop met een slechte start kan beter zijn. Dit zijn de eenvoudigste problemen van reinforcement learning: steeds weer dezelfde keuze, na elke keuze een reward, en geen states om tussen te bewegen.

    **Een voorbeeld uit het dagelijks leven.** In een nieuwe kantine probeer je de gerechten. Neem je na een paar lunches steeds het gerecht dat tot nu toe het lekkerst was, of probeer je af en toe een ander gerecht, voor het geval het beter is?

    Een gokautomaat met één hendel heet in het Engels een *one-armed bandit*: de automaat geeft een willekeurig bedrag, en gemiddeld minder dan je erin stopt. Een rij automaten, elk met een eigen gemiddelde, geeft het probleem de naam: de **multi-armed bandit**. Welke automaat geeft het meest?
    """)
    return


@app.cell(hide_code=True)
def predict_greedy():
    guess_greedy = L.guess(
        "greedy",
        question="Een **greedy** agent trekt altijd aan de automaat met de grootste schatting tot nu toe. De course laat de agent 2.000 willekeurige bandits van tien automaten spelen, en bij elke bandit 1.000 keer trekken. In hoeveel bandits trekt de agent aan het eind aan de beste automaat?",
        options={
            "most": "In ongeveer 9 van de 10 bandits",
            "right": "In ongeveer 1 op de 3",
            "few": "In slechts ongeveer 1 op de 10",
        },
        reveal="**In ongeveer 1 op de 3.** Over de laatste 100 van de 1.000 keer trekken trekt greedy in 36,7% van de 2.000 bandits aan de beste automaat. De agent blijft trekken aan een automaat die tot nu toe goed gaf, en probeert pas een nieuwe als elke geprobeerde automaat gemiddeld minder dan 0 gaf: van de tien automaten probeert de agent er gemiddeld 3,3. Oefening 2 telt dit aandeel.",
    )
    guess_greedy
    return (guess_greedy,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een bandit in deze les heeft tien automaten, genummerd 0 tot en met 9; elke automaat is een *arm*, een action van de agent. Een keer trekken geeft een willekeurige reward rond het gemiddelde van de automaat, met een standaardafwijking van 1. Elke automaat heeft een ander gemiddelde, en de agent kent de gemiddelden niet. Anders dan GRID in [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl) heeft een bandit één state: een keer trekken verandert niets aan de volgende keer. De afbeelding toont één bandit; aan elke automaat is 40 keer getrokken.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        Automaat 9 geeft het meest, gemiddeld 3,32, en automaat 0 komt op de tweede plaats met 2,04; het gemiddelde van een automaat is de verwachtingswaarde van de reward, zoals in [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl). Met 40 keer trekken per automaat zie je het in de afbeelding. De agent moet het ontdekken door te trekken, en elke keer trekken aan een slechte automaat kost reward.

        De schatting van een automaat is het **steekproefgemiddelde**: het totaal van de rewards van de automaat, gedeeld door het aantal keer trekken $N(a)$. Zolang er nog nooit aan een automaat getrokken is, is de schatting 0.
        """),
        L.formula(r"[[param:Q(a)]] = \frac{[[data:r_1]] + [[data:r_2]] + \dots + [[data:r_{N(a)}]]}{N(a)}"),
        mo.md(r"""
        Met deze schattingen heeft de agent van [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl) weer de twee keuzes. Exploitation trekt aan de automaat met de grootste schatting; exploration trekt aan een andere, om te ontdekken wat de andere automaat geeft. Op de bandit van de afbeelding, met 200 keer trekken, trekt de greedy agent alle 200 keer aan automaat 0: de eerste reward zette automaat 0 boven de 0 van de andere, dus probeerde de agent de andere nooit. Een agent met ε-greedy en $\varepsilon$ = 0,1 trekt 94 keer aan automaat 0 en 90 keer aan automaat 9, en krijgt 2,43 per keer tegen 2,09. Elke keer exploration kost nu reward, en elke keer exploitation kan een betere automaat laten liggen: de **exploration-exploitation trade-off**.

        Eén bandit zegt weinig: een agent kan geluk hebben. De course maakt 2.000 bandits, elk met tien gemiddelden rond 0 met een standaardafwijking van 1, laat elke agent elke bandit 1.000 keer spelen, en neemt bij elke trek de gemiddelde reward over de 2.000. Deze test komt uit het boek over reinforcement learning van Sutton en Barto. Uitgezet tegen de trek laat de gemiddelde reward zien hoe snel een agent leert, en hoeveel de agent aan het eind krijgt.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **multi-armed bandit** | | automaten, elk een action, met een willekeurige reward rond een gemiddelde dat de agent niet kent; er is één state | `means` |
        | **steekproefgemiddelde** | | de schatting van een automaat: het totaal van de rewards gedeeld door het aantal keer trekken | `estimates` |
        | **exploration-exploitation trade-off** | | exploration kost nu reward, exploitation kan een betere action laten liggen | `epsilon` |
        | **greedy** | | altijd de action met de grootste schatting nemen: ε-greedy met $\varepsilon$ = 0 | |
        | **trek** | | één stap van de agent: een automaat kiezen en de reward krijgen | |
        | $Q(a)$ | Q van a | de schatting van automaat $a$ | `q[a]` |
        | $N(a)$ | N van a | hoe vaak er tot nu toe aan automaat $a$ getrokken is | `counts[a]` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een agent met steekproefgemiddelden op een bandit:

    1. Geef elke automaat een totaal van 0 en een aantal van 0; elke schatting is 0.
    2. Kies een automaat, ε-greedy: met kans $\varepsilon$ een willekeurige, anders de automaat met de grootste schatting.
    3. Trek eraan: de bandit geeft een reward.
    4. Tel de reward op bij het totaal van de automaat, en 1 bij het aantal.
    5. Reken de schatting opnieuw uit: het totaal gedeeld door het aantal.
    6. Ga verder bij stap 2, tot de laatste trek. Om de agent te beoordelen, laat je de agent veel bandits spelen en neem je bij elke trek de gemiddelde reward.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een kleine bandit van drie automaten, met gemiddelden 0,5, 1,5 en −0,5, en $\varepsilon$ = 0,2: tien keer trekken volgens stap 1 tot en met 5. `rng.normal()` geeft een willekeurig getal rond 0 met een standaardafwijking van 1, dus `_means[_a] + _rng.normal()` is een reward rond het gemiddelde van automaat `_a`. Elke regel toont de trek, de automaat, de reward, de drie schattingen en de drie aantallen. Voer de cel uit.
    """)
    return


@app.cell
def example_bandit():
    import numpy as np

    # een kleine bandit van drie automaten: de gemiddelden, die de agent niet kent
    _means = [0.5, 1.5, -0.5]
    _rng = np.random.default_rng(9)

    # 1. een totaal en een aantal van 0 voor elke automaat; elke schatting is 0
    _totals = [0.0, 0.0, 0.0]
    _counts = [0, 0, 0]
    _q = [0.0, 0.0, 0.0]
    for _t in range(1, 11):
        # 2. epsilon-greedy met epsilon = 0.2: een willekeurige automaat, anders de automaat met de grootste schatting
        if _rng.random() < 0.2:
            _a = int(_rng.integers(0, 3))
        else:
            _a = int(np.argmax(_q))
        # 3. trek eraan: een reward rond het gemiddelde van de automaat
        _r = _means[_a] + _rng.normal()
        # 4. tel de reward op bij het totaal van de automaat, en 1 bij het aantal
        _totals[_a] = _totals[_a] + _r
        _counts[_a] = _counts[_a] + 1
        # 5. de schatting van de automaat opnieuw: het totaal gedeeld door het aantal
        _q[_a] = _totals[_a] / _counts[_a]
        print(_t, _a, round(_r, 2), [round(_x, 2) for _x in _q], _counts)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Bij trek 1 is elke schatting 0, en `np.argmax` neemt de eerste: automaat 0. De reward van 0,74 zet automaat 0 boven de andere, en de agent blijft automaat 0 nemen; de schatting verandert met elke reward, 0,74, 0,95, 0,65 en 0,67, elke keer het totaal tot nu toe gedeeld door het aantal. Trek 5 en 6 zijn willekeurige keuzes, want greedy had automaat 0 genomen: automaat 2 geeft −2,53, automaat 1 geeft 1,45. Vanaf dan heeft automaat 1 de grootste schatting, en neemt de agent automaat 1; na tien keer trekken is de schatting 1,34, dicht bij het gemiddelde van 1,5. Zonder de willekeurige trek 6 had de agent automaat 1 misschien nooit geprobeerd.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_estimates():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De schatting van elke automaat

        `estimates(totals, counts)` geeft een lijst `q` terug met de schatting van elke automaat: stap 5 van het recept, voor alle automaten tegelijk. `totals[i]` is het totaal van de rewards van automaat `i` tot nu toe, `counts[i]` hoe vaak eraan getrokken is. Vervang de twee keer `...`: wat een automaat krijgt waaraan nooit getrokken is, en het steekproefgemiddelde van de andere. De check roept `estimates([3.0, 0.0, -1.5], [2, 0, 3])` aan, wat `[1.5, 0.0, -0.5]` moet geven, en vijf andere gevallen, waaronder de totalen en aantallen van een agent na 200 keer trekken.
        """),
        L.task("Laat `estimates` het steekproefgemiddelde van elke automaat teruggeven, en `0.0` voor een automaat waaraan nooit getrokken is."),
    ])
    return


@app.cell(hide_code=True)
def exercise_estimates():
    box_estimates = L.code_box("estimates", lines=8)
    box_estimates
    return (box_estimates,)


@app.cell(hide_code=True)
def check_estimates(box_estimates):
    result_estimates = L.check("estimates", box_estimates.value)
    result_estimates
    return (result_estimates,)


@app.cell(hide_code=True)
def hints_estimates():
    L.hints(
        "estimates",
        [
            "Stap 1 en 5 van het recept: een automaat met aantal 0 krijgt `0.0`, elke andere automaat het totaal gedeeld door het aantal. `q.append` zet een getal achteraan in de lijst `q`.",
            "Voor automaat 0 van het voorbeeld is `totals[0]` gelijk aan `3.0` en `counts[0]` gelijk aan `2`, dus de schatting is 3,0 / 2 = 1,5. In de loop is `i` het nummer van de automaat.",
            "`            q.append(___)`",
        ],
        why="`q.append(totals[i] / counts[i])` onder `else`, en `q.append(0.0)` onder `if`: zonder keer trekken heeft een automaat geen rewards om te middelen, dus begint de schatting bij 0, zoals elke $Q$ bij Q-learning.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Laat drie agents spelen

    De knop laat drie agents dezelfde 300 bandits spelen, elk 1.000 keer trekken: greedy, $\varepsilon$ = 0,01 en $\varepsilon$ = 0,1. Alle drie gebruiken je `estimates` zodra je functie de check haalt, en tot dan de functie van de course. De run van één agent gaat zo; `choose` is de ε-greedy van Q-learning, zonder state, want een bandit heeft slechts één state:

    ```python
    for t in range(1000):
        for b in range(300):   # elke bandit
            a = choose(q[b], epsilon, rng)
            r = means[b][a] + rng.normal()
            totals[b][a] = totals[b][a] + r
            counts[b][a] = counts[b][a] + 1
            q[b] = estimates(totals[b], counts[b])
    ```

    De grafiek toont de gemiddelde reward van $\varepsilon$ = 0,1 over de 300 bandits bij elke trek, dun, en het gemiddelde over de laatste 20 keer trekken, dik; gestreept het gemiddelde over de laatste 20 keer trekken van greedy en van $\varepsilon$ = 0,01. Ernaast zie je wat greedy en $\varepsilon$ = 0,1 op dat moment weten van de eerste van de 300 bandits: de gemiddelden als korte streepjes, de schattingen in het <span class="c-param">oranje</span>, en hoe vaak de agents aan elke automaat trokken. Het duurt een paar seconden; daarna kun je terug door de run.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    run_button = mo.ui.run_button(label="Laat de drie agents spelen met mijn estimates")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def demo_run(box_estimates, result_estimates, run_button):
    if not run_button.value:
        _out = mo.md("*Druk op de knop om de drie agents te laten spelen.*")
    else:
        _yours = C.mine(box_estimates.value, result_estimates.state == "correct")
        _w = C.live_words(L.lang)
        _now = {}   # de live grafiek, wat de agents na de laatste trek van de eerste bandit weten, en bij welke trek
        _raw = {0.0: [], 0.01: []}   # de gemiddelde reward van greedy en van 0.01 bij elke trek, voor de gestreepte lijnen

        def _progress(_t, _r, _seen):
            if "seen" not in _now:   # vanaf de eerste trek: wat de agents weten, naast de grafiek
                _now["chart"].watch(lambda: C.now_picture(C.watched_means(), _now["seen"], _now["t"], L.lang))
            _now["seen"], _now["t"] = _seen, _t
            _raw[0.0].append(_r[0.0])
            _raw[0.01].append(_r[0.01])
            if _t % 25 == 0 or _t == C.STEPS:
                _now["chart"].reference(C.smoothed(_raw[0.0]), "greedy", role="ink")
                _now["chart"].reference(C.smoothed(_raw[0.01]), "ε = 0,01", role="ink")
            _now["chart"].add(_t, _r[0.1], f"**Trek {_t} van {C.STEPS}** · de drie agents op 300 bandits")

        _now["chart"] = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"], "alt": _w["alt"]},
                       C.STEPS, role="ink", low=-0.4, high=2.0, window=20,
                       key=[("ink", False, "ε = 0,1: dun bij elke trek, dik het gemiddelde over de laatste 20 keer trekken")])
        _now["chart"].line("**Bezig met spelen** …")
        _run = C.demo(_yours or C.estimates, show=_progress)
        if _run == "estimates":
            _out = _now["chart"].stopped("*Je `estimates` gaf tijdens de run een fout, of geen lijst van tien getallen. Haalt je functie de check nog?*")
        elif _run == "slow":
            _out = _now["chart"].stopped("*De run duurde langer dan twee minuten. Doet je `estimates` meer werk dan nodig?*")
        else:
            _last = {_e: sum(_run["curve"][_e][-100:]) / 100 for _e in C.EPSILONS}
            _who = "je `estimates`" if _yours else "de `estimates` van de course, omdat je eigen functie de check nog niet haalt"
            _out = mo.vstack([
                L.explore("rewind", "trek", lambda _v: (C.rewind_frame(_run, int(_v), L.lang), C.rewind_text(_run, int(_v), L.lang)),
                          start=C.STEPS, values=C.LOOKS, role="",
                          task="Ga terug naar trek 1, en dan weer vooruit: wanneer stopt greedy met automaten van de eerste bandit proberen, en wanneer vindt ε = 0,1 de beste?"),
                mo.md(f"Gespeeld met {_who}. Over de laatste 100 keer trekken kreeg greedy {C.number(_last[0.0], 2, L.lang)} per keer, ε = 0,01 {C.number(_last[0.01], 2, L.lang)} en ε = 0,1 {C.number(_last[0.1], 2, L.lang)}."),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    De 2.000 bandits van de course vertellen hetzelfde verhaal. Over de eerste 20 keer trekken krijgt greedy 0,74 per keer en $\varepsilon$ = 0,1 krijgt 0,69: exploration kost eerst reward. Rond trek 100 ligt $\varepsilon$ = 0,1 voor, 1,18 tegen 1,03. Over de laatste 100 keer trekken krijgt $\varepsilon$ = 0,1 gemiddeld 1,36, $\varepsilon$ = 0,01 krijgt 1,31 en greedy 1,04. Als een agent de gemiddelden kende en altijd aan de beste automaat trok, kreeg de agent 1,53: het gemiddelde van de beste automaat over de 2.000 bandits. Greedy trekt aan het eind in 27,5% van de bandits aan automaat 0, de eerste automaat die de agent probeerde, terwijl automaat 0 in slechts 9,1% de beste is.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_best_share():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Hoe vaak de beste automaat

        De gemiddelde reward is één manier om een agent te beoordelen; een andere is hoe vaak de agent aan de beste automaat trekt. `best_share(actions, best)` geeft voor elke trek terug in welk deel van de runs de agent aan de beste automaat van de run trok. `actions` heeft een rij voor elke run en een kolom voor elke trek, met de automaat waaraan getrokken is; `best` is een kolom met de beste automaat van elke run, dus `actions == best` vergelijkt elke trek van een run met de beste automaat van de run, zoals in [Broadcasting](/course/open/np-broadcasting?lang=nl). Vervang de twee keer `...`: `hits`, `True` waar een trek de beste automaat nam, en het aandeel `True` in elke kolom.

        De check roept `best_share(small_actions, small_best)` aan, met drie runs van vier keer trekken, wat de aandelen 0, 1/3, 2/3 en 1 moet geven, en daarna de runs van de course met 2.000 bandits:

        ```python
        small_actions = np.array([[0, 2, 2, 2],
                                  [1, 1, 0, 0],
                                  [2, 2, 2, 1]])
        small_best = np.array([[2], [0], [1]])
        ```
        """),
        L.task("Laat `best_share` voor elke trek het deel van de runs teruggeven waarin aan de beste automaat getrokken werd."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best_share():
    box_best_share = L.code_box("best_share", lines=4)
    box_best_share
    return (box_best_share,)


@app.cell(hide_code=True)
def check_best_share(box_best_share):
    result_best_share = L.check("best_share", box_best_share.value)
    result_best_share
    return (result_best_share,)


@app.cell(hide_code=True)
def hints_best_share():
    L.hints(
        "best_share",
        [
            "`hits = actions == best` bevat `True` voor elke trek die de beste automaat van de run nam. Een aandeel `True` is een gemiddelde, want `True` telt als 1 en `False` als 0: `hits` heeft er een methode voor, met een axis.",
            "In [Sommen en gemiddelden langs een axis](/course/open/np-axis?lang=nl) ging `axis=0` langs de rijen omlaag: één uitkomst per kolom. Hier is een kolom een trek, en langs de rijen omlaag ga je over de runs.",
            "`    return hits.___`",
        ],
        why="`return hits.mean(axis=0)`: het gemiddelde langs elke kolom, over de runs, geeft één aandeel per trek. `np.mean(actions == best, axis=0)` geeft hetzelfde.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_long():
    mo.md(r"""
    ### Oefening 3. Een langere run

    Na 1.000 keer trekken ligt $\varepsilon$ = 0,1 voor op $\varepsilon$ = 0,01: over de laatste 100 keer trekt de agent in 79,5% van de bandits aan de beste automaat tegen 59,9%, en krijgt de agent 1,36 per keer tegen 1,31. De course liet beide agents op dezelfde 2.000 bandits 10.000 keer trekken.
    """)
    return


@app.cell(hide_code=True)
def choice_long_cell():
    choice_long = L.choice(
        "long-run",
        question="Welke agent krijgt na 10.000 keer trekken meer reward per keer over de laatste 1.000 keer?",
        options={
            "same01": "ε = 0,1, net als over de eerste 1.000 keer",
            "right": "ε = 0,01: de agent trekt minder vaak aan een willekeurige automaat",
            "equal": "Allebei evenveel: dan kennen beide agents de beste automaat van elke bandit, dus krijgen beide evenveel per keer",
        },
        correct="right",
        why={
            "right": "Over de laatste 1.000 van 10.000 keer trekken krijgt ε = 0,01 gemiddeld 1,50 per keer en ε = 0,1 krijgt 1,38. Aan het eind kent ε = 0,1 de beste automaat vaker, in 95,3% van de bandits tegen 87,8%, maar in 1 van de 10 keer neemt ε = 0,1 een willekeurige automaat, met gemiddeld een reward van ongeveer 0: zo kan de agent nooit in meer dan 91% van de keren aan de beste automaat trekken. ε = 0,01 neemt slechts in 1 van de 100 keer een willekeurige automaat, en als de grootste schatting van ε = 0,01 niet bij de beste automaat hoort, geeft de gekozen automaat gemiddeld slechts 0,15 minder.",
            "same01": "Over de eerste 1.000 keer vindt ε = 0,1 de beste automaat sneller. Wat doet de agent daarna nog steeds in 1 van elke 10 keer?",
            "equal": "Ook waar beide de beste automaat kennen, neemt ε = 0,1 in 1 van de 10 keer een willekeurige automaat: wat kost dat per keer?",
        },
    )
    choice_long
    return (choice_long,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het: een andere ε

    De slider laat de agent de 2.000 bandits van de course spelen met een andere $\varepsilon$, van 0 tot 0,5. De grafiek toont de gemiddelde reward bij elke trek, gemiddeld over 20 keer trekken, met greedy gestreept en het gemiddelde van de beste automaat als dunne lijn.
    """)
    return


@app.cell(hide_code=True)
def try_cell():
    _out = L.explore("try-epsilon", "ε", lambda _v: C.try_frame(float(_v), L.lang), start=0.1, values=C.TRY_EPSILONS,
                     role="hyper", task="Ga van 0 naar 0,5: welke ε krijgt de meeste reward over de laatste 100 keer trekken, en waarom krijgt een grotere ε daarboven minder?")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl) exploreerde de agent met ε-greedy in een rooster van states, waar een stap ook verandert waar de agent is. Een bandit heeft één state, dus de exploration-exploitation trade-off is het hele probleem: je `estimates` geeft het steekproefgemiddelde van elke automaat, en je `best_share` laat zien hoe vaak elke agent de beste vindt. Q-learning schoof $Q$ een stap $\eta$ op naar elk nieuw target, in plaats van totalen en aantallen bij te houden. In de volgende les, *Gemiddelden bijhouden: verschuivende automaten en optimistische starts*, zie je dat een steekproefgemiddelde op dezelfde manier bij te houden is, één reward tegelijk.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_long, result_best_share, result_estimates):
    L.wrap_up(result_estimates, result_best_share, choice_long)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
