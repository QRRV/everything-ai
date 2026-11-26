# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-bandit-updates. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the bandits, the runs and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rl-bandit-updates", "nl")
    C = L.checks   # de verschuivende bandits, de agents en runs van de course, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Welke automaat geeft het meest? De multi-armed bandit](/course/open/rl-bandits?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_average():
    quick_average = L.choice(
        "quick-average",
        question="Een automaat gaf de rewards 2, 0 en 4. Wat is het steekproefgemiddelde?",
        options={
            "last": "4, de laatste reward",
            "right": "2: het totaal 6 gedeeld door de 3 keer trekken",
            "total": "6, het totaal van de drie rewards, want de schatting houdt elke reward bij",
        },
        correct="right",
        why={
            "right": "Het steekproefgemiddelde is het totaal van de rewards gedeeld door het aantal keer trekken: (2 + 0 + 4) / 3 = 2.",
            "last": "Het steekproefgemiddelde gebruikt elke reward van de automaat, niet alleen de laatste: wat krijg je als je het totaal deelt door het aantal keer trekken? Zie [Welke automaat geeft het meest? De multi-armed bandit](/course/open/rl-bandits?lang=nl).",
            "total": "Het totaal groeit met elke trek, en de schatting deelt het totaal ergens door: waardoor? Zie [Welke automaat geeft het meest? De multi-armed bandit](/course/open/rl-bandits?lang=nl).",
        },
    )
    quick_average
    return (quick_average,)


@app.cell(hide_code=True)
def quick_check_greedy():
    quick_greedy = L.choice(
        "quick-greedy",
        question="Een greedy agent trok één keer aan automaat 0 van drie en kreeg 0,8, dus de schattingen zijn 0,8, 0 en 0. Aan welke automaat trekt de agent nu?",
        options={
            "right": "Weer aan automaat 0, want de schatting is het grootst",
            "other": "Aan automaat 1 of 2, om te ontdekken wat er uitkomt",
            "random": "Aan een willekeurige van de drie, want de agent heeft elke automaat hooguit één keer geprobeerd en weet nog te weinig",
        },
        correct="right",
        why={
            "right": "Greedy neemt altijd de grootste schatting: 0,8 tegen 0 en 0, dus weer automaat 0. Automaat 1 of 2 probeert de agent pas als automaat 0 onder 0 zakt.",
            "other": "Een greedy agent trekt nooit aan een automaat alleen om te zien wat er uitkomt: welke automaat heeft de grootste schatting? Zie [Welke automaat geeft het meest? De multi-armed bandit](/course/open/rl-bandits?lang=nl).",
            "random": "Alleen ε-greedy neemt af en toe een willekeurige automaat, en greedy is ε-greedy met $\\varepsilon$ = 0: welke automaat neemt de agent? Zie [Welke automaat geeft het meest? De multi-armed bandit](/course/open/rl-bandits?lang=nl).",
        },
    )
    quick_greedy
    return (quick_greedy,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De nieuwssite met vier koppen heeft een probleem dat de bandits uit [Welke automaat geeft het meest? De multi-armed bandit](/course/open/rl-bandits?lang=nl) niet hadden: een kop met veel klikken in de ochtend krijgt er 's avonds minder, als het nieuws verder is. Een steekproefgemiddelde telt een klik van vorige week even zwaar als een klik van een minuut geleden, dus blijft de site tonen wat vroeger werkte. Prijzen veranderen, smaken veranderen, en de mensen voor wie een agent werkt veranderen; in zo'n wereld moet een agent oude rewards minder laten meetellen.

    **Een voorbeeld uit het dagelijks leven.** Je favoriete café heeft een nieuwe kok, en de laatste drie maaltijden vielen tegen. Zegt het gemiddelde van al je bezoeken door de jaren heen of je terug moet gaan, of zeggen de laatste paar maaltijden meer?

    Er is ook iets kleiners te leren: een steekproefgemiddelde heeft helemaal geen totalen nodig. Het is met één reward tegelijk bij te houden, met dezelfde update als bij Q-learning.
    """)
    return


@app.cell(hide_code=True)
def predict_drift():
    guess_drift = L.guess(
        "drift",
        question="In de bandits van deze les zet elk gemiddelde na elke trek een kleine willekeurige stap: de gemiddelden verschuiven. De course laat twee agents met ε-greedy en $\\varepsilon$ = 0,1 elk 2.000 van zulke bandits spelen, met 10.000 keer trekken. De ene houdt het steekproefgemiddelde bij; de andere schuift de schatting een tiende van de weg op naar elke nieuwe reward. Hoe vaak trekt elke agent over de laatste 1.000 keer trekken aan de automaat die op dat moment de beste is?",
        options={
            "same": "Allebei ongeveer 80 op de 100 keer: de rewards zijn dezelfde",
            "right": "De agent die een tiende opschuift ongeveer 80 op de 100 keer, het steekproefgemiddelde ongeveer 60",
            "average": "Het steekproefgemiddelde ongeveer 80 op de 100 keer, want het gebruikt elke reward en heeft dus de minste ruis in de schattingen",
        },
        reveal="**De agent die een tiende opschuift ongeveer 80 op de 100 keer, het steekproefgemiddelde ongeveer 60.** Over de laatste 1.000 van de 10.000 keer trekken trekt de agent die een tiende van de weg opschuift naar elke reward in 80,5% van de keren aan de beste automaat, het steekproefgemiddelde in 59,9%. *Het idee* laat zien waarom, en de demo speelt de run opnieuw op 300 bandits.",
    )
    guess_drift
    return (guess_drift,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        In [Welke automaat geeft het meest? De multi-armed bandit](/course/open/rl-bandits?lang=nl) hield de agent voor elke automaat een totaal en een aantal bij, en deelde het ene door het andere. Hetzelfde steekproefgemiddelde is bij te houden met één getal per automaat. Stel dat een automaat tot nu toe $n - 1$ rewards gaf, met de schatting $Q$, en nu een reward $r$ geeft, de $n$-de. Het nieuwe steekproefgemiddelde is het oude, een deel $1/n$ van de weg opgeschoven naar $r$:
        """),
        L.formula(r"[[param:Q]] \leftarrow [[param:Q]] + [[hyper:\eta]] \left( [[data:r]] - [[param:Q]] \right)"),
        mo.md(r"""
        Hier is $r - Q$ hoe ver de reward van de schatting ligt, en $\eta$ is de **learning rate**: welk deel van de afstand de schatting opschuift. Het is de update van [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl), met de reward als target. Met $\eta = 1/n$ geeft de update precies het steekproefgemiddelde. Voor de rewards 2, 0 en 4 uit de instapcheck gaat de schatting van 0 naar 0 + 1 · (2 − 0) = 2, dan naar 2 + ½ · (0 − 2) = 1, dan naar 1 + ⅓ · (4 − 1) = 2: het totaal 6 gedeeld door 3, zonder totaal. Dit is een **incremental update**: geen totalen en geen lijst van rewards, één getal per automaat dat met elke reward opschuift.

        Met een vaste $\eta$, zoals 0,1, is elke stap hetzelfde deel van de afstand. Elke update houdt $1 - \eta$ van de oude schatting en telt er $\eta$ van de nieuwe reward bij, dus wordt bij elke update het deel van elke oudere reward in de schatting vermenigvuldigd met $1 - \eta$: tien updates later telt een reward nog slechts $0{,}9^{10} \approx 0{,}35$ van wat de reward bij binnenkomst meetelde. Recente rewards tellen zwaarder. Oefening 2 rekent uit hoeveel elke reward meetelt.

        De afbeelding toont een automaat waarvan het gemiddelde na 100 keer trekken van 0 naar 2 springt, met beide schattingen.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        Na 200 keer trekken staat het steekproefgemiddelde op 1,05: het middelt nog steeds de 100 rewards rond 0 met de 100 rond 2, en elke nieuwe reward schuift het slechts 1/200 van de weg op. De schatting met $\eta$ = 0,1 staat op 2,11. Ongeveer 20 keer trekken na de sprong zit de schatting bij 2, en vanaf dan beweegt de schatting met elke reward rond 2: de schatting komt nooit zo dicht bij het gemiddelde als een steekproefgemiddelde bij een automaat die gelijk blijft.

        Een **nonstationary problem** is een probleem waarvan de rewards in de loop van de tijd veranderen. In de bandits van deze les zet elk gemiddelde van elke automaat na elke trek een willekeurige stap met een standaardafwijking van 0,01. Na 10.000 keer trekken ligt een gemiddelde doorgaans ongeveer 1 van de beginwaarde af, zo ver als de gemiddelden van een bandit aan het begin uit elkaar liggen.

        Tot nu toe begon elke schatting bij 0. Met $\eta = 1/n$ is de startwaarde na de eerste reward verdwenen: de eerste update heeft $\eta$ = 1 en zet de schatting op de eerste reward. Met $\eta$ = 0,1 verdwijnt de startwaarde nooit helemaal: na $n$ updates telt de startwaarde nog $(1 - \eta)^n$ mee. Daar is iets mee te doen. Laat elke schatting beginnen bij 5, ver boven wat een automaat geeft. De eerste reward van een automaat valt tegen en de schatting zakt, dus gaat zelfs een greedy agent door naar een automaat die de agent nog niet probeerde, nog op 5. Voordat de agent bij één automaat blijft, heeft de agent alle automaten geprobeerd. Dit zijn **optimistic initial values**: exploration die uit de startwaarden komt, niet uit trekken aan willekeurige automaten.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **incremental update** | | een schatting een deel $\eta$ van de weg opschuiven naar elke nieuwe reward: één getal per automaat, geen totalen | `update` |
        | **nonstationary problem** | | een probleem waarvan de rewards in de loop van de tijd veranderen, zoals bandits met verschuivende gemiddelden | |
        | **optimistic initial values** | | schattingen die beginnen boven wat een action geeft, zodat zelfs een greedy agent elke action probeert | `start` |
        | $\eta$ | èta | de learning rate: $1/n$ geeft het steekproefgemiddelde, een vaste laat recente rewards zwaarder tellen | `eta` |
        | $Q$ | Q | de schatting van één automaat | `q` |
        | $r$ | r | de reward van de nieuwste trek | `reward` |
        | $n$ | n | het aantal rewards van de automaat tot nu toe, de nieuwste meegeteld | `n` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een agent met incremental updates op een bandit:

    1. Geef elke automaat een schatting, 0 of een startwaarde boven elke reward, en een aantal van 0.
    2. Kies een automaat, ε-greedy: met kans $\varepsilon$ een willekeurige, anders de automaat met de grootste schatting.
    3. Trek eraan: de bandit geeft een reward $r$.
    4. Tel 1 op bij het aantal $n$ van de automaat, en neem de learning rate: $\eta = 1/n$ voor het steekproefgemiddelde, een vaste zoals 0,1 voor een probleem dat verandert.
    5. Schuif de schatting op: $Q \leftarrow Q + \eta \, (r - Q)$.
    6. Ga verder bij stap 2, tot de laatste trek.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eén automaat geeft de rewards 2,0; 0,0; 4,0; 1,0 en 3,0. De cel neemt de rewards twee keer door stap 4 en 5: met $\eta = 1/n$ en met de vaste $\eta$ = 0,5. Elke regel toont het aantal, de reward en beide schattingen. Voer de cel uit.
    """)
    return


@app.cell
def example_updates():
    # de rewards van één automaat, in de volgorde van binnenkomst
    _rewards = [2.0, 0.0, 4.0, 1.0, 3.0]

    # 1. beide schattingen beginnen bij 0, en het aantal bij 0
    _q_average = 0.0
    _q_constant = 0.0
    _n = 0
    for _r in _rewards:
        # 4. één reward meer; de learning rate is 1/n, of de vaste 0.5
        _n = _n + 1
        # 5. elke schatting schuift een deel van de weg op naar de reward
        _q_average = _q_average + (1 / _n) * (_r - _q_average)
        _q_constant = _q_constant + 0.5 * (_r - _q_constant)
        print(_n, _r, _q_average, _q_constant)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Met $\eta = 1/n$ is de schatting na elke reward het steekproefgemiddelde: 2,0, dan (2 + 0) / 2 = 1,0, dan 6 / 3 = 2,0, dan 7 / 4 = 1,75, en na vijf rewards 10 / 5 = 2,0, zonder totaal. Met $\eta$ = 0,5 schuift elke reward de schatting halverwege op: 1,0; 0,5; 2,25; 1,625 en 2,3125. De schatting eindigt boven het steekproefgemiddelde: de laatste reward, 3,0, telt voor de helft mee, en de 4,0 telt twee keer zo zwaar als de 0,0 die eraan voorafging.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_update():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Eén reward tegelijk

        `update(q, reward, eta)` geeft de nieuwe schatting terug: de schatting `q`, een deel `eta` van de weg opgeschoven naar `reward`, stap 5 van het recept. Vervang de `...`. De check roept `update(1.0, 3.0, 0.5)` aan, wat `2.0` moet geven, en zes andere gevallen, met learning rates van $1/n$ en vaste.
        """),
        L.task("Laat `update` de schatting `q` teruggeven, een deel `eta` van de weg opgeschoven naar `reward`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_update():
    box_update = L.code_box("update", lines=4)
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
            "Stap 5 van het recept: de nieuwe schatting is de oude, `q`, plus `eta` keer hoe ver `reward` van `q` ligt.",
            "Bij `update(1.0, 3.0, 0.5)` ligt de reward 3,0 − 1,0 = 2,0 boven de schatting. De helft is 1,0, dus de nieuwe schatting is 1,0 + 1,0 = 2,0.",
            "`    return q + eta * ___`",
        ],
        why="`return q + eta * (reward - q)`: de afstand van de schatting tot de reward, een deel `eta` daarvan, opgeteld bij de schatting. `(1 - eta) * q + eta * reward` geeft hetzelfde.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Laat twee agents spelen op verschuivende bandits

    De knop laat twee agents met $\varepsilon$ = 0,1 dezelfde 300 verschuivende bandits spelen, elk met 10.000 keer trekken: één met $\eta = 1/n$, het steekproefgemiddelde, en één met $\eta$ = 0,1. De agents gebruiken allebei je `update` zodra je functie de check haalt, en tot dan de functie van de course. De run van één agent gaat zo; `choose` is de ε-greedy van de multi-armed bandit:

    ```python
    for t in range(10000):
        for b in range(300):   # elke bandit
            a = choose(q[b], epsilon, rng)
            r = means[b][a] + rng.normal()
            counts[b][a] = counts[b][a] + 1
            q[b][a] = update(q[b][a], r, 1 / counts[b][a])   # de andere agent: 0.1
        means = means + rng.normal(0, 0.01, (300, 10))   # elk gemiddelde verschuift een beetje
    ```

    De grafiek toont hoe vaak de agent met $\eta$ = 0,1 aan de beste automaat van dat moment trekt: het deel van de 300 bandits over elke 10 keer trekken, dun, en het gemiddelde over de laatste 100 keer trekken, dik; gestreept hetzelfde gemiddelde voor $\eta = 1/n$. Ernaast zie je wat beide agents weten van één van de 300 bandits: de gemiddelden van dat moment als korte streepjes, de schattingen in <span class="c-param">oranje</span>, en hoe vaak er aan elke automaat getrokken is. Het duurt ongeveer tien seconden; daarna kun je terug door de run.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    run_button = mo.ui.run_button(label="Laat de twee agents spelen met mijn update")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def demo_run(box_update, result_update, run_button):
    if not run_button.value:
        _out = mo.md("*Druk op de knop om de twee agents te laten spelen.*")
    else:
        _yours = C.mine(box_update.value, result_update.state == "correct")
        _w = C.live_words(L.lang)
        _now = {}   # de live grafiek, wat de agents na de laatste trek van één bandit weten, en de trek
        _raw = {"average": [], "constant": []}   # het aandeel beste automaten van elke agent bij elke trek

        def _progress(_t, _shares, _seen):
            if "seen" not in _now:   # vanaf de eerste trek: wat de agents weten, naast de grafiek
                _now["chart"].watch(lambda: C.now_picture(_now["seen"], _now["t"], L.lang))
            _now["seen"], _now["t"] = _seen, _t
            _raw["average"].append(_shares["average"])
            _raw["constant"].append(_shares["constant"])
            if _t % 500 == 0 or _t == C.DEMO_STEPS:
                _now["chart"].reference(C.smoothed(_raw["average"]), "1/n", role="ink")
            if _t % 10 == 0:
                _now["chart"].add(_t / 1000, C.mean_of(_raw["constant"], _t - 9, _t),
                                  f"**Trek {C.number(_t, 0, L.lang)} van {C.number(C.DEMO_STEPS, 0, L.lang)}** · twee agents op 300 verschuivende bandits")

        _now["chart"] = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"], "alt": _w["alt"]},
                       C.DEMO_STEPS / 1000, role="ink", low=0.0, high=1.0, window=10,
                       key=[("ink", False, "η = 0,1: dun over elke 10 keer trekken, dik het gemiddelde over de laatste 100 keer trekken")])
        _now["chart"].line("**Bezig met spelen** …")
        _run = C.demo(_yours or C.update, show=_progress)
        if _run == "update":
            _out = _now["chart"].stopped("*Je `update` gaf tijdens de run een fout, of geen getal. Haalt je functie de check nog?*")
        elif _run == "slow":
            _out = _now["chart"].stopped("*De run duurde langer dan twee minuten. Doet je `update` meer werk dan nodig?*")
        else:
            _last = {_a: C.mean_of(_run["curve"][_a], C.DEMO_STEPS - 999, C.DEMO_STEPS) for _a in C.AGENTS}
            _who = "je `update`" if _yours else "de `update` van de course, want je eigen functie haalt de check nog niet"
            _out = mo.vstack([
                L.explore("rewind", "trek", lambda _v: (C.rewind_frame(_run, int(_v), L.lang), C.rewind_text(_run, int(_v), L.lang)),
                          start=C.DEMO_STEPS, values=C.LOOKS, role="",
                          task="Ga terug naar trek 1, en dan weer vooruit: vanaf welke trek trekt η = 0,1 vaker aan de beste automaat, en hoe ver liggen de schattingen van 1/n aan het eind van de gemiddelden af?"),
                mo.md(f"Gespeeld met {_who}. Over de laatste 1.000 keer trekken trok de agent met η = 0,1 in {C.percent(_last['constant'], L.lang)} van de keren aan de beste automaat, de agent met 1/n in {C.percent(_last['average'], L.lang)}."),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    De 2.000 bandits van de course vertellen hetzelfde verhaal. Over de eerste 1.000 keer trekken ligt het steekproefgemiddelde voor: de agent trekt in 67,7% van de keren aan de beste automaat tegen 61,7%, en krijgt 1,32 per keer tegen 1,27. Zolang de gemiddelden nauwelijks verschoven zijn, geven meer rewards een betere schatting. Vanaf ongeveer trek 1.000 ligt $\eta$ = 0,1 voor, en over de laatste 1.000 van de 10.000 keer trekken trekt het steekproefgemiddelde in slechts 59,9% van de keren aan de beste automaat: bij een automaat waaraan de agent duizenden keren trok, schuift de schatting bij een nieuwe reward minder dan een duizendste van de weg op. De agents krijgen aan het eind allebei meer per keer dan over de eerste 1.000 keer trekken, 1,93 voor $\eta$ = 0,1 en 1,77 voor het steekproefgemiddelde: doordat de gemiddelden verschuiven, gaan de gemiddelden verder uit elkaar liggen, en geeft de beste automaat van een bandit meer.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_reward_shares():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Hoeveel elke reward meetelt

        Met een vaste $\eta$ bestaat de schatting na $n$ rewards uit de rewards zelf, elk met een aandeel, en wat er over is van de startwaarde. De nieuwste reward telt voor $\eta$. Elke update na een reward vermenigvuldigt het aandeel van de reward met $1 - \eta$.

        `reward_shares(eta, n)` geeft een array terug met het aandeel van elk van `n` rewards in de schatting, de eerste reward eerst. `i = np.arange(1, n + 1)` bevat de nummers van de rewards, 1 tot en met `n`. Vervang de `...`. De check roept `reward_shares(0.5, 3)` aan, wat `[0.125, 0.25, 0.5]` moet geven, en vijf andere gevallen.
        """),
        L.task("Laat `reward_shares` het aandeel van elke reward teruggeven in de schatting na `n` updates met de learning rate `eta`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_reward_shares():
    box_reward_shares = L.code_box("reward_shares", lines=5)
    box_reward_shares
    return (box_reward_shares,)


@app.cell(hide_code=True)
def check_reward_shares(box_reward_shares):
    result_reward_shares = L.check("reward_shares", box_reward_shares.value)
    result_reward_shares
    return (result_reward_shares,)


@app.cell(hide_code=True)
def hints_reward_shares():
    L.hints(
        "reward_shares",
        [
            "Reward `i` krijgt `eta` bij binnenkomst, en elke update daarna vermenigvuldigt het aandeel met 1 − η. `**` verheft een getal tot een macht, en werkt in één keer op de hele array `i`.",
            "Hoeveel updates volgen er op reward `i` als er `n` rewards zijn? Voor `reward_shares(0.5, 3)`: na de laatste reward, nummer 3, komt geen update meer, dus telt de reward voor 0,5. Na de eerste komen er twee: 0,5 × 0,5 × 0,5 = 0,125.",
            "`    return eta * ___`",
        ],
        why="`return eta * (1 - eta) ** (n - i)`: `eta` voor de reward zelf, en een factor 1 − η voor elk van de `n - i` updates erna, voor alle rewards in één keer.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_spike():
    mo.md(r"""
    ### Oefening 3. Greedy, en toch exploration

    De course liet een greedy agent met $\eta$ = 0,1 spelen op 2.000 bandits waarvan de gemiddelden gelijk blijven, met 1.000 keer trekken en elke schatting beginnend bij 5. Ernaast speelde een agent die bij 0 begint en exploration met ε-greedy doet, $\varepsilon$ = 0,1, ook met $\eta$ = 0,1. De optimistische agent krijgt eerst minder: 0,34 per keer over de eerste 100 keer trekken, tegen 0,96. Over de laatste 100 keer trekken trekt de agent in 86,4% van de bandits aan de beste automaat tegen 76,6%, en over alle 1.000 keer trekken krijgt de agent 1,29 per keer tegen 1,24. Rond trek 11 gebeurt er iets vreemds: het deel van de bandits waarin de agent aan de beste automaat trekt is 10,0% bij trek 10, springt naar 41,5% bij trek 11, en zakt terug naar 26,1% bij trek 12.
    """)
    return


@app.cell(hide_code=True)
def choice_spike_cell():
    choice_spike = L.choice(
        "spike",
        question="Waarom trekt de optimistische agent bij trek 11 in 41,5% van de bandits aan de beste automaat, en bij trek 12 in slechts 26,1%?",
        options={
            "lucky": "Toevallig: de rewards van trek 11 zijn net groot",
            "right": "Na tien keer trekken is elke automaat één keer geprobeerd, de beste heeft vaker dan elke andere automaat de grootste schatting, en nog één keer trekken laat de schatting onder een andere zakken",
            "learned": "Bij trek 11 weet de agent welke automaat de beste is, en bij trek 12 begint de agent weer aan exploration door aan willekeurige automaten te trekken, zoals een agent met ε-greedy zou doen nadat de beste automaat gevonden is",
        },
        correct="right",
        why={
            "right": "Elke schatting begint bij 5. Een eerste reward $r$ brengt een automaat op 5 + 0,1 · ($r$ − 5) = 4,5 + 0,1$r$, onder de automaten die nog niet geprobeerd zijn, dus probeert de agent in de eerste tien keer trekken elke automaat één keer. Na trek 10 heeft de automaat met de grootste eerste reward de grootste schatting, en dat is vaker de beste automaat dan een andere: in 41,5% van de bandits. Nog een keer trekken brengt de schatting gemiddeld omlaag naar ongeveer 4,35, onder de grootste van de negen andere, dus gaat de agent bij trek 12 meestal door naar een andere automaat. Pas na veel keer trekken aan elke automaat telt de startwaarde niet meer mee.",
            "lucky": "Het deel gaat over 2.000 bandits: kan toeval alleen het deel in één trek van 10,0% naar 41,5% tillen, en bij de volgende trek weer laten zakken?",
            "learned": "De agent is greedy, dus trekt nooit aan een willekeurige automaat: wat gebeurt er met de schatting van een automaat elke keer dat eraan getrokken wordt, zolang elke schatting nog ver boven elke reward ligt?",
        },
    )
    choice_spike
    return (choice_spike,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Probeer het: een andere startwaarde

    De slider laat de greedy agent met $\eta$ = 0,1 de 2.000 bandits van de course spelen met een andere startwaarde, van 0 tot 10. De grafiek toont hoe vaak de agent aan de beste automaat trekt, gemiddeld over 10 keer trekken, met de agent die bij 0 begint en exploration met $\varepsilon$ = 0,1 doet gestreept. Een start hoeft niet boven elke reward te liggen: boven de meeste eerste rewards is genoeg om de agent elke automaat te laten proberen. Hoe hoger de start, hoe langer de agent blijft teruggaan naar automaten die al weinig gaven.
    """)
    return


@app.cell(hide_code=True)
def try_cell():
    _out = L.explore("try-start", "start", lambda _v: C.try_frame(float(_v), L.lang), start=5.0, values=C.STARTS,
                     role="hyper", task="Ga van 0 naar 10: welke startwaarde krijgt de meeste reward over alle 1.000 keer trekken, en vanaf welke start vindt de agent aan het eind in meer dan 85% van de bandits de beste automaat?")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Welke automaat geeft het meest? De multi-armed bandit](/course/open/rl-bandits?lang=nl) hield de agent totalen en aantallen bij. Je `update` houdt hetzelfde steekproefgemiddelde bij met één getal per automaat, en met een vaste learning rate kan je `update` bandits met verschuivende gemiddelden volgen: zoals je `reward_shares` laat zien, vervagen oude rewards. Het is de update van [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl), met de reward als target, en met de startwaarden kan zelfs een greedy agent aan exploration doen. In de volgende les, *Project: je eigen bandit-agent*, bouw je een agent in je eigen projectmap en vergelijk je een greedy, een ε-greedy en een optimistische versie op verschuivende bandits.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_spike, result_reward_shares, result_update):
    L.wrap_up(result_update, result_reward_shares, choice_spike)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
