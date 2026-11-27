# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-sarsa. Route: course/route/lessen/reinforcement-learning.yaml. Checks, het klif van de course, de runs van SARSA en Q-learning, en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-sarsa", "nl")
    C = L.checks   # het klif van de course, de runs van SARSA en Q-learning, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Leren na elke stap: TD(0)](/course/open/rl-td-prediction?lang=nl) en [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_td():
    quick_td = L.choice(
        "quick-td",
        question="Een stap gaat van $S$ naar $S'$ met reward 0 en $\\gamma$ = 1, terwijl $V(S)$ = 0,5 en $V(S')$ = 0,6. Wat is de TD error $\\delta$?",
        options={
            "target": "0,6: het target zelf",
            "right": "0,1: het target min de oude schatting",
            "minus": "−0,1: de oude schatting min het target",
            "add": "1,1: de reward, het target en de oude schatting samen",
        },
        correct="right",
        why={
            "right": "$\\delta = R + \\gamma V(S') - V(S) = 0 + 0{,}6 - 0{,}5 = 0{,}1$: de stap ging iets beter dan $V(S)$ voorspelde. Deze les gebruikt hetzelfde soort target voor $Q$.",
            "target": "Het target is $R + \\gamma V(S') = 0{,}6$, maar de TD error zegt hoe ver de oude schatting ervan af ligt: wat haal je eraf? Zie [Leren na elke stap: TD(0)](/course/open/rl-td-prediction?lang=nl).",
            "minus": "Het teken telt: een stap die beter gaat dan voorspeld moet $V(S)$ omhoog duwen. Wat komt eerst? Zie [Leren na elke stap: TD(0)](/course/open/rl-td-prediction?lang=nl).",
            "add": "De TD error is een verschil, geen totaal: welke twee getallen vergelijkt de TD error? Zie [Leren na elke stap: TD(0)](/course/open/rl-td-prediction?lang=nl).",
        },
    )
    quick_td
    return (quick_td,)


@app.cell(hide_code=True)
def quick_check_q():
    quick_q = L.choice(
        "quick-q",
        question="Een agent in Q-learning kiest in state $s$ action $a$, krijgt reward $r$ en komt in $s'$. Welk getal van $s'$ gaat in het target $r + \\gamma \\cdot \\ldots$?",
        options={
            "next": "De $Q$ van de action die de agent als volgende neemt in $s'$",
            "right": "De grootste $Q$ in $s'$, over alle actions",
            "mean": "Het gemiddelde van de vier $Q$'s in $s'$, omdat de agent elk ervan kan nemen",
            "table": "$V(s')$ uit de tabel `T`",
        },
        correct="right",
        why={
            "right": "Q-learning neemt $\\max_{a'} Q(s', a')$: de waarde van $s'$ als de agent daar de beste action neemt, wat de agent ook als volgende doet. De nieuwe methode van deze les neemt wel de action die als volgende komt.",
            "next": "Q-learning kijkt niet naar de action die als volgende komt: welke $Q$ van $s'$ kiest de max? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
            "mean": "Q-learning neemt één $Q$ van $s'$, niet alle vier: welke kiest de max? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
            "table": "Q-learning heeft geen tabel met kansen nodig, alleen de Q-table die de agent leert: welke $Q$ van $s'$ gebruikt Q-learning? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
        },
    )
    quick_q
    return (quick_q,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een agent die leert door te proberen maakt met opzet fouten: met ε-greedy neemt de agent af en toe een willekeurige action, om te zien wat er gebeurt. In een simulatie kost dat niets; een robot die in een echt gebouw leert, naast een trap, betaalt wel voor elke willekeurige stap. Het maakt dus uit wat de agent leert: wat de actions waard zijn als de agent altijd de beste neemt, of wat de actions waard zijn met de eigen willekeurige stappen erin. Q-learning doet het eerste, en SARSA, de methode van deze les, het tweede. Op dezelfde taak leren de twee een andere weg. Het verschil heeft een naam die je overal in RL tegenkomt, on-policy tegenover off-policy, en bepaalt van welke data een methode kan leren.

    **Een voorbeeld uit het dagelijks leven.** Je loopt langs een gracht met je ogen op je telefoon, dus af en toe zet je een stap zonder te kijken. De kortste weg loopt vlak langs het water. Plan je alsof je altijd op tijd opkijkt, dan loop je langs de rand; plan je met je eigen onoplettende stappen erbij, dan loop je een meter verder van het water. Beide plannen kloppen, elk voor een andere wandelaar.
    """)
    return


@app.cell(hide_code=True)
def predict_safe():
    guess_safe = L.guess(
        "safe",
        question="Een rooster van 4 bij 12 heeft onderaan een klif, tussen de start en het doel. Elke stap geeft −1; een stap in het klif geeft −100 en zet de agent terug op de start. SARSA en Q-learning leren allebei 500 episodes lang, met ε-greedy, ε = 0,1, en η = 0,5. Welke methode heeft over de laatste 100 episodes, gemiddeld over 50 runs, de hoogste gemiddelde return per episode tijdens het leren?",
        options={
            "q": "Q-learning: de methode leert de kortste weg, 13 stappen langs de rand, dus de episodes kosten het minst",
            "same": "Ongeveer gelijk: na 400 episodes kennen beide het klif, dus lopen beide hetzelfde",
            "right": "SARSA: de episodes kosten ongeveer half zoveel",
            "fall": "Geen van beide doet het goed: met ε = 0,1 blijven beide vallen, ongeveer één keer per episode",
        },
        reveal="**SARSA: de episodes kosten ongeveer half zoveel.** Over de laatste 100 episodes is de gemiddelde return van SARSA −28 en van Q-learning −50, en SARSA heeft de hoogste in 49 van de 50 runs. Q-learning heeft de kortste weg geleerd, 13 stappen langs de rand, maar met ε = 0,1 is één op de tien stappen willekeurig, en naast de rand valt een willekeurige stap omlaag in het klif: Q-learning valt in één op de drie episodes. SARSA loopt over de bovenste rij, 17 stappen, en valt in één op de twintig episodes.",
    )
    guess_safe
    return (guess_safe,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Het klif.** Een rooster van 4 rijen en 12 kolommen. De agent begint in S, linksonder, en de episode eindigt in G, rechtsonder. De tien vakjes ertussen zijn het klif. In elk vakje kan de agent omhoog, omlaag, naar links of naar rechts; aan de rand van het rooster blijft de agent staan. Elke stap geeft een reward van −1, en een stap in het klif geeft −100 en zet de agent terug op S, waar de episode verdergaat; $\gamma$ = 1. De return van een episode is dan min het aantal stappen, met 99 extra eraf voor elke val. De kortste weg is omhoog, 11 stappen naar rechts en omlaag: 13 stappen, een return van −13. In code is een state (rij, kolom), geteld vanaf linksboven: S is (3, 0), G is (3, 11), en het klif loopt van (3, 1) tot en met (3, 10). Beide methoden beginnen met elke $Q$ op 0 en leren met ε-greedy, ε = 0,1, en met learning rate $\eta$ = 0,5.

        **SARSA.** Q-learning schuift $Q(S, A)$ op naar $R + \gamma \max_{a'} Q(S', a')$: de waarde van $S'$ als de agent daar de beste action neemt. SARSA kiest eerst met ε-greedy de action $A'$ die de agent echt neemt in $S'$, en schuift $Q(S, A)$ op naar $R + \gamma Q(S', A')$:
        """),
        L.formula(r"[[param:Q(S, A)]] \leftarrow [[param:Q(S, A)]] + [[hyper:\eta]] \left( [[data:R]] + [[hyper:\gamma]] [[param:Q(S', A')]] - [[param:Q(S, A)]] \right)"),
        mo.md(r"""
        De naam somt op wat één update nodig heeft: $S$, $A$, $R$, $S'$ en $A'$. Daarna neemt de agent $A'$, dus de action in het target is de action die de agent als volgende doet.

        **On-policy en off-policy.** Tijdens het leren volgt de agent ε-greedy. SARSA leert de waarden van precies deze policy, met alle willekeurige stappen: SARSA is **on-policy**. Q-learning leert de waarden van altijd de beste action nemen, terwijl de agent ε-greedy volgt: Q-learning is **off-policy**, en leert over een andere policy dan de policy die de agent volgt. Naast het klif maakt dat het verschil. Eén op de tien stappen is willekeurig, en één op de vier willekeurige stappen gaat omlaag: langs de rand valt de agent één keer per 40 stappen. Het target van SARSA bevat soms de −100 van zo'n val, dus SARSA leert dat de vakjes langs de rand minder waard zijn, en blijft er uit de buurt. Het target van Q-learning neemt de grootste $Q$, nooit de stap omlaag: Q-learning leert de korte weg, alsof de agent nooit meer een willekeurige stap zet.

        ### Nieuwe woorden en tekens

        | Woord of teken | Zeg je als | Wat het betekent | In code |
        |---|---|---|---|
        | **SARSA** | sarsa | TD learning van $Q(s, a)$ richting $R + \gamma Q(S', A')$, met $A'$ de action die de agent echt als volgende neemt | `sarsa_update` |
        | **on-policy** | | de waarden leren van de policy die de agent tijdens het leren volgt, willekeurige stappen meegeteld | |
        | **off-policy** | | de waarden leren van een andere policy dan de policy die de agent volgt, zoals altijd de beste action nemen | |
        | $A'$ | A accent | de action die de agent neemt in $S'$ | `nxt_action` |
        | $S'$ | S accent | de state na de stap | `nxt` |
        """),
        L.watch_out("SARSA heeft $A'$ nodig om bij te werken, dus SARSA kiest $A'$ eerst en neemt $A'$ dan echt bij de volgende stap. Zou SARSA na de update opnieuw kiezen, dan zat er in het target een action die de agent nooit nam."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    SARSA en Q-learning op het klif, allebei met ε = 0,1 en $\eta$ = 0,5, 50 runs van elk 500 episodes. Links de gemiddelde return van de laatste 10 episodes over de 50 runs: SARSA doorgetrokken, Q-learning gestreept; de eerste episodes liggen ver onder −100 en blijven aan de onderrand. Rechts de beste action van elke state in run 1 als kleine pijlen, en in <span class="c-param">oranje</span> de weg waar de pijlen vanaf S heen leiden, zodra de pijlen naar G leiden.
    """)
    return


@app.cell(hide_code=True)
def explore_cliff_cell():
    explore_cliff = L.explore("play", "episodes", lambda _v: C.play_frame(int(_v), L.lang), start=C.EPISODES,
                              values=C.LOOKS, role="",
                              task="Ga naar 50: welke pijlen leiden het eerst naar G, en langs welke rij? Ga dan door naar 500: welke lijn blijft hoger, hoewel de weg langer is?")
    explore_cliff
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    De eerste episode kost beide agents honderden stappen: over de 50 runs is de gemiddelde return −1.833 voor SARSA en −1.800 voor Q-learning. Met elke $Q$ op 0 dwaalt de agent rond en valt vaak. Na 50 episodes leiden de pijlen van Q-learning in run 1 langs de rand, 13 stappen; na 100 episodes gaan de pijlen van SARSA omhoog naar de bovenste rij, 17 stappen. Vanaf dan blijft de lijn van SARSA erboven: over de laatste 100 episodes is de gemiddelde return van SARSA −28 en van Q-learning −50. Q-learning valt in één op de drie episodes, en SARSA in één op de twintig.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    SARSA, voor één episode:

    1. Begin in S, het startvakje: dat is de eerste state $S$. Kies een action $A$ met ε-greedy.
    2. Neem $A$: je krijgt een reward $R$ en komt in $S'$.
    3. Kies de volgende action $A'$ in $S'$, met ε-greedy.
    4. Schuif $Q(S, A)$ op naar het target $R + \gamma Q(S', A')$, met $\eta$ keer het verschil.
    5. $S'$ wordt $S$ en $A'$ wordt $A$. Is $S$ gelijk aan G, dan is de episode voorbij; anders ga je verder bij stap 2.

    Q-learning verschilt alleen in stap 4: het target is $R + \gamma \max_{a'} Q(S', a')$, dus Q-learning heeft $A'$ niet nodig om bij te werken. In G kiest SARSA ook nog een $A'$, maar elke $Q$ van G blijft 0: vanuit G begint geen stap meer, dus het laatste target is gewoon $R$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stel dat de agent na een aantal episodes in (2, 5) staat, op de rij naast het klif, en naar rechts gaat, naar (2, 6), met reward −1. $Q$((2, 5), right) is −7, en de vier waarden van (2, 6) zijn: up −8, down −114, left −8 en right −6. $\eta$ = 0,5 en $\gamma$ = 1.

    - Q-learning: het target is $-1 + \max(-8, -114, -8, -6) = -1 - 6 = -7$, dus $Q$((2, 5), right) $= -7 + 0{,}5 \cdot (-7 - (-7)) = -7$: er verandert niets.
    - SARSA, als ε-greedy in (2, 6) de beste action kiest, right: het target is $-1 + (-6) = -7$, hetzelfde, en er verandert niets.
    - SARSA, als de willekeurige action down is: het target is $-1 + (-114) = -115$, en $Q$((2, 5), right) $= -7 + 0{,}5 \cdot (-115 - (-7)) = -7 - 54 = -61$.

    Een willekeurige stap omlaag komt hier één keer per 40 stappen: 0,1 keer 1/4. Elke keer trekt SARSA de $Q$ van langs de rand lopen ver omlaag, en de gewone stappen daarna trekken de waarde weer omhoog, telkens de helft van de afstand. Gemiddeld blijft de $Q$ van SARSA voor de vakjes langs de rand dus lager dan de $Q$ van Q-learning: SARSA telt de vallen door ε-greedy mee, en de pijlen van SARSA draaien weg van de rand.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_update():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Eén stap van SARSA

        `sarsa_update(Q, state, action, reward, nxt, nxt_action, eta, gamma)` geeft de nieuwe $Q$ van `state` en `action` terug: stap 4 van het recept. De agent nam `action` in `state`, kreeg `reward`, kwam in `nxt`, en neemt daar `nxt_action`. `Q` is een dict met een dict per state, zoals in Q-learning: `Q[state][action]`. `eta` en `gamma` zijn $\eta$ en $\gamma$. Vervang de `...` door het `target`.

        De check roept `sarsa_update(Q, (2, 5), "right", -1, (2, 6), "down", 0.5, 1.0)` aan, de stap uit het uitgewerkte voorbeeld waarin de willekeurige action down is, met `Q` de Q-table uit het uitgewerkte voorbeeld. Dan vier andere stappen met `Q_500`, de Q-table van run 1 van SARSA in de course na 500 episodes: op de bovenste rij, een val in het klif, en twee met $\gamma$ = 0,9.
        """),
        L.task("Laat `sarsa_update` Q(s, a) opschuiven naar de reward plus γ keer de Q van de volgende action in de volgende state."),
    ])
    return


@app.cell(hide_code=True)
def exercise_update():
    box_update = L.code_box("sarsa_update", lines=4)
    box_update
    return (box_update,)


@app.cell(hide_code=True)
def check_update(box_update):
    result_update = L.check("sarsa_update", box_update.value)
    result_update
    return (result_update,)


@app.cell(hide_code=True)
def hints_update():
    L.hints(
        "sarsa_update",
        [
            "Stap 4 van het recept en de formule van *Het idee*: het `target` is de `reward` plus `gamma` keer één $Q$ van de volgende state. Welke state is de volgende, en welke action neemt de agent daar?",
            "In het uitgewerkte voorbeeld, met de willekeurige stap omlaag: $-1 + 1 \\cdot Q$((2, 6), down) $= -1 - 114 = -115$. De functie krijgt (2, 6) en down onder twee van de namen tussen de haakjes: welke?",
            "`    target = reward + gamma * ___`",
        ],
        why="`target = reward + gamma * Q[nxt][nxt_action]` is het target van SARSA: de reward plus de verdisconteerde $Q$ van de action die de agent echt als volgende neemt. Met de grootste $Q$ van `nxt` op deze plek zou het Q-learning zijn.",
    )
    return


@app.cell(hide_code=True)
def text_after_update():
    mo.md(r"""
    In de afbeelding leert je `sarsa_update` run 1 uit het speelblok. Na 500 episodes gaan de pijlen vanaf S omhoog, over de bovenste rij en omlaag naar G: 17 stappen. Run 1 is een van de runs met minder geluk: in de laatste 100 episodes viel de agent 8 keer, en de gemiddelde return is −39,3.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_episode():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een hele episode

        `sarsa_episode(Q, eta, epsilon, gamma, rng)` speelt één episode van SARSA op het klif, werkt `Q` na elke stap bij en geeft `total` terug, de return. De functie gebruikt deze onderdelen van de course: `step(state, action)`, het klif, dat `nxt` en `reward` teruggeeft; `choose(Q, state, epsilon, rng)`, ε-greedy zoals je het in Q-learning schreef, met `rng` als bron van willekeurige getallen; `sarsa_update`, de eigen oplossing van de course voor oefening 1; en `START` en `GOAL`, (3, 0) en (3, 11). De loop `while state != GOAL:` herhaalt het blok eronder zolang de agent niet op G staat. Vervang de `...`: `nxt_action` moet de action worden waarmee de agent verdergaat in de state waar de agent terechtkomt, stap 3 van het recept. Twee regels verder laat `action = nxt_action` de agent deze action echt nemen.

        De check roept `sarsa_episode(Q_50, 0.5, 0.1, 1.0, np.random.default_rng(1))` aan, met `Q_50` de Q-table van run 1 in de course na 50 episodes en een nieuwe generator van willekeurige getallen met seed 1, en vergelijkt de return en de Q-table die je functie achterlaat met de uitkomst van de course. Dan vier andere gevallen: seed 2 met ε = 0,2, een table met nullen, `Q_500` met $\eta$ = 0,1 en $\gamma$ = 0,9, en `Q_50` met $\eta$ = 0,3, ε = 0,05 en $\gamma$ = 0,9.
        """),
        L.task("Laat `sarsa_episode` de volgende action kiezen met ε-greedy, in de state waar de agent net is beland, voordat Q wordt bijgewerkt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_episode():
    box_episode = L.code_box("sarsa_episode", lines=13)
    box_episode
    return (box_episode,)


@app.cell(hide_code=True)
def check_episode(box_episode):
    result_episode = L.check("sarsa_episode", box_episode.value)
    result_episode
    return (result_episode,)


@app.cell(hide_code=True)
def hints_episode():
    L.hints(
        "sarsa_episode",
        [
            "Stap 3 van het recept: `nxt_action` is de ε-greedy keuze van de agent in de state waar de agent net is beland. `choose` krijgt de Q-table `Q`, een state, ε en de willekeurige getallen, zoals op de regel boven de loop.",
            "Boven de loop koos de agent de eerste action in de state waar de agent begon. Waar staat de agent na `step`, en welke naam op de regel met `step` bevat deze state?",
            "`        nxt_action = choose(Q, ___)`",
        ],
        why="`nxt_action = choose(Q, nxt, epsilon, rng)` kiest de volgende action in `nxt` voordat Q wordt bijgewerkt, want het target van SARSA heeft deze action nodig; twee regels verder laat `action = nxt_action` de agent deze action echt nemen.",
    )
    return


@app.cell(hide_code=True)
def text_after_episode():
    mo.md(r"""
    In de afbeelding speelt je `sarsa_episode` 200 episodes achter elkaar, vanaf een table met nullen: de eerste dwaalt honderden stappen rond, en aan het einde nemen de pijlen de bovenste rij, 17 stappen. De laatste 50 episodes geven een gemiddelde return van −21,9.

    ### Alle runs tegelijk met NumPy

    Eén run zegt weinig. `C.course_returns()` geeft de return van elke episode van elke run, voor SARSA en voor Q-learning, als twee arrays met shape `(50, 500)`: een rij per run, een kolom per episode. `C.course_falls()` geeft het aantal vallen van elke episode in dezelfde shape. `_sarsa[:, 400:]` neemt de laatste 100 episodes van elke run, en `np.mean` zonder axis neemt het gemiddelde van al deze getallen. `np.mean(..., axis=1)` neemt het gemiddelde van elke rij: één getal per run. `np.max` en `np.min` geven het grootste en het kleinste getal van een array. Voer de cel uit.
    """)
    return


@app.cell
def example_numpy():
    import numpy as np

    _sarsa, _q = C.course_returns()               # shape (50, 500): een rij per run, een kolom per episode
    _falls_sarsa, _falls_q = C.course_falls()
    _last_sarsa = np.mean(_sarsa[:, 400:], axis=1)   # de gemiddelde return van de laatste 100 episodes, één getal per run
    _last_q = np.mean(_q[:, 400:], axis=1)
    print("shape:", _sarsa.shape)
    print("gemiddelde return van de laatste 100 episodes, SARSA:", round(float(np.mean(_sarsa[:, 400:])), 1), "Q-learning:", round(float(np.mean(_q[:, 400:])), 1))
    print("vallen per episode, laatste 100 episodes, SARSA:", round(float(np.mean(_falls_sarsa[:, 400:])), 3), "Q-learning:", round(float(np.mean(_falls_q[:, 400:])), 3))
    print("runs waarin SARSA de hoogste gemiddelde return heeft:", int((_last_sarsa > _last_q).sum()))
    print("beste en slechtste run, SARSA:", round(float(np.max(_last_sarsa)), 1), round(float(np.min(_last_sarsa)), 1))
    print("beste en slechtste run, Q-learning:", round(float(np.max(_last_q)), 1), round(float(np.min(_last_q)), 1))
    return (np,)


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    Over de laatste 100 episodes valt SARSA 0,051 keer per episode en Q-learning 0,332 keer, zes à zeven keer zo vaak. SARSA heeft de hoogste gemiddelde return in 49 van de 50 runs. De runs lopen ver uiteen: de beste run van SARSA haalt −20,8 en de slechtste −63,4, de beste van Q-learning −34,8 en de slechtste −61,0. Met $\eta$ = 0,5 blijven de waarden bewegen, en een paar vallen met pech na elkaar kunnen de pijlen van SARSA een tijdje dichter naar de rand sturen.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Je `sarsa_update` naast Q-learning

    De demo speelt de 50 runs uit het speelblok met je `sarsa_update` voor SARSA, en de Q-learning van de course ernaast. Waar je functie de check nog niet haalt, neemt de functie van de course het over. Kies ε voor beide methoden: met 0,1 zie je het speelblok terug; probeer daarna 0,2, 0,05 en 0,01. De grafiek toont de gemiddelde return van de laatste 10 episodes over de 50 runs, SARSA doorgetrokken en Q-learning gestreept; ernaast de pijlen van run 1 op dit moment.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    epsilon = mo.ui.dropdown(options=C.eps_options(L.lang), value=C.eps_text(0.1, L.lang), label=C.DEMO_TEXT[L.lang]["eps"])
    play_button = mo.ui.run_button(label="Speel 50 runs met mijn sarsa_update")
    mo.hstack([epsilon, play_button], justify="start", gap=1.5, wrap=True)
    return epsilon, play_button


@app.cell(hide_code=True)
def demo_play(box_update, epsilon, play_button, result_update):
    learned = None
    if not play_button.value:
        _out = mo.md("*Kies ε, en druk dan op de knop om beide methoden het klif te laten leren.*")
    else:
        _mine = result_update.state == "correct"
        _fn = run_code(box_update.value).names["sarsa_update"] if _mine else C.sarsa_update
        _w = C.DEMO_TEXT[L.lang]
        _now = {"shown": 0}

        def _progress(_n, _run):
            if "run" not in _now:
                _now["chart"].watch(lambda: C.now_picture(_now["run"], _now["n"], L.lang))
            _now["run"], _now["n"] = _run, _n
            if _n % 5 == 0 or _n == 1:
                _now["chart"].reference(C.chart_points(_run, "q", 1, _n), "", role="ink", replace=True)
                _now["chart"].extend(C.chart_points(_run, "sarsa", _now["shown"] + 1, _n), f"**Episode {_n} van {C.EPISODES}**")
                _now["shown"] = _n

        _live = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"], "alt": _w["alt"]}, C.EPISODES, role="ink",
                       low=C.LOW, high=0.0, window=1, key=C.curve_key(L.lang))
        _live.line("**Leren** …")
        _now["chart"] = _live
        _r = C.run(_fn, epsilon.value, show=_progress)
        if _r == "sarsa":
            _out = _live.stopped("*Je `sarsa_update` gaf een fout tijdens de run, of gaf iets wat geen getal is. Haalt je functie de check nog?*")
        elif _r == "slow":
            _out = _live.stopped("*De run duurde langer dan een minuut. Doet je functie meer werk dan nodig?*")
        else:
            learned = _r
            _who = "je `sarsa_update`" if _mine else "de `sarsa_update` van de course, omdat je eigen functie de check nog niet haalt"
            _out = mo.vstack([
                L.explore("rewind", "episodes", lambda _v: C.rewind_frame(_r, int(_v), L.lang), start=C.EPISODES,
                          values=C.LOOKS, role="",
                          task="Ga terug naar 20, en dan weer vooruit: wanneer leiden de pijlen van elke methode voor het eerst naar G, en wanneer komt de doorgetrokken lijn van SARSA boven de gestreepte lijn uit?"),
                mo.md(f"Gespeeld met {_who} en ε = {C.eps_text(epsilon.value, 'nl')}. Over de laatste 100 episodes is de gemiddelde return van SARSA {C.number(C.last_mean(_r, 'sarsa', C.EPISODES, 100), 1, 'nl')} en van Q-learning {C.number(C.last_mean(_r, 'q', C.EPISODES, 100), 1, 'nl')}."),
            ])
    _out
    return (learned,)


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met de functies van de course, over de laatste 100 episodes: met ε = 0,2 heeft SARSA een gemiddelde return van −42,7 en Q-learning −106,4; met ε = 0,05 is het −22,5 tegen −30,2; met ε = 0,01 −17,0 tegen −16,3, en nu ligt Q-learning voor. Hoe meer de agent verkent, hoe meer de weg van Q-learning langs de rand kost tijdens het leren. Met ε = 0,01 komt een willekeurige stap omlaag één keer per 400 stappen, en wint de kortere weg zelfs tijdens het leren. Op dit kleine rooster verkent zelfs ε = 0,01 genoeg; bij grotere taken kan te weinig verkennen een agent ervan weerhouden ooit een betere weg te vinden.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_robot():
    mo.md(r"""
    ### Oefening 3. Leren in een simulator

    Een team traint een bezorgrobot in een simulator van een gebouw met een lang balkon zonder hek. Een val kost daar −100, en er gaat niets kapot. De robot leert met ε-greedy, ε = 0,1. Daarna werkt de robot in het echte gebouw en neemt altijd de beste action: geen willekeurige stappen meer.
    """)
    return


@app.cell(hide_code=True)
def choice_robot_cell():
    choice_robot = L.choice(
        "robot",
        question="Met welke methode moet het team trainen, voor de weg die de robot in het echte gebouw gaat lopen?",
        options={
            "sarsa": "SARSA: een weg die wegblijft van de rand is altijd de betere, ook zonder willekeurige stappen",
            "right": "Q-learning: de methode leert wat de beste actions waard zijn, en de robot neemt alleen deze",
            "same": "Maakt niet uit: na genoeg episodes leren beide dezelfde weg",
            "neither": "Geen van beide: een robot die met willekeurige stappen leerde kan niet zonder",
        },
        correct="right",
        why={
            "right": "Q-learning is off-policy: de waarden horen bij altijd de beste action nemen, wat de robot tijdens het leren ook deed. Zonder willekeurige stappen valt de korte weg langs de rand nooit: op het klif van deze les 13 stappen, tegen 17 voor de weg van SARSA. SARSA is on-policy: SARSA leerde de waarden van de robot met de willekeurige stappen, en loopt om een gevaar heen dat er niet meer is.",
            "sarsa": "Valt de robot zonder willekeurige stappen ooit van de rand? Welke methode leerde de waarden van precies deze robot?",
            "same": "Op het klif nemen na 500 episodes de pijlen van Q-learning de rand en de pijlen van SARSA de bovenste rij: waarom blijven de twee wegen verschillend zolang ε = 0,1?",
            "neither": "De pijlen zeggen welke action in elke state de beste is; heeft de robot willekeurige stappen nodig om de pijlen te volgen?",
        },
    )
    choice_robot
    return (choice_robot,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Leren na elke stap: TD(0)](/course/open/rl-td-prediction?lang=nl) leerde je $V$ van een vaste policy na elke stap. SARSA doet hetzelfde voor $Q(s, a)$ van de policy die de agent volgt, terwijl ε-greedy deze policy blijft verbeteren als de waarden veranderen: TD learning die kiest. [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl) is de off-policy tegenhanger. De volgende les, *Een kaart van RL-methoden*, zet de methoden die je kent naast elkaar: met of zonder model van de wereld, waarden leren of de policy zelf, on-policy of off-policy, van hele episodes of van de eigen schattingen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_robot, result_episode, result_update):
    L.wrap_up(result_update, result_episode, choice_robot)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
