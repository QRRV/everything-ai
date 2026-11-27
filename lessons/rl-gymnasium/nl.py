# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-gymnasium. Route: course/route/lessen/reinforcement-learning.yaml. Checks, de runs van de course in drie werelden van Gymnasium, en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-gymnasium", "nl")
    C = L.checks   # de runs van de course in FrozenLake, Taxi en CliffWalking, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl) en [Voorzichtig of gedurfd: SARSA en Q-learning](/course/open/rl-sarsa?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_target():
    quick_target = L.choice(
        "quick-target",
        question="Een Q-learning-agent zet een stap met reward −1 en komt in een state waarvan de grootste $Q$ 4 is. Wat is met $\\gamma$ = 0,5 het target van deze stap?",
        options={
            "reward": "−1, alleen de reward van de stap",
            "right": "1, de reward plus $\\gamma$ keer deze grootste $Q$",
            "plain": "3, de reward plus de grootste $Q$, zonder $\\gamma$",
            "all": "1,5, $\\gamma$ keer de reward en de grootste $Q$ bij elkaar opgeteld",
        },
        correct="right",
        why={
            "right": "$-1 + 0{,}5 \\cdot 4 = 1$: de reward plus de verdisconteerde grootste $Q$ waar de agent terechtkwam. In deze les krijgt het target er een vraag bij: eindigde de episode daar echt?",
            "reward": "Alleen de reward is het target als er na de stap niets meer volgt: wat telt Q-learning erbij op voor de state waar de agent terechtkwam? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
            "plain": "Latere rewards tellen minder zwaar: wat doet $\\gamma$ met de grootste $Q$ van de volgende state? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
            "all": "$\\gamma$ verdisconteert alleen wat na de stap komt, niet de reward van de stap zelf: met welk getal vermenigvuldig je $\\gamma$? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
        },
    )
    quick_target
    return (quick_target,)


@app.cell(hide_code=True)
def quick_check_cliff():
    quick_cliff = L.choice(
        "quick-cliff",
        question="Op het klif van 4 bij 12 leert Q-learning met ε-greedy, ε = 0,1. Welke weg wijst de grootste $Q$ na 500 episodes van S naar G?",
        options={
            "top": "Omhoog naar de bovenste rij, daarlangs en weer omlaag: 17 stappen, ver van de rand",
            "right": "Langs de rand van het klif: 13 stappen",
            "middle": "Langs de middelste rij: 15 stappen, een rij van de rand",
            "none": "Nergens heen: de agent valt steeds",
        },
        correct="right",
        why={
            "right": "Q-learning leert de waarden van altijd de beste action nemen, dus loopt de weg langs de rand, 13 stappen; SARSA, on-policy, leert de bovenste rij. In deze les komt hetzelfde klif terug als wereld van Gymnasium.",
            "top": "Dat is de weg van SARSA, dat leert met de eigen willekeurige stappen erbij: van welke policy leert Q-learning de waarden? Zie [Voorzichtig of gedurfd: SARSA en Q-learning](/course/open/rl-sarsa?lang=nl).",
            "middle": "Het target van Q-learning neemt de grootste $Q$, nooit een willekeurige stap omlaag: hoe dicht bij de rand kan de weg dan lopen? Zie [Voorzichtig of gedurfd: SARSA en Q-learning](/course/open/rl-sarsa?lang=nl).",
            "none": "Door de willekeurige stappen valt de agent af en toe tijdens het leren, maar de grootste $Q$ wijst wel naar G: langs welke rij? Zie [Voorzichtig of gedurfd: SARSA en Q-learning](/course/open/rl-sarsa?lang=nl).",
        },
    )
    quick_cliff
    return (quick_cliff,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In elke les over RL tot nu toe kwam de wereld met eigen code. GRID van Q-learning had `step(state, action, rng)`, het klif van SARSA had `step(state, action)`, en elke wereld had eigen states, eigen namen voor actions en een eigen manier om te zeggen dat een episode voorbij was. Een agent voor de ene wereld werkte niet in een andere. In 2016 bracht OpenAI Gym uit: een library met veel environments, van een bevroren meer tot Atari-spellen, achter één interface. Sinds 2022 onderhoudt de Farama Foundation de library als Gymnasium. Onderzoekers testen een nieuwe methode op dezelfde environments, in dezelfde versies, zodat de uitkomsten te vergelijken zijn, en libraries voor RL, zoals Stable-Baselines3 in een latere les, trainen op elke environment die de interface volgt. In deze les schrijf je één Q-learning-agent en laat je deze agent drie werelden leren zonder een regel te veranderen.

    **Een voorbeeld uit het dagelijks leven.** Elke auto heeft het gaspedaal rechts, de rem daar links van en het stuur voor je. Kun je in één auto rijden, dan kun je in bijna elke auto rijden, welke motor er ook onder de motorkap zit. Gymnasium geeft elke environment dezelfde pedalen.
    """)
    return


@app.cell(hide_code=True)
def predict_fail():
    guess_fail = L.guess(
        "fail",
        question="De course traint dezelfde Q-learning-agent in de drie werelden van de afbeelding: FrozenLake 10.000 episodes, Taxi 5.000 en CliffWalking 500, met η = 0,1, γ = 0,99 en ε = 0,1, en elke Q op 0 aan het begin. Elke wereld krijgt 10 runs, elk met eigen willekeurige getallen. In welke wereld leren sommige runs de taak niet, denk je?",
        options={
            "taxi": "Taxi: 500 states en 6 actions zijn te veel om in 5.000 episodes te leren",
            "cliff": "CliffWalking: in 500 episodes valt de agent steeds van het klif",
            "right": "FrozenLake, de kleinste",
            "none": "Geen: in alle drie de werelden leren alle 10 runs de taak",
        },
        reveal="**FrozenLake, de kleinste.** In 3 van de 10 runs wijst de grootste Q nooit naar G, en bereikt de agent G in 0% van de episodes; de andere 7 bereiken G in 73,0% tot en met 74,0% van de episodes, dicht bij 74,4%, het beste wat een policy binnen 100 stappen kan. In Taxi brengen alle 10 runs elke passagier weg, en op CliffWalking nemen alle 10 de kortste weg, 13 stappen. Na de demo zie je waarom de kleinste wereld de moeilijkste is.",
    )
    mo.vstack([mo.Html(C.world_picture(L.lang)), guess_fail])
    return (guess_fail,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Eén interface.** Gymnasium is een library voor Python, geïmporteerd als `gym`. `env = gym.make("FrozenLake-v1")` maakt een environment bij de naam; `-v1` is de versie, en de versie gaat omhoog zodra de wereld verandert, zodat een uitkomst altijd zegt uit welke wereld de uitkomst kwam. Elke environment heeft dezelfde twee methoden. `obs, info = env.reset(seed=0)` begint een episode en geeft de eerste observation, `obs`: wat de agent van de state te zien krijgt. `obs, reward, terminated, truncated, info = env.step(action)` neemt één action en geeft vijf dingen terug: de observation na de stap, de reward, twee vlaggen die zeggen of de episode voorbij is, en `info`, een dict met extra informatie die de agent niet hoeft te leren. `seed=0` legt de willekeurige getallen van de wereld vast, zodat dezelfde seed dezelfde episodes geeft; `env.reset()` zonder seed begint de volgende episode, en de reeks willekeurige getallen loopt door.

        **Spaces.** `env.observation_space` en `env.action_space` zeggen hoe observations en actions eruitzien: de **observation space** en de **action space**. In de drie werelden van deze les zijn beide `Discrete(n)`: de gehele getallen 0 tot en met n − 1. `env.action_space.n` is deze n, en `env.action_space.sample()` geeft een willekeurige action. Andere werelden hebben andere spaces: in CartPole uit de volgende les is een observation vier reële getallen, een `Box`. Een agent met een Q-table heeft een `Discrete` observation space nodig, voor een rij per observation, en een `Discrete` action space, voor een kolom per action:

        | Wereld | `observation_space` | `action_space` | de actions | tijdslimiet |
        |---|---|---|---|---|
        | FrozenLake-v1 | `Discrete(16)` | `Discrete(4)` | 0 links, 1 omlaag, 2 rechts, 3 omhoog | 100 stappen |
        | Taxi-v4 | `Discrete(500)` | `Discrete(6)` | 0 zuid, 1 noord, 2 oost, 3 west, 4 ophalen, 5 afzetten | 200 stappen |
        | CliffWalking-v1 | `Discrete(48)` | `Discrete(4)` | 0 omhoog, 1 rechts, 2 omlaag, 3 links | geen |

        Op FrozenLake en CliffWalking is de observation het vakje, rij voor rij geteld vanaf linksboven: op FrozenLake rij × 4 + kolom. In Taxi zitten er vier dingen tegelijk in: de rij en de kolom van de taxi, waar de passagier is, bij een van de vier haltes of in de taxi, en de bestemming: 5 × 5 × 5 × 4 = 500 observations.

        **Geëindigd of afgekapt.** Een episode kan om twee redenen stoppen. `terminated` is `True` als de taak zelf eindigt: de agent staat op G of in een wak, of de passagier is weggebracht. Daarna komt er niets meer. `truncated` is `True` als een tijdslimiet de episode afkapt: na 100 stappen op FrozenLake, na 200 in Taxi. De wereld is daar niet afgelopen: de taxi staat nog ergens in de straten, en de passagier wacht nog steeds. Een episode zo vroeg stoppen heet **truncation**: afkappen. Voor het leren zit het verschil in het target. Na `terminated` is het target van de laatste stap alleen de reward. Na `truncated` heeft de state nog een toekomst, dus blijft het target de reward plus γ keer de grootste Q van de volgende state, net als na elke andere stap. Oudere versies van Gym hadden één vlag, `done`, voor beide redenen; Gymnasium splitst deze vlag in tweeën.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **Gymnasium** | | een library voor Python met veel environments achter één interface: `reset`, `step` en spaces | `import gymnasium as gym` |
        | **observation space** | | wat een observation kan zijn, zoals `Discrete(16)`: een geheel getal van 0 tot en met 15 | `env.observation_space` |
        | **action space** | | wat een action kan zijn, zoals `Discrete(4)` | `env.action_space` |
        | **truncation** | | een episode vroeg stoppen, door een tijdslimiet, terwijl de taak niet geëindigd is: afkappen | `truncated` |
        | `terminated` | | de taak is geëindigd, en er volgt niets meer | |
        | `Discrete(n)` | discreet n | de gehele getallen 0 tot en met n − 1 | `.n` |
        """),
        L.watch_out("Hetzelfde getal is in een andere wereld een andere action: 0 is links op FrozenLake en omhoog op CliffWalking. Jouw agent gebruikt straks alleen wat de spaces zeggen, en werkt daardoor in elke wereld; een agent met \"0 is links\" in de code werkt in één wereld."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Q-learning in elke wereld van Gymnasium met `Discrete` spaces:

    1. Maak de wereld met `gym.make`, en een Q-table met nullen met `env.observation_space.n` rijen en `env.action_space.n` kolommen.
    2. Geef de wereld één keer een seed, met `env.reset(seed=...)`, zodat je een run kunt herhalen.
    3. Begin een episode met `env.reset()`: dat geeft de eerste observation.
    4. Kies een action, ε-greedy: met kans ε een willekeurige van de `env.action_space.n` actions, anders de action met de grootste $Q$ in de rij van de observation.
    5. Neem de action met `env.step(action)`: je krijgt de volgende observation, de reward, `terminated`, `truncated` en `info`.
    6. Reken het target uit: alleen de reward als `terminated`; anders, ook als `truncated`, de reward plus γ keer de grootste $Q$ van de volgende observation. Schuif $Q$ een stap η op naar het target.
    7. Ga vanaf de volgende observation verder bij stap 4, tot `terminated` of `truncated` `True` is. Begin dan de volgende episode bij stap 3.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    FrozenLake, met `env.reset(seed=6)`: de agent begint in 0, linksboven, en neemt de actions rechts, rechts, omlaag en omlaag. De cel hieronder maakt de wereld, print de spaces en zet de vier stappen; `info` bevat `prob`, de kans op wat er gebeurde. `break` stopt de loop zodra de episode voorbij is. Voer de cel uit.
    """)
    return


@app.cell
def example_frozen():
    import gymnasium as gym

    _env = gym.make("FrozenLake-v1")
    print("observation_space:", _env.observation_space, "action_space:", _env.action_space)
    _obs, _info = _env.reset(seed=6)
    print("start:", _obs)
    for _action in [2, 2, 1, 1]:   # rechts, rechts, omlaag, omlaag
        _nxt, _reward, _terminated, _truncated, _info = _env.step(_action)
        print("van", _obs, "action", _action, "naar", _nxt, "reward", _reward,
              "terminated", _terminated, "truncated", _truncated, "prob", round(_info["prob"], 3))
        _obs = _nxt
        if _terminated or _truncated:
            break
    return (gym,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste drie stappen gaan waar de agent om vroeg: 0 naar 1, 1 naar 2, en omlaag van 2 naar 6, elk met kans 1/3, `prob` 0,333. De vierde, omlaag vanaf 6, glijdt naar rechts uit, het wak bij 7 in: `terminated` is `True`, de reward is 0, en de episode is voorbij. Na een wak komt er niets meer, dus is het target van deze stap alleen de reward, 0.

    Nu Taxi, in run 4 van de course na 5.000 episodes. In state 249 staat de taxi midden in de straten, wacht de passagier bij Y, en is G de bestemming. De zes Q's van 249 zijn −4,85, −6,56, −5,21, 5,30, −7,82 en −6,47; de grootste, 5,30, hoort bij west. Stel dat een stap met reward −1 in 249 uitkomt, en het is de 200e stap van de episode: `truncated` is `True` en `terminated` is `False`. Het target is −1 + 0,99 × 5,30 = 4,25, hetzelfde als voor elke andere stap naar 249: vanaf daar brengt de grootste Q de passagier in 14 stappen weg. Een target van alleen −1 zou de agent vertellen dat het midden van de straten nergens heen leidt, alleen omdat de klok daar afliep.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_play():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een willekeurige agent in elke wereld

        `play_random(env, seed)` speelt één episode met willekeurige actions in een wereld van Gymnasium, en geeft `total` terug, de return, `steps`, het aantal stappen, en `terminated`, of de taak eindigde. `env.reset(seed=seed)` begint de episode, en `env.action_space.seed(seed)` legt de willekeurige getallen van `env.action_space.sample()` vast, zodat dezelfde seed dezelfde episode geeft. `total, steps = 0, 0` zet beide op 0, en `terminated, truncated = False, False` zet beide vlaggen op `False`. `while not (...):` herhaalt het blok zolang de test tussen de haakjes niet waar is. `total += reward` telt `reward` bij `total` op, en `steps += 1` telt 1 bij `steps` op. Vervang de twee `...`: de test, zodat de loop stopt zodra de episode geëindigd of afgekapt is, en de aanroep die de `action` neemt en de vijf waarden aan `obs`, `reward`, `terminated`, `truncated` en `info` geeft.

        De check roept `play_random(gym.make("FrozenLake-v1"), 0)` aan. Daarna vier andere episodes: Taxi met seeds 0 en 27, CliffWalking met seed 0, en FrozenLake met seed 115.
        """),
        L.task("Laat `play_random` stoppen als de episode eindigt of wordt afgekapt, en zet elke stap met `env.step`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_play():
    box_play = L.code_box("play_random", lines=11)
    box_play
    return (box_play,)


@app.cell(hide_code=True)
def check_play(box_play):
    result_play = L.check("play_random", box_play.value)
    result_play
    return (result_play,)


@app.cell(hide_code=True)
def hints_play():
    L.hints(
        "play_random",
        [
            "Stap 5 en 7 van het recept. De loop gaat door zolang `terminated` en `truncated` allebei niet `True` zijn. De andere `...` is de aanroep die de `action` neemt en `obs, reward, terminated, truncated, info` teruggeeft, in deze volgorde.",
            "`while not (a or b):` stopt zodra `a` of `b` `True` is. In het uitgewerkte voorbeeld gaf de stap vanaf 6 met action 1 de waarden 7, 0, `True`, `False` en `info` terug: welke methode van `env` zette deze stap?",
            "`        obs, reward, terminated, truncated, info = ___`",
        ],
        why="`while not (terminated or truncated):` stopt bij een einde en bij afkappen allebei, en `obs, reward, terminated, truncated, info = env.step(action)` zet één stap. Dezelfde regels werken in elke wereld van Gymnasium.",
    )
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    In de afbeelding speelt je `play_random` de seeds 0, 1 en 2 in elke wereld. FrozenLake eindigt binnen een paar stappen, in een wak; een willekeurige agent bereikt daar G in 1 op de 72 episodes. In Taxi kapt de klok elk van deze episodes af na 200 stappen, met een return ver onder 0; bij toeval brengt een willekeurige agent de passagier weg in ongeveer 1 op de 21 episodes. CliffWalking heeft geen tijdslimiet, dus gaat een willekeurige wandeling door tot G: 3.791 stappen met seed 0, en gemiddeld 6.453.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_target():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Het target, geëindigd of afgekapt

        `target(Q, reward, nxt, terminated, truncated, gamma)` geeft het target van Q-learning voor één stap terug: stap 6 van het recept. `Q` is een Q-table als NumPy-array, met een rij per observation en een kolom per action. `Q[nxt]` is de rij van `nxt`, de observation na de stap, en de methode `.max()` van een array geeft het grootste getal. `reward`, `terminated` en `truncated` komen uit `env.step`, en `gamma` is γ. Vervang de twee `...`: de test voor een stap waarna niets meer volgt, en de grootste Q van `nxt`.

        De check roept `target(Q_taxi, -1, 249, False, False, 0.99)` aan, met `Q_taxi` de Q-table van run 4 van de course in Taxi: een stap zoals elke andere, naar 249, het midden uit het uitgewerkte voorbeeld. Daarna vier andere stappen: dezelfde stap als 200e van de episode, met `truncated` `True`; twee stappen in FrozenLake vanuit `Q_ones`, een tabel waarin elke Q 1 is: het wak bij 7 in, en op G bij de 100e stap, waar beide vlaggen `True` zijn; en een stap met γ = 0,9 naar 257, het midden met de passagier in de taxi.
        """),
        L.task("Laat `target` alleen de reward geven als de taak eindigde, en anders de reward plus γ keer de grootste Q van de volgende observation."),
    ])
    return


@app.cell(hide_code=True)
def exercise_target():
    box_target = L.code_box("target", lines=4)
    box_target
    return (box_target,)


@app.cell(hide_code=True)
def check_target(box_target):
    result_target = L.check("target", box_target.value)
    result_target
    return (result_target,)


@app.cell(hide_code=True)
def hints_target():
    L.hints(
        "target",
        [
            "Stap 6 van het recept: na een einde is het target alleen de `reward`; na elke andere stap, ook na afkappen, is het de `reward` plus `gamma` keer de grootste Q van de volgende observation. Welke van de twee vlaggen zegt dat de taak zelf eindigde?",
            "In het uitgewerkte voorbeeld is het grootste getal van rij 249 5,30, en het target van de 200e stap −1 + 0,99 × 5,30 = 4,25, terwijl `truncated` `True` is. Welke rij van `Q` hoort bij `nxt`, en welke methode geeft het grootste getal?",
            "`    return reward + gamma * ___`",
        ],
        why="`if terminated:` geeft alleen de reward als de taak eindigde; na afkappen houdt `return reward + gamma * Q[nxt].max()` de toekomst van `nxt`, omdat de wereld daar doorgaat.",
    )
    return


@app.cell(hide_code=True)
def text_after_target():
    mo.md(r"""
    In de afbeelding staat je `target` voor vier van de stappen. De 200e stap van Taxi houdt de toekomst, 4,25, net als elke andere stap naar het midden. Met `Q_ones` heeft het wak bij 7 een Q van 1, dus zou een target dat niet naar `terminated` vraagt 0,99 geven voor erin vallen, alsof er na het wak iets goeds komt. In een tabel die bij 0 begint, valt deze fout niet op: de Q's van een einde blijven 0, omdat daar geen stap begint.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_episode():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Eén agent voor elke wereld

        `q_episode(env, Q, rng)` speelt één episode Q-learning in een wereld van Gymnasium, werkt `Q` na elke stap bij en geeft `total` terug, de return. De functie weet niets van de wereld behalve de spaces. `Q[obs, action]` is de Q van `obs` en `action`, en `np.argmax(Q[obs])` de action met de grootste Q van `obs`. `rng` geeft de willekeurige getallen, zoals in Q-learning: `rng.random()` een getal van 0 tot 1, en `rng.integers(n)` een geheel getal van 0 tot en met n − 1, hetzelfde als `rng.integers(0, n)`. `EPSILON`, `ETA` en `GAMMA` zijn ε, η en γ: 0,1, 0,1 en 0,99. `target` is de eigen oplossing van de course voor oefening 2. `env.reset()` begint de episode, en `env.step(action)` geeft `nxt`, de observation na de stap, `reward`, `terminated`, `truncated` en `info`, zoals in oefening 1; `obs = nxt` laat de agent verdergaan. Vervang de twee `...`: uit hoeveel actions de willekeurige action gekozen wordt, in welke wereld ook, en het target van de stap.

        De check roept `q_episode(gym.make("FrozenLake-v1"), Q_frozen_2000, np.random.default_rng(1))` aan, met `Q_frozen_2000` de Q-table van run 4 van de course in FrozenLake na 2.000 episodes, de wereld met een seed via `env.reset(seed=1)`, en een nieuwe random generator met seed 1. De check vergelijkt de return en de Q-table die je functie achterlaat met de uitkomst van de course. Daarna vier andere gevallen: Taxi vanuit `Q_zeros`, een tabel met nullen, en vanuit `Q_taxi_250`, na 250 episodes; CliffWalking vanuit `Q_zeros` en vanuit `Q_cliff_50`.
        """),
        L.task("Laat `q_episode` een willekeurige action kiezen uit alle actions van de wereld, en Q opschuiven naar het target van oefening 2."),
    ])
    return


@app.cell(hide_code=True)
def exercise_episode():
    box_episode = L.code_box("q_episode", lines=14)
    box_episode
    return (box_episode,)


@app.cell(hide_code=True)
def check_episode(box_episode):
    result_episode = L.check("q_episode", box_episode.value)
    result_episode
    return (result_episode,)


@app.cell(hide_code=True)
def hints_episode():
    L.hints(
        "q_episode",
        [
            "Stap 4 en 6 van het recept. Een willekeurige `action` komt uit `rng.integers`, met evenveel keuzes als de wereld actions heeft: de action space weet hoeveel. Het target is wat `target` voor deze stap geeft.",
            "`Discrete(6)` van Taxi heeft 6 als `.n`, en `env.action_space` is de action space van de wereld. De tweede `...` roept `target` aan met de waarden van de stap in de volgorde van de `def`-regel: `Q`, `reward`, `nxt`, `terminated`, `truncated`, en `GAMMA` als γ.",
            "`            action = rng.integers(___)`",
        ],
        why="`rng.integers(env.action_space.n)` kiest uit alle actions van een wereld, 6 in Taxi en 4 in de andere twee, en `target(Q, reward, nxt, terminated, truncated, GAMMA)` is het target van oefening 2. De functie heeft verder niets van de wereld nodig: dat levert één interface op.",
    )
    return


@app.cell(hide_code=True)
def text_after_episode():
    mo.md(r"""
    In de afbeelding leert je `q_episode` CliffWalking in 500 episodes en Taxi in 2.000, elk vanuit een tabel met nullen. Op het klif wijst de grootste Q langs de rand, zoals Q-learning deed in *Voorzichtig of gedurfd*; in Taxi brengt de grootste Q de passagier van Y naar G. Dezelfde functie leert zonder verandering ook FrozenLake: deze wereld komt in de demo.

    ### De Q-tables met NumPy

    Een Q-table is een array, dus kan NumPy er in één keer naar kijken. `C.course_q(name)` geeft de Q-table van run 4 van de course aan het eind, voor de wereld `name`. `.shape` geeft de rijen en kolommen. `np.abs` maakt elk getal positief, `.sum(axis=1)` telt elke rij op, en `> 0` zegt per rij of er een Q veranderde; `.sum()` telt dan de rijen waarvoor dat geldt. `np.argmax(..., axis=1)` geeft de beste action van elke rij, `np.max(..., axis=1)` de grootste Q, `np.round(..., 2)` rondt af op 2 decimalen, en `.reshape(4, 4)` zet de 16 getallen van FrozenLake in de 4 rijen van het meer. Voer de cel uit.
    """)
    return


@app.cell
def example_numpy():
    import numpy as np

    _frozen = C.course_q("FrozenLake-v1")    # run 4 van de course, na 10.000 episodes
    _taxi = C.course_q("Taxi-v4")            # na 5.000 episodes
    _cliff = C.course_q("CliffWalking-v1")   # na 500 episodes
    print("shapes:", _frozen.shape, _taxi.shape, _cliff.shape)
    _frozen_rows = (np.abs(_frozen).sum(axis=1) > 0).sum()   # de rijen met een Q die niet 0 is
    _taxi_rows = (np.abs(_taxi).sum(axis=1) > 0).sum()
    _cliff_rows = (np.abs(_cliff).sum(axis=1) > 0).sum()
    print("rijen die veranderden:", _frozen_rows, _taxi_rows, _cliff_rows)
    print("FrozenLake, de beste action van elk vakje:")
    print(np.argmax(_frozen, axis=1).reshape(4, 4))
    print("FrozenLake, de grootste Q van elk vakje:")
    print(np.round(np.max(_frozen, axis=1).reshape(4, 4), 2))
    return (np,)


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    Op FrozenLake veranderden 11 van de 16 rijen: de 4 wakken en G blijven 0, omdat daar geen stap begint. In Taxi veranderden 400 van de 500 rijen: in de andere 100 is de passagier al op de bestemming, waar een episode eindigt. Op CliffWalking veranderden 37 van de 48 rijen: de 10 vakjes van het klif en G blijven 0.

    De beste actions van FrozenLake lijken eerst vreemd: in S, het vakje linksboven, is het 0, links, tegen de rand van het meer, en in de drie vakjes rechts ervan is het 3, omhoog, weer tegen de rand. Op glad ijs is dat slim. Links vanaf S glijdt omhoog of omlaag uit, of blijft staan, en gaat nooit een wak in; omhoog vanaf de bovenste rij glijdt naar links of rechts uit, maar nooit omlaag, naar de wakken bij 5 en 7. De wakken en G tonen 0: elke Q is daar 0, en `np.argmax` neemt de eerste. De grootste Q van S is 0,57, lager dan de kans van 74,0% om G te bereiken: G komt pas na veel stappen, en γ = 0,99 verdisconteert elke stap.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Je `q_episode` in een wereld naar keuze

    De demo traint je `q_episode` in de wereld en de run die je kiest, vanuit een tabel met nullen, zoals de course deed: run 1 met seed 0, run 2 met seed 1, enzovoort. Waar je functie de check nog niet haalt, neemt de functie van de course het over. De grafiek toont de gemiddelde return van de laatste episodes, 100 op FrozenLake, 50 in Taxi en 10 op CliffWalking; op FrozenLake is dat het deel van de episodes dat G bereikt. Ernaast staat wat de grootste Q nu in de wereld doet. Begin met FrozenLake, run 1, en daarna run 4.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    world_choice = mo.ui.dropdown(options=C.world_options(L.lang), value="FrozenLake", label=C.DEMO_TEXT[L.lang]["world"])
    run_choice = mo.ui.dropdown(options=C.run_options(L.lang), value=C.DEMO_TEXT[L.lang]["run"] + " 1", label=C.DEMO_TEXT[L.lang]["run"])
    train_button = mo.ui.run_button(label="Train mijn q_episode")
    mo.hstack([world_choice, run_choice, train_button], justify="start", gap=1.5, wrap=True)
    return run_choice, train_button, world_choice


@app.cell(hide_code=True)
def demo_train(box_episode, result_episode, run_choice, train_button, world_choice):
    if not train_button.value:
        _out = mo.md("*Kies een wereld en een run, en druk dan op de knop om je agent te laten leren.*")
    else:
        _mine = result_episode.state == "correct"
        _fn = run_code(box_episode.value, C.E_NAMES).names["q_episode"] if _mine else C.q_episode
        _name, _seed = world_choice.value, run_choice.value
        _w = C.DEMO_TEXT[L.lang]
        _now = {"shown": 0}

        def _progress(_n, _run):
            if "run" not in _now:
                _now["chart"].watch(lambda: C.now_picture(_now["run"], _now["n"], L.lang))
            _now["run"], _now["n"] = _run, _n
            if _n % C.STRIDE[_name] == 0 or _n == C.EPISODES[_name]:
                _now["chart"].extend(C.chart_points(_run, _now["shown"] + 1, _n), f"**Episode {C.number(_n, 0, L.lang)} van {C.number(C.EPISODES[_name], 0, L.lang)}**")
                _now["shown"] = _n

        _live = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"].format(w=C.WINDOW[_name]),
                        "alt": _w["alt"].format(w=C.WINDOW[_name], world=C.SHORT[_name])},
                       C.EPISODES[_name], role="ink", low=C.RANGE[_name][0], high=C.RANGE[_name][1], window=1,
                       xticks=C.TICKS[_name])
        _live.line("**Bezig met leren** …")
        _now["chart"] = _live
        _r = C.demo_run(_fn, _name, _seed, show=_progress)
        if _r == "error":
            _out = _live.stopped("*Je `q_episode` gaf een fout tijdens de run, of gaf iets wat geen getal is. Haalt je functie de check nog?*")
        elif _r == "slow":
            _out = _live.stopped("*De run duurde langer dan twee minuten. Doet je functie meer werk dan nodig is?*")
        else:
            _who = "je `q_episode`" if _mine else "de `q_episode` van de course, omdat jouw functie de check nog niet haalt"
            _out = mo.vstack([
                L.explore("rewind", "episodes", lambda _v: C.rewind_frame(_r, int(_v), L.lang), start=C.EPISODES[_name],
                          values=C.looks(_name), role="",
                          task="Ga terug naar 0, en dan weer vooruit: vanaf wanneer wijst de grootste Q naar het doel, en wat doet de lijn vanaf dat moment?"),
                mo.md(f"{C.SHORT[_name]}, run {_seed + 1}, met {_who}. {C.now_text(_name, _r['Q'], L.lang)}"),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met de functie van de course leren 3 van de 10 runs van FrozenLake het nooit: runs 1, 2 en 10. De grootste Q wijst in deze runs nooit naar G; run 1 bereikt G één keer, in episode 9.313, en één reward is niet genoeg. De andere zeven bereiken G in 73,0% tot en met 74,0% van de episodes. De oorzaak is het begin. Met elke Q op 0 neemt `np.argmax` de eerste action, links, dus loopt de agent meestal tegen de linkerrand van het meer: met ε = 0,1 bereikt zo'n agent G in ongeveer 1 op de 10.740 episodes, en een willekeurige agent in 1 op de 72. Tot de agent voor het eerst G bereikt, verandert geen enkele reward een Q, en trekt niets de agent weg van links. Begin elke Q op 1 in plaats van 0, een optimistische start zoals in *Gemiddelden bijhouden: verschuivende automaten en optimistische starts*, en elke action lijkt goed tot de agent de action geprobeerd heeft: dan bereiken alle 10 runs G in 74,0% van de episodes.

    In Taxi en op CliffWalking kost elke stap −1, dus lijkt een action die geprobeerd is al snel slechter dan een action op 0 die nog niet geprobeerd is: daar werken de nullen als een eigen optimistische start. Alle 10 runs van Taxi brengen alle 300 passagiers weg, met een gemiddelde return van 7,93, het best mogelijke, en alle 10 runs van CliffWalking nemen de 13 stappen langs de rand.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cut():
    mo.md(r"""
    ### Oefening 4. Een robotarm tegen de klok

    Een team traint een robotarm in een simulator om een kopje op een tafel te zetten. Elke stap geeft −1, het kopje neerzetten geeft +10 en beëindigt de episode, en een tijdslimiet kapt elke episode af na 50 stappen. In één episode houdt de arm bij de 50e stap het kopje vlak boven de tafel als de klok afloopt: `terminated` is `False` en `truncated` is `True`.
    """)
    return


@app.cell(hide_code=True)
def choice_cut_cell():
    choice_cut = L.choice(
        "cut_off",
        question="Naar welk target moet de Q van deze laatste stap opschuiven?",
        options={
            "right": "−1 plus γ keer de grootste Q waar de arm nu is",
            "reward": "Alleen −1, want de episode is voorbij",
            "zero": "0, want een afgekapte stap hoort niet mee te tellen",
            "ten": "+10, de reward voor het kopje neerzetten, want de arm stond op het punt om dat toch te doen",
        },
        correct="right",
        why={
            "right": "De klok hoort niet bij de taak: de wereld gaat door vanaf waar de arm het kopje vasthoudt, één stap van +10. Dus houdt het target γ keer de grootste Q daar, net als na elke andere stap. Met alleen −1 zou de arm leren dat het kopje boven de tafel houden weinig waard is, alleen omdat de klok daar vaak afloopt.",
            "reward": "De episode is voorbij voor de agent, maar is de taak dat ook? Wat was er gevolgd als de klok niet was afgelopen?",
            "zero": "De stap is gebeurd, en de reward telt mee zoals elke andere. En de state waarin de arm na de stap is?",
            "ten": "Een target kan geen reward tellen die niet kwam: welk getal uit de Q-table zegt wat de state van de arm waard is?",
        },
    )
    choice_cut
    return (choice_cut,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl) en [Voorzichtig of gedurfd: SARSA en Q-learning](/course/open/rl-sarsa?lang=nl) kwam elke wereld met eigen code. Met Gymnasium leert dezelfde `q_episode` FrozenLake, Taxi en CliffWalking, en werkt elke methode uit dit domein in elke wereld die de interface volgt. De volgende les, *CartPole, MountainCar en de benchmarks van RL*, gaat naar werelden waarin een observation een lijst reële getallen is in plaats van een vakje, zodat een Q-table deze getallen eerst in bakjes moet verdelen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_cut, result_episode, result_play, result_target):
    L.wrap_up(result_play, result_target, result_episode, choice_cut)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
