# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-libraries. Route: course/route/lessen/reinforcement-learning.yaml. Checks, de runs van PPO van de course, de stand-in-modellen en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import torch
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-libraries", "nl")
    C = L.checks   # de runs van PPO in CartPole van de course, de stand-in-modellen van de check, en de afbeeldingen
    torch.set_num_threads(1)   # de kleine netwerken van PPO zijn op één thread even snel, en een run hangt dan niet af van het aantal kernen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [CartPole, MountainCar en de benchmarks van RL](/course/open/rl-classic-environments?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rows():
    quick_rows = L.choice(
        "quick-rows",
        question="De course verdeelde de hoek en de hoeksnelheid van CartPole elk in 12 bins, en gaf de positie en de snelheid van het karretje elk 1 bin. Hoeveel rijen kreeg de Q-table?",
        options={
            "plus": "26, één voor elke bin van elk getal",
            "right": "144, één voor elk paar bins van de paal",
            "four": "48, 12 bins voor elk van de vier getallen",
            "all": "20.736: 12 × 12 × 12 × 12, want elk getal telt mee",
        },
        correct="right",
        why={
            "right": "12 × 12 × 1 × 1 = 144 rijen met elk 2 Q's, met bins die met de hand gekozen zijn. Het netwerk van deze les neemt de vier getallen direct, zonder bins.",
            "plus": "Een rij staat voor één combinatie van bins, één bin van elk getal tegelijk: hoeveel combinaties maken 12, 12, 1 en 1 bins? Zie [CartPole, MountainCar en de benchmarks van RL](/course/open/rl-classic-environments?lang=nl).",
            "four": "Vier keer 12 telt bins, geen combinaties van bins, en de getallen van het karretje kregen geen 12 bins: hoeveel bins kregen deze getallen? Zie [CartPole, MountainCar en de benchmarks van RL](/course/open/rl-classic-environments?lang=nl).",
            "all": "De positie en de snelheid van het karretje kregen elk 1 bin, dus negeert de tabel deze twee getallen: hoeveel rijen blijven er dan over? Zie [CartPole, MountainCar en de benchmarks van RL](/course/open/rl-classic-environments?lang=nl).",
        },
    )
    quick_rows
    return (quick_rows,)


@app.cell(hide_code=True)
def quick_check_greedy():
    quick_greedy = L.choice(
        "quick-greedy",
        question="De course testte de Q-learner met alleen de grootste Q, zonder willekeurige actions. Waarom laat je de willekeurige actions weg in een test?",
        options={
            "right": "Om te zien wat de agent leerde, zonder het verkennen van de training",
            "speed": "Willekeurige actions maken een test trager om te berekenen",
            "rule": "Gymnasium staat willekeurige actions alleen toe tijdens de training",
            "more": "Zodat de agent tijdens de test nog van de eigen fouten kan blijven leren",
        },
        correct="right",
        why={
            "right": "Willekeurige actions zijn om dingen uit te proberen tijdens het leren; een test vraagt wat de agent doet met wat de agent weet. Met alleen de grootste Q bleef de paal gemiddeld 49 tot 309 stappen staan. PPO van deze les verkent op een eigen manier, en ook dat zet een test uit.",
            "speed": "Een willekeurige action is net zo snel gekozen als de grootste Q: gaat de test over tijd? Zie [CartPole, MountainCar en de benchmarks van RL](/course/open/rl-classic-environments?lang=nl).",
            "rule": "Gymnasium neemt elke action aan die de agent geeft, willekeurig of niet: wie beslist welke actions een test neemt? Zie [CartPole, MountainCar en de benchmarks van RL](/course/open/rl-classic-environments?lang=nl).",
            "more": "Een test verandert niets aan de Q-table: wat moet een test dan meten? Zie [CartPole, MountainCar en de benchmarks van RL](/course/open/rl-classic-environments?lang=nl).",
        },
    )
    quick_greedy
    return (quick_greedy,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    De eigen Q-learner van de course met 12 bins hield de paal van CartPole in de tests 49 tot 309 stappen overeind, ver van de 500 van de tijdslimiet, en werkte met bins die met de hand gekozen waren. Wie RL gebruikt, schrijft het algoritme zelden zelf. Een kleine fout in de code van een algoritme voor RL stopt het algoritme zelden met een foutmelding: de agent leert gewoon slechter, en niets zegt waarom. Toen Henderson en anderen (2018) hetzelfde algoritme uit verschillende code draaiden, met dezelfde instellingen, liepen de resultaten ver uiteen. Een **RL-library** heeft versies van de algoritmen die getest zijn tegen bekende resultaten op benchmark environments, met één interface voor allemaal. In deze les traint **Stable-Baselines3** een agent op CartPole in een paar regels. Een library maakt één run alleen nog niet betrouwbaar: een andere seed geeft een andere run. Dus leer je ook een agent te beoordelen zoals onderzoek in RL dat doet, over meerdere seeds en veel testepisodes.

    **Een voorbeeld uit het dagelijks leven.** Je koopt een weegschaal in plaats van er zelf een te bouwen: de fabriek heeft de weegschaal getest, dus vertrouw je de getallen meer dan de getallen van een zelfgebouwde. Toch zegt één keer wegen vlak na het eten weinig over wat je echt weegt. Je weegt je op meerdere ochtenden en kijkt naar allemaal.
    """)
    return


@app.cell(hide_code=True)
def predict_seeds():
    guess_seeds = L.guess(
        "seeds",
        question="PPO van Stable-Baselines3 traint 25.000 stappen op CartPole, met de standaardinstellingen van de library: zo'n 20 seconden op de computer van de course. Vijf seeds trainen elk een agent, en elke agent wordt getest op 100 episodes. In hoeveel van de 500 testepisodes blijft de paal alle 500 stappen staan, denk je?",
        options={
            "few": "Zo'n 1 op de 10",
            "half": "Ongeveer de helft",
            "right": "Allemaal, zonder uitzondering",
            "none": "Geen enkele: 25.000 stappen zijn veel te weinig voor een neural network",
        },
        reveal="**Allemaal, zonder uitzondering.** Na 25.000 stappen, tussen 288 en 341 episodes training, houdt elk van de 5 seeds de paal alle 500 stappen overeind, in elk van de 100 testepisodes. De Q-table van de course met 12 bins kwam na 2.000 episodes tot gemiddeld 49 tot 309 stappen. Met 10.000 stappen verschillen de seeds nog flink: het gemiddelde van de 100 testepisodes ligt tussen 293 en 416, afhankelijk van de seed. En de demo laat zien dat ook na 25.000 stappen niet elke seed perfect is.",
    )
    guess_seeds
    return (guess_seeds,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een library van algoritmen.** Stable-Baselines3, kort SB3 (Raffin en anderen, 2021), is een Python-library van geteste algoritmen voor RL, gebouwd op PyTorch: PPO, DQN, A2C, SAC, TD3 en DDPG. Elk algoritme is een class met dezelfde paar methods. `PPO("MlpPolicy", env, seed=0, device="cpu")` maakt een model: met de policy `"MlpPolicy"`, de environment `env`, een seed voor alle willekeurige getallen, en de processor, `"cpu"`, omdat de kleine netwerken daar sneller draaien dan op een videokaart. `model.learn(total_timesteps=25000)` traint het model 25.000 stappen van de environment, over zoveel episodes als deze stappen opleveren. `model.predict(obs)` geeft de action voor een observation. `model.save("ppo_cartpole")` schrijft het getrainde model naar een bestand, en `PPO.load("ppo_cartpole")` leest het terug. DQN en A2C werken net zo: alleen de naam van de class verandert.

        **PPO.** Proximal policy optimization (Schulman en anderen, 2017) leert de policy zelf, zonder Q-table. De policy is een neural network: met `"MlpPolicy"` een MLP met twee hidden layers van 64 neuronen, dat de vier getallen van CartPole direct krijgt, zonder bins, en een kans geeft voor elke action. Daarnaast traint PPO een tweede netwerk met dezelfde vorm als value function: net als V in eerdere lessen schat dit netwerk de return vanuit een state. PPO speelt 2.048 stappen met de huidige policy, en beoordeelt dan elke action: ging het beter dan V verwachtte, of slechter? Daarna stuurt PPO de policy bij, zodat de actions die beter uitpakten waarschijnlijker worden en de andere minder, in 10 rondes over deze 2.048 stappen, maar nooit ver van de policy waarmee PPO speelde. Dat laatste is het *proximal*, en het houdt de training rustig. Dan speelt PPO de volgende 2.048 stappen. Een later niveau van de course bouwt PPO zelf; hier gebruik je het.
        """),
        mo.Html(C.ppo_picture(L.lang)),
        mo.md(r"""
        **Verkennen en testen.** Bij de training trekt PPO elke action uit de kansen van het netwerk: een action met een kans van 0,8 wordt 8 van de 10 keer genomen. Zo verkent PPO, in plaats van met ε-greedy, en terwijl PPO leert, schuiven de kansen naar 0 en 1, dus verkent PPO minder. Een test moet laten zien wat de agent geleerd heeft, dus neemt een test elke keer de waarschijnlijkste action. `predict` doet dat als je het argument `deterministic` de waarde `True` geeft, met de naam erbij; zonder dat argument trekt `predict` net als bij de training. `predict` geeft twee dingen terug: de action, en een state die alleen netwerken met een geheugen gebruiken, dus `action, _ = model.predict(...)` houdt de action en laat de state vallen.

        **Een agent beoordelen.** Eén run van RL is één steekproef. De seed bepaalt de eerste getallen van de netwerken, de trekkingen en de starts van de episodes, en twee seeds kunnen flink verschillend leren. Henderson en anderen (2018) trainden hetzelfde algoritme met dezelfde instellingen op twee groepen van 5 seeds, en de twee groepen leken twee verschillende algoritmen. Beoordeel een agent dus zo: train de agent met meerdere seeds, 5 of meer; test elke getrainde agent op veel episodes, zoals 100, met seeds die de training niet gebruikte; en meld elke seed, of de mediaan en de slechtste seed, nooit alleen de beste seed. Agarwal en anderen (2021) laten zorgvuldiger manieren zien om resultaten over een paar seeds samen te vatten.

        **Drie libraries.** Stable-Baselines3 is een van meerdere libraries, elk gemaakt voor iets anders:

        | Library | Gemaakt voor | Hoe |
        |---|---|---|
        | Stable-Baselines3 | een getest algoritme gebruiken in een paar regels | één interface voor elk algoritme, met documentatie en tests tegen bekende resultaten |
        | CleanRL | een algoritme lezen en aanpassen | elk algoritme in één bestand van een paar honderd regels, van het maken van de environment tot de laatste update |
        | RLlib | veel environments en learners tegelijk | onderdeel van het package Ray, dat het werk verdeelt over veel processorkernen of computers |

        De course installeert alleen Stable-Baselines3. CleanRL (Huang en anderen, 2022) is bedoeld om te lezen en over te nemen, bestand voor bestand, en RLlib (Liang en anderen, 2018) komt met Ray mee.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **Stable-Baselines3** | steebel beesleins three | een Python-library van geteste algoritmen voor RL met één interface, kort SB3 | `from stable_baselines3 import PPO` |
        | **PPO** | pee-pee-oo | proximal policy optimization: leert een policy als neural network, telkens een klein stukje | `PPO("MlpPolicy", env)` |
        | **CleanRL** | klien ar-el | algoritmen voor RL, elk in één bestand om te lezen en aan te passen | |
        | **RLlib** | ar-el-lib | een library die environments en learners verdeelt over veel processorkernen of computers | |
        | `total_timesteps` | total time steps | hoeveel stappen van de environment `learn` traint, over alle episodes samen | `model.learn(total_timesteps=25000)` |
        | `deterministic` | | een argument van `predict`: met `True` de waarschijnlijkste action in plaats van een trekking | `model.predict(obs, ...)` |
        """),
        L.watch_out("Eén run zegt weinig. Een resultaat van RL geldt voor de seeds en de testepisodes waarop het gemeten is: hoe meer van allebei, hoe meer het zegt."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een agent trainen met een RL-library, en de agent beoordelen:

    1. Maak de environment, en het model met een eigen seed: `PPO("MlpPolicy", env, seed=..., device="cpu")`.
    2. Laat het model leren: `model.learn(total_timesteps=...)`.
    3. Test het model in een eigen environment, op veel episodes met seeds die de training niet gebruikte, met bij elke stap de waarschijnlijkste action.
    4. Doe stap 1 tot en met 3 voor meerdere seeds, 5 of meer.
    5. Meld elke seed, of de mediaan en de slechtste seed van de testgemiddelden; nooit alleen de beste seed.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel importeert `PPO`, maakt CartPole, en traint een model met seed 0 voor 10.000 stappen: zo'n 8 seconden op de computer van de course. `model.num_timesteps` is hoeveel stappen het model echt nam. Daarna speelt de cel één testepisode in een eigen environment, `_test`, met seed 1000, met bij elke stap de waarschijnlijkste action; elke 100 stappen, en aan het eind, print de cel de positie van het karretje en de hoek van de paal. Voer de cel uit.
    """)
    return


@app.cell
def example_ppo():
    import gymnasium as gym
    from stable_baselines3 import PPO

    _env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", _env, seed=0, device="cpu")
    model.learn(total_timesteps=10000)
    print("getraind:", model.num_timesteps, "stappen")

    _test = gym.make("CartPole-v1")
    _obs, _info = _test.reset(seed=1000)
    _steps = 0
    _terminated, _truncated = False, False
    while not (_terminated or _truncated):
        _action, _ = model.predict(_obs, deterministic=True)   # de waarschijnlijkste action
        _obs, _reward, _terminated, _truncated, _info = _test.step(_action)
        _steps += 1
        if _steps % 100 == 0 or _terminated:
            print("stap", _steps, "positie", round(float(_obs[0]), 2), "hoek", round(float(_obs[2]), 3))
    print("terminated", _terminated, "truncated", _truncated, "na", _steps, "stappen")
    return PPO, gym, model


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `learn` stopt bij 10.240 stappen, niet bij 10.000: PPO leert van hele batches van 2.048 stappen, en 5 batches zijn de eerste die 10.000 halen. In de testepisode blijft de paal alle 500 stappen staan, tot de tijdslimiet de episode afkapt: `truncated` is `True`. Het karretje schuift langzaam naar rechts, tot 0,22, en bij elke print ligt de hoek binnen 0,003 radialen van 0. Eén episode zegt alleen weinig: over 100 testepisodes, seeds 1000 tot en met 1099, houdt deze agent de paal gemiddeld 416 stappen overeind, en in 59 ervan alle 500. Op je eigen computer kan de run iets anders uitpakken: de kleine verschillen in hoe processors getallen afronden, groeien over de duizenden updates van een run.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_play():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een eigen testepisode

        `play_test(model, seed)` speelt één testepisode van CartPole met een getraind `model` en geeft de return terug. De functie maakt de wereld met `gym.make`, start de wereld met `env.reset(seed=seed)`, dat `obs, info` geeft, en speelt tot `terminated` of `truncated`; `env.step(action)` geeft `obs, reward, terminated, truncated, info`, en `total` telt elke `reward` op. Vervang de `...` in de regel `action, _ = ...` zo dat `model.predict` de waarschijnlijkste action voor `obs` geeft.

        De check roept `play_test(steady, 0)` aan. `steady` is een stand-in voor een getraind model: met een `predict` zoals een model van Stable-Baselines3, met een vaste policy erin in plaats van een netwerk, de regel `3 * obs[2] + obs[3] > 0` van *CartPole, MountainCar en de benchmarks van RL*. Met de waarschijnlijkste action geeft `steady` de action van deze policy; trekt `steady` een action, zoals bij de training, dan duwt `steady` een op de 4 keer de andere kant op. Daarna vier andere gevallen: `steady` met seed 1, en twee andere stand-ins, `falling`, die duwt naar waar de paal heen valt, en `leaning`, die duwt naar waar de paal heen helt.
        """),
        L.task("Laat `play_test` bij elke stap `model` om de waarschijnlijkste action vragen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_play():
    box_play = L.code_box("play_test", lines=10)
    box_play
    return (box_play,)


@app.cell(hide_code=True)
def check_play(box_play):
    result_play = L.check("play_test", box_play.value)
    result_play
    return (result_play,)


@app.cell(hide_code=True)
def hints_play():
    L.hints(
        "play_test",
        [
            "Stap 3 van het recept: bij elke stap moet `action, _` krijgen wat `model.predict` voor `obs` geeft, met de waarschijnlijkste action in plaats van een trekking. Welk argument van `model.predict` zet het trekken uit, en welke waarde hoort erbij?",
            "`predict` heeft de plaatsen `observation`, `state`, `episode_start` en `deterministic`, in deze volgorde, dus komt een losse `True` op de tweede plaats in `state` terecht. Geef het argument met de naam erbij, met `=` tussen de naam en de waarde.",
            "`        action, _ = model.predict(obs, ___)`",
        ],
        why="`action, _ = model.predict(obs, deterministic=True)` neemt bij elke stap de waarschijnlijkste action: `steady` houdt de paal dan alle 500 stappen overeind, `falling` tot het karretje van de baan rijdt, en `leaning` tot de paal valt.",
    )
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    In de afbeelding de return van je `play_test` met elke stand-in en de seeds 0 tot en met 4. Een test met de waarschijnlijkste action laat zien wat een policy doet als de policy het meent. Met trekkingen, zoals bij de training, houdt `steady` de paal over dezelfde vijf seeds gemiddeld 451 stappen overeind, en met seed 4 maar 334: een test die trekt, meet de policy en het toeval samen. In de demo hieronder speelt je `play_test` met echte agents van Stable-Baselines3, en de `predict` van zo'n agent werkt op dezelfde manier.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_judge():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Beoordelen over seeds

        `judge(returns)` geeft twee getallen terug over een reeks testresultaten: de mediaan van de seeds, en de slechtste seed. `returns` is een NumPy-array met een rij per seed en een kolom per testepisode, zoals `returns_10k`: de testreturns van de seeds 0 tot en met 4 van de course na 10.000 stappen, 5 rijen van 100. `per_seed` moet het gemiddelde van elke rij bevatten, één getal per seed; `np.median(per_seed)` is de mediaan van deze gemiddelden, en `per_seed.min()` het kleinste. Vervang de `...`: het gemiddelde van elke rij van `returns`.

        De check roept `judge(returns_10k)` aan. Daarna drie andere gevallen: `returns_25k`, dezelfde seeds na 25.000 stappen; `returns_mixed`, 7 seeds van 20 testepisodes elk; en `returns_two`, 2 seeds van 3.
        """),
        L.task("Laat `per_seed` één gemiddelde per seed bevatten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_judge():
    box_judge = L.code_box("judge", lines=3)
    box_judge
    return (box_judge,)


@app.cell(hide_code=True)
def check_judge(box_judge):
    result_judge = L.check("judge", box_judge.value)
    result_judge
    return (result_judge,)


@app.cell(hide_code=True)
def hints_judge():
    L.hints(
        "judge",
        [
            "Stap 5 van het recept: één gemiddelde per seed. `per_seed` moet `returns.mean(...)` zijn met de as die langs de testepisodes van één seed loopt, zodat elke rij van `returns` één getal geeft. Welke as is dat, in een array met een rij per seed?",
            "As 0 loopt langs de kolommen naar beneden, dwars door de rijen: een gemiddelde over de seeds, één voor elke testepisode. De andere as loopt langs elke rij. Geef de as aan `mean` met de naam erbij, `axis`.",
            "`    per_seed = returns.mean(___)`",
        ],
        why="`per_seed = returns.mean(axis=1)` geeft één gemiddelde per seed: voor `returns_10k` 416,39, 398,43, 292,57, 293,41 en 353,09, dus een mediaan van 353,09 en een slechtste seed van 292,57.",
    )
    return


@app.cell(hide_code=True)
def text_after_judge():
    mo.md(r"""
    In de afbeelding de vijf seeds van `returns_10k`, met je mediaan en je slechtste seed. Na 10.000 stappen houdt de beste seed de paal gemiddeld 416 stappen overeind en de slechtste 293: door één seed te kiezen kan hetzelfde algoritme er veel beter of veel slechter uitzien. De mediaan is de middelste seed, en de seed met het meeste geluk of de meeste pech verschuift de mediaan nauwelijks; de slechtste seed laat zien hoe slecht een run kan gaan. De beste seed alleen zegt geen van beide.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Zelf vijf seeds trainen en testen

    De demo traint PPO met de standaardinstellingen van Stable-Baselines3, net als het uitgewerkte voorbeeld, voor het aantal stappen dat je kiest, met vijf seeds: 0 tot en met 4, of 5 tot en met 9. De grafiek toont de return van elke episode van de training, dun, en het gemiddelde van de laatste 10, dik; een seed die klaar is, blijft staan als streepjeslijn. Ernaast wat het policy-netwerk nu doet: voor elke hoek en hoeksnelheid, met het karretje in het midden en stilstaand, hoe vaak het netwerk naar rechts duwt. De lijn is waar `3 * obs[2] + obs[3]` 0 is, de policy die de paal overeind hield in *CartPole, MountainCar en de benchmarks van RL*. Na de training speelt je `play_test` met elke agent 100 testepisodes, en geeft je `judge` de mediaan en de slechtste seed; haalt een van je functies de check nog niet, dan gebruikt de demo de functie van de course. Begin met 10.000 stappen en seeds 0 tot en met 4: zo'n minuut op de computer van de course.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    steps_choice = mo.ui.dropdown(options=C.steps_options(L.lang), value=C.number(10000, 0, L.lang), label=C.DEMO_TEXT[L.lang]["steps"])
    seeds_choice = mo.ui.dropdown(options=C.set_options(L.lang), value=C.DEMO_TEXT[L.lang]["set"].format(a=0, b=4), label=C.DEMO_TEXT[L.lang]["seeds"])
    train_button = mo.ui.run_button(label="Train en test vijf seeds")
    mo.hstack([steps_choice, seeds_choice, train_button], justify="start", gap=1.5, wrap=True)
    return seeds_choice, steps_choice, train_button


@app.cell(hide_code=True)
def demo_train(box_judge, box_play, result_judge, result_play, seeds_choice, steps_choice, train_button):
    if not train_button.value:
        _out = mo.md("*Kies de stappen en de seeds, en druk dan op de knop om vijf agents te trainen en te testen.*")
    else:
        _mine = result_play.state == "correct", result_judge.state == "correct"
        _play = run_code(box_play.value, C.P_NAMES).names["play_test"] if _mine[0] else C.play_test
        _judge = run_code(box_judge.value, C.J_NAMES).names["judge"] if _mine[1] else C.judge
        _steps, _seeds = steps_choice.value, C.seed_list(seeds_choice.value)
        _w = C.DEMO_TEXT[L.lang]
        _runs, _now = [], {}

        def _progress(_n, _run):
            if _now.get("run") is not _run:
                _now["run"] = _run
                _now["chart"].watch(lambda: C.now_picture(_now["run"], L.lang))
            _now["chart"].extend([(_n, _run["episodes"][-1][1])])

        for _k, _seed in enumerate(_seeds):
            _live = L.live(C.chart_labels(_seed, L.lang), C.actual(_steps), role="ink", low=0, high=500,
                           window=C.WINDOW, xticks=C.ticks(C.actual(_steps)))
            for _done in _runs:
                _live.reference(C.curve(_done), "", role="ink")
            _now["chart"] = _live
            _live.line(_w["learning"].format(s=_seed, k=_k + 1, n=len(_seeds)))
            _r = C.train(_seed, _steps, show=_progress, limit=150.0)
            if _r == "slow":
                break
            _runs.append(_r)
        if len(_runs) < len(_seeds):
            _out = _live.stopped("*Een seed had meer dan tweeënhalve minuut nodig om te trainen, dus stopte de demo. Is je computer met iets anders bezig?*")
        else:
            _live.line(_w["testing"].format(n=len(_runs)))
            _t = C.demo_test(_runs, _play, _judge)
            if _t == "error":
                _out = _live.stopped("*Je `play_test` of je `judge` gaf een fout, of iets anders dan getallen. Halen je functies de check nog?*")
            elif _t == "slow":
                _out = _live.stopped("*Het testen duurde meer dan twee minuten. Doet je `play_test` meer werk dan nodig is?*")
            else:
                _who = {(True, True): "je `play_test`", (True, False): "je `play_test` en de `judge` van de course, omdat je eigen `judge` de check nog niet haalt",
                        (False, True): "de `play_test` van de course, omdat je eigen `play_test` de check nog niet haalt", (False, False): "de `play_test` en de `judge` van de course, omdat je eigen functies de check nog niet halen"}[_mine]
                _out = mo.vstack([
                    L.explore("rewind", "stappen", lambda _v: C.rewind_frame(_runs[0], int(_v), L.lang), start=_runs[0]["steps"],
                              values=C.looks(_runs[0]), role="",
                              task=f"Dit is seed {_runs[0]['seed']}. Ga terug naar 0, en dan weer vooruit: wanneer gaat de afbeelding van de policy op de lijn lijken, en wat doet de grafiek dan?"),
                    mo.md(C.table(_runs, _t["returns"], L.lang)),
                    mo.md(_w["verdict"].format(who=_who, median=C.number(_t["median"], 2, L.lang), worst=C.number(_t["worst"], 2, L.lang),
                                               n=len(_runs), steps=C.number(_steps, 0, L.lang))),
                ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met 10.000 stappen halen seeds 0 tot en met 4 in de tests een gemiddelde van 293 tot 416 stappen, en geen van de vijf houdt de paal in elke testepisode overeind. Met 25.000 stappen doen alle vijf dat wel, in alle 100 testepisodes. Seeds 5 tot en met 9 vertellen een minder net verhaal. Na 25.000 stappen laat seed 5 de paal nog in 2 van de 100 testepisodes vallen, en seed 8 in 1. En seed 8 wordt eerst slechter en dan pas beter: na 5.000 stappen, eigenlijk 6.144, halen de tests gemiddeld 325 stappen, na 10.000 nog maar 199. Meer stappen helpen niet altijd meteen, en vijf seeds die er allemaal perfect uitzien, beloven niet dat een zesde dat ook is.

    De dikke lijn van de grafiek blijft onder de tests. Bij de training trekt PPO de actions, en elke trekking die de verkeerde kant op gaat, kan een episode vroeg laten eindigen: seed 0 haalt bij 10.000 stappen gemiddeld 111 stappen in de laatste 10 episodes van de training, tegen 416 in de tests. Beoordeel een agent op de tests, niet op de grafiek van de training.

    De afbeelding van de policy laat zien wat het netwerk geleerd heeft. Het netwerk duwt naar rechts waar de paal naar rechts valt, en de hoek kantelt deze grens, iets minder dan de lijn. Het netwerk vond een eigen policy, dicht bij de ene regel die de paal overeind hield in *CartPole, MountainCar en de benchmarks van RL*, alleen uit rewards.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fair():
    mo.md(r"""
    ### Oefening 3. Een goed verslag

    Een team vergelijkt een nieuwe agent met PPO op CartPole. Het team traint elk van de twee met 10 seeds en test elke getrainde agent op 100 episodes. Bij de nieuwe agent houden 3 seeds de paal in elke testepisode overeind, en halen de andere 7 gemiddeld 150, 175, 190, 205, 220, 240 en 260 stappen. Bij PPO houden 9 seeds de paal in elke testepisode overeind, en haalt er één 410.
    """)
    return


@app.cell(hide_code=True)
def choice_fair_cell():
    choice_fair = L.choice(
        "fair",
        question="Welke zin geeft een juist beeld van de nieuwe agent?",
        options={
            "best": "De agent houdt de paal in elke testepisode alle 500 stappen overeind, net als PPO",
            "right": "Mediaan 230, slechtste seed 150, en 3 van de 10 seeds op 500: onder PPO",
            "train": "De beste episode van de training haalde 500 stappen, dus de agent leert CartPole",
            "mean": "De agent haalt gemiddeld 500 stappen, in de 3 beste seeds",
        },
        correct="right",
        why={
            "right": "De mediaan, de slechtste seed en het aantal seeds op 500 komen uit alle 10 seeds, dus ziet een lezer de 3 goede seeds en de 7 andere. PPO, met een mediaan van 500 en een slechtste seed van 410, doet het beter.",
            "best": "Dat geldt voor 3 van de 10 seeds: wat gebeurde er bij de andere 7?",
            "train": "Een episode van de training is één trekking uit de kansen van de policy, geen test: hoe deden de getrainde agents het in de tests?",
            "mean": "De 3 beste seeds achteraf uitkiezen laat er 7 weg: laat zo'n keuze niet elk algoritme er goed uitzien?",
        },
    )
    choice_fair
    return (choice_fair,)


@app.cell(hide_code=True)
def text_exercise_library():
    mo.md(r"""
    ### Oefening 4. Welke library?

    Een student wil elke regel van PPO zien, van de batch van 2.048 stappen tot het bijsturen dat elke verandering van de policy klein houdt, en daarna een andere manier proberen om de verandering klein te houden.
    """)
    return


@app.cell(hide_code=True)
def choice_library_cell():
    choice_library = L.choice(
        "library",
        question="Welke library past het best?",
        options={
            "sb3": "Stable-Baselines3: de best geteste PPO, gebruikt in een paar regels",
            "right": "CleanRL: PPO in één bestand van een paar honderd regels",
            "rllib": "RLlib: PPO verdeeld over veel kernen en computers tegelijk",
            "none": "Geen van de drie: een library is niet te lezen, dus schrijf PPO eerst zelf vanaf nul",
        },
        correct="right",
        why={
            "right": "CleanRL zet heel PPO in één bestand, dus kan de student het bestand van boven naar beneden lezen en één regel aanpassen. Stable-Baselines3 verdeelt PPO over Python-classes in meerdere bestanden, gemaakt om te gebruiken en niet om te lezen, en RLlib is gemaakt voor schaal.",
            "sb3": "Stable-Baselines3 is gemaakt om te gebruiken: PPO is er verdeeld over meerdere Python-classes in meerdere bestanden. Welke library houdt één algoritme in één bestand?",
            "rllib": "RLlib is gemaakt om op grote schaal te draaien, en de code is verdeeld over een groot package: welke library is gemaakt om te lezen?",
            "none": "Alle drie zijn open source, dus alle code is te lezen: welke library houdt PPO in één bestand?",
        },
    )
    choice_library
    return (choice_library,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [CartPole, MountainCar en de benchmarks van RL](/course/open/rl-classic-environments?lang=nl) leerde een tabel met bins CartPole maar voor een deel. Hier leerde PPO uit een library, met een neural network in plaats van een tabel, CartPole in 25.000 stappen, beoordeeld over vijf seeds van 100 testepisodes elk. Een later niveau van de course maakt PPO open en bouwt het zelf. De volgende les, *De honden van Pavlov en de katten van Thorndike: conditionering*, gaat terug naar waar leren van rewards begon: in de psychologie, toen er nog geen computers waren.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fair, choice_library, result_judge, result_play):
    L.wrap_up(result_play, result_judge, choice_fair, choice_library)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
