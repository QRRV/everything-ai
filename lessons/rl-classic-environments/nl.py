# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-classic-environments. Route: course/route/lessen/reinforcement-learning.yaml. Checks, de vaste policies van de course, de Q-learner met bins en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-classic-environments", "nl")
    C = L.checks   # de policies en runs van de course in CartPole en MountainCar, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Eén interface voor elke environment: Gymnasium](/course/open/rl-gymnasium?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_table():
    quick_table = L.choice(
        "quick-table",
        question="Een wereld van Gymnasium heeft `Discrete(500)` als observation space en `Discrete(6)` als action space. Welke vorm krijgt een Q-table voor deze wereld?",
        options={
            "flip": "6 rijen en 500 kolommen",
            "right": "500 rijen, één per observation, en 6 kolommen",
            "plus": "506 rijen: één voor elke observation en één voor elke action",
            "one": "500 rijen met elk één getal",
        },
        correct="right",
        why={
            "right": "Een rij per observation en een kolom per action: 500 × 6 = 3.000 Q's, zoals in Taxi. In deze les zijn de observations reële getallen, en moet een tabel deze getallen eerst in bins verdelen.",
            "flip": "De agent zoekt de rij van de observation op en vergelijkt dan de actions binnen deze rij: welke van de twee heeft de rijen nodig? Zie [Eén interface voor elke environment: Gymnasium](/course/open/rl-gymnasium?lang=nl).",
            "plus": "Een Q hoort bij een paar: één observation met één action. Hoeveel paren zijn er? Zie [Eén interface voor elke environment: Gymnasium](/course/open/rl-gymnasium?lang=nl).",
            "one": "Q-learning houdt een Q bij voor elke action van een observation, om de Q's te vergelijken: hoeveel kolommen vraagt dat? Zie [Eén interface voor elke environment: Gymnasium](/course/open/rl-gymnasium?lang=nl).",
        },
    )
    quick_table
    return (quick_table,)


@app.cell(hide_code=True)
def quick_check_cut():
    quick_cut = L.choice(
        "quick-cut",
        question="Een tijdslimiet kapt een episode af na de 500e stap, terwijl de taak zelf doorgaat. Wat geeft `env.step` voor deze laatste stap als `terminated` en `truncated`?",
        options={
            "ended": "`terminated` is `True`, `truncated` is `False`",
            "right": "`terminated` is `False`, `truncated` is `True`",
            "both": "Allebei `True`, want de episode is hoe dan ook voorbij",
            "none": "Allebei `False`: een limiet zet geen vlag",
        },
        correct="right",
        why={
            "right": "De taak eindigde niet, de klok liep af: `truncated`. Het target van deze stap houdt γ keer de grootste Q van de volgende state. CartPole uit deze les heeft precies zo'n limiet: 500 stappen.",
            "ended": "`terminated` zegt dat de taak zelf eindigde, zoals een auto bij het doel: gebeurde dat hier? Zie [Eén interface voor elke environment: Gymnasium](/course/open/rl-gymnasium?lang=nl).",
            "both": "Voorbij is voorbij, maar de twee vlaggen houden twee redenen uit elkaar: welke reden geldt hier? Zie [Eén interface voor elke environment: Gymnasium](/course/open/rl-gymnasium?lang=nl).",
            "none": "Zonder vlag: hoe weet de code dan dat de episode moet stoppen? Zie [Eén interface voor elke environment: Gymnasium](/course/open/rl-gymnasium?lang=nl).",
        },
    )
    quick_cut
    return (quick_cut,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een nieuwe methode in RL is pas iets waard als de methode het beter doet dan de oude methoden op dezelfde taken. Onderzoekers testen dus op een paar gedeelde werelden, altijd in dezelfde versie, en melden hoe de agent het daar deed. Elke klassieke wereld werd een van deze **benchmark environments** omdat de wereld iets anders test. CartPole, uit 1983, vraagt een agent een paal op een karretje in evenwicht te houden, met vier getallen als houvast. MountainCar, uit 1990, geeft bij elke stap dezelfde −1 tot een auto een vlag bereikt, dus zegt de reward niets over hoe je er komt. Atari-spellen vragen een agent te leren van de pixels van een scherm, en met MuJoCo beweegt een agent lichamen met actions die reële getallen zijn. In deze les speel je CartPole en MountainCar, verdeel je de getallen van deze werelden in bins zodat een Q-table de getallen kan leren, en zie je wat Atari en MuJoCo vragen dat een tabel niet kan geven.

    **Een voorbeeld uit het dagelijks leven.** Een rijexamen toetst parkeren, invoegen op de snelweg en een noodstop. Elk onderdeel test iets anders, en goed parkeren zegt weinig over de noodstop. Doet een methode van RL het alleen goed op CartPole, dan heeft de methode het parkeren gehaald.
    """)
    return


@app.cell(hide_code=True)
def predict_flag():
    guess_flag = L.guess(
        "flag",
        question="In MountainCar duwt een willekeurige agent bij elke stap naar links, helemaal niet of naar rechts, willekeurig. In hoeveel van 1.000 episodes bereikt de agent de vlag binnen de tijdslimiet van 200 stappen, denk je?",
        options={
            "half": "Ongeveer de helft",
            "tenth": "Ongeveer 1 op de 10",
            "right": "Geen enkele",
            "most": "Bijna allemaal, want 200 stappen is ruim genoeg tijd",
        },
        reveal="**Geen enkele.** In 1.000 episodes met willekeurige duwen bereikt de auto de vlag nooit: de duwen heffen elkaar op, en de motor alleen is te zwak voor de helling, want bij elke stap naar rechts duwen komt er ook niet. Duwt een vaste policy steeds in de richting waarin de auto beweegt, dan bereikt de auto de vlag in elk van 100 episodes, in gemiddeld 120 stappen. Voor een agent die leert, zegt de −1 bij elke stap niets tot de agent de vlag één keer bereikt heeft: dat test MountainCar.",
    )
    mo.vstack([mo.Html(C.world_picture(L.lang)), guess_flag])
    return (guess_flag,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Reële getallen.** In CartPole en MountainCar is een observation een paar reële getallen, en is de observation space een `Box`: `Box(low, high, (4,))` bevat 4 getallen, elk tussen een eigen laagste en hoogste waarde. In CartPole is `obs[0]` de positie van het karretje, op een baan van −2,4 tot 2,4, `obs[1]` de snelheid van het karretje, `obs[2]` de hoek van de paal in radialen, positief als de paal naar rechts helt, en `obs[3]` de hoeksnelheid: hoe snel de hoek verandert, positief als de paal naar rechts valt. De action space is `Discrete(2)`: 0 duwt het karretje naar links, 1 naar rechts. Elke stap geeft +1, en een stap duurt 0,02 seconde, dus 500 stappen zijn 10 seconden balanceren. In MountainCar is `obs[0]` de positie van de auto, van −1,2 tot 0,6, en `obs[1]` de snelheid, van −0,07 tot 0,07. De motor trekt met 0,001 en de helling met hooguit 0,0025, dus kan de auto de heuvel niet op rijden: de auto moet heen en weer schommelen om vaart te maken.

        | Wereld | observation space | action space | reward | eindigt als | tijdslimiet |
        |---|---|---|---|---|---|
        | CartPole-v1 | `Box`, 4 getallen | `Discrete(2)`: 0 links, 1 rechts | +1 per stap | paal voorbij 12°, of karretje voorbij ±2,4 | 500 stappen |
        | MountainCar-v0 | `Box`, 2 getallen | `Discrete(3)`: 0 links, 1 niets, 2 rechts | −1 per stap | auto bij de vlag, 0,5 | 200 stappen |

        **Een sparse reward.** In CartPole zegt elke stap de agent iets: de paal staat nog. In MountainCar geeft elke stap −1, wat de agent ook doet, en pas de vlag bereiken stopt de −1's. Tot de agent de vlag één keer bereikt heeft, zegt de reward niets over welke duw beter was. Blijft een reward bijna overal hetzelfde en zegt de reward alleen op zeldzame momenten iets, zoals bij het doel, dan is het een **sparse reward**: zo'n reward maakt exploration moeilijk.

        **Bins.** Een Q-table heeft een rij voor elke observation, en een reëel getal kan eindeloos veel waarden aannemen. **Discretization** verdeelt elk getal in bins. Je kiest een bereik en de randen tussen de bins; de bin van een getal is het aantal randen onder het getal. Met 12 bins voor de positie van MountainCar, van −1,2 tot 0,6, liggen de 11 randen 0,15 uit elkaar, bij −1,05, −0,9, enzovoort tot en met 0,45. De positie −0,5 ligt boven 4 van deze randen, dus komt de positie in bin 4. Een getal met één bin heeft geen randen en komt altijd in bin 0: de tabel negeert het getal. De bins van alle getallen van een observation kiezen samen één rij, zoals de cijfers van een getal: met 10 bins voor elk van vier getallen geven bins 3, 0, 7 en 2 rij 3072. In het algemeen: begin bij rij 0, en vermenigvuldig voor elk getal de rij met het aantal bins van dat getal en tel de bin erbij op. Met 4 bins voor beide getallen van MountainCar geven bin 2 voor de positie en bin 1 voor de snelheid rij 2 × 4 + 1 = 9. De tabel krijgt zoveel rijen als alle aantallen bins met elkaar vermenigvuldigd: 12 × 12 = 144 voor MountainCar met 12 bins elk. CartPole verdeelt in deze les alleen de paal: 1 bin voor de positie van het karretje en 1 voor de snelheid, en zoveel als je kiest voor de hoek en de hoeksnelheid.
        """),
        mo.Html(C.bins_picture(L.lang)),
        mo.md(r"""
        **Pixels en lichamen.** De Arcade Learning Environment (Bellemare en anderen, 2013) maakte van spellen van de Atari 2600, een spelcomputer uit 1977, environments. Een observation is het scherm, 210 bij 160 pixels, elk in een van 128 kleuren; een action is een van hooguit 18 standen van de joystick en de knop; de reward komt uit de score. Geen tabel kan een rij voor elk scherm bevatten: er zijn meer mogelijke schermen dan atomen in het heelal. In 2015 zette DQN (Mnih en anderen) een neural network op de plaats van de tabel. Met hetzelfde network en dezelfde instellingen voor elk spel leerde DQN 49 spellen van alleen de pixels en de score, en in 29 daarvan haalde DQN minstens 75% van de score van een professionele menselijke gametester. **MuJoCo** (Todorov, Erez en Tassa, 2012) is een physics simulator voor lichamen met gewrichten. In Hopper van MuJoCo ziet een lichaam op één been 11 getallen en geeft het 3 getallen terug, een draaikracht voor elk van de drie gewrichten, elk een reëel getal van −1 tot 1: de action space is `Box(-1, 1, (3,))`. Een Q-table heeft een kolom nodig voor elke action; als je elke kracht in 10 bins verdeelt, krijg je al 10 × 10 × 10 = 1.000 kolommen, en de Ant van MuJoCo, met 8 gewrichten, zou er 100.000.000 nodig hebben. De DeepMind Control Suite (Tassa en anderen, 2018) bevat meer taken van deze soort, ook gebouwd op MuJoCo. Atari en MuJoCo vragen packages die de course niet installeert, en deze werelden leren vraagt een neural network in plaats van een tabel, zoals bij DQN.
        """),
        mo.Html(C.bench_picture(L.lang)),
        mo.md(r"""
        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **benchmark environment** | | een wereld waarop veel onderzoekers testen, in een vaste versie, zodat de uitkomsten te vergelijken zijn | `gym.make("CartPole-v1")` |
        | **CartPole** | kart-pool | een paal op een karretje in evenwicht houden met vier getallen, met twee actions: naar links of naar rechts duwen | `"CartPole-v1"` |
        | **sparse reward** | | een reward die alleen op zeldzame momenten iets zegt, zoals bij het doel | |
        | **discretization** | | reële getallen in bins verdelen, zodat een tabel een rij krijgt voor elke combinatie van bins | `row_of(obs, edges)` |
        | **MuJoCo** | moe-jo-ko | een physics simulator voor lichamen met gewrichten, met actions die reële getallen zijn | |
        | `Box(low, high, shape)` | box | reële getallen, elk tussen een eigen laagste en hoogste waarde | `env.observation_space` |
        | randen | | de grenzen tussen de bins van een getal | `edges` |
        """),
        L.watch_out("Bins zijn een keuze, en bepalen wat de tabel uit elkaar kan houden. Een rij dekt elke observation in de bins van de rij tegelijk, en de tabel leert één Q per action voor al deze observations samen."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Q-learning in een wereld waarin de observations reële getallen zijn:

    1. Kies voor elk getal van de observation een bereik en een aantal bins; de randen liggen tussen de bins, gelijk verdeeld over het bereik. Een getal dat er niet toe doet krijgt één bin en geen randen.
    2. De bin van een getal is het aantal randen onder het getal.
    3. De rij van een observation: begin bij 0, en vermenigvuldig voor elk getal de rij met het aantal bins van het getal en tel de bin erbij op.
    4. Maak een Q-table met nullen, met zoveel rijen als alle aantallen bins met elkaar vermenigvuldigd, en een kolom per action.
    5. Leer met Q-learning zoals eerder, met de rij van de observation overal waar de tabel een rij nodig heeft.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    MountainCar, met `env.reset(seed=0)`, en een vaste policy die duwt in de richting waarin de auto beweegt: naar rechts, 2, als de snelheid 0 of meer is, en naar links, 0, als de snelheid onder 0 is. De cel maakt de wereld, print de spaces en speelt één episode; elke 20 stappen, en aan het eind, print de cel de positie en de snelheid. `_steps % 20` is de rest van `_steps` gedeeld door 20, dus `_steps % 20 == 0` is waar bij stap 20, 40, 60 enzovoort. Voer de cel uit.
    """)
    return


@app.cell
def example_car():
    import gymnasium as gym

    _env = gym.make("MountainCar-v0")
    print("observation_space:", _env.observation_space)
    print("action_space:", _env.action_space)
    _obs, _info = _env.reset(seed=0)
    print("start: positie", round(float(_obs[0]), 3), "snelheid", round(float(_obs[1]), 4))
    _steps = 0
    _terminated, _truncated = False, False
    while not (_terminated or _truncated):
        if _obs[1] >= 0:
            _action = 2   # duw naar rechts, de kant op waar de auto heen gaat
        else:
            _action = 0   # duw naar links
        _obs, _reward, _terminated, _truncated, _info = _env.step(_action)
        _steps += 1
        if _steps % 20 == 0 or _terminated:
            print("stap", _steps, "positie", round(float(_obs[0]), 3), "snelheid", round(float(_obs[1]), 4))
    print("terminated", _terminated, "truncated", _truncated, "na", _steps, "stappen")
    return (gym,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De auto begint bij −0,473, stilstaand, dus duwt de policy naar rechts. De auto klimt tot ongeveer −0,30, rolt terug, en vanaf dan duwt de policy naar links: langs het dal en de linkerheuvel op, tot −1,113 bij stap 80. Daar keert de auto, en met de helling en de motor in dezelfde richting raast de auto omlaag en de rechterheuvel op, naar de vlag bij stap 122: `terminated` is `True`. Over 100 episodes bereikt deze policy de vlag elke keer, in gemiddeld 120 stappen; bij elke stap naar rechts duwen komt nooit bij de vlag, en willekeurige duwen deden het in 1.000 episodes niet.

    Nu de bins van één observation, met de hand. Bij stap 60 staat de auto op positie −0,758 met snelheid −0,0303. Met 12 bins voor elk getal liggen de randen van de positie bij −1,05, −0,9, −0,75, …: −0,758 ligt boven de eerste twee, dus is de bin 2. De randen van de snelheid liggen 0,0117 uit elkaar, bij −0,0583, −0,0467, −0,035, −0,0233, …: −0,0303 ligt boven de eerste drie, dus is de bin 3. De rij van deze observation is 2 × 12 + 3 = 27, een van de 144 rijen van de tabel.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_policy():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een vaste policy voor CartPole

        `pole_policy(obs)` krijgt een observation van CartPole en geeft een action terug: 1 duwt het karretje naar rechts, 0 naar links. Van de vier getallen van `obs` is `obs[0]` de positie van het karretje, `obs[1]` de snelheid van het karretje, `obs[2]` de hoek van de paal en `obs[3]` de hoeksnelheid. Een positieve hoek helt naar rechts, en een positieve hoeksnelheid betekent dat de paal naar rechts valt. Vervang de `...` door een test die waar is als de paal naar rechts valt, zodat het karretje eronder geduwd wordt.

        De check roept `hold(pole_policy, 0)` aan: deze functie speelt één episode van CartPole met je policy, vanaf `env.reset(seed=0)`, en geeft het aantal stappen tot de paal valt of het karretje van de baan rijdt, of tot de tijdslimiet de episode afkapt, bij 500. Je policy is goed als de paal minstens 100 stappen blijft staan, met seed 0 en met seeds 1 tot en met 4.
        """),
        L.task("Laat `pole_policy` het karretje duwen naar de kant waar de paal naartoe valt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_policy():
    box_policy = L.code_box("pole_policy", lines=4)
    box_policy
    return (box_policy,)


@app.cell(hide_code=True)
def check_policy(box_policy):
    result_policy = L.check("pole_policy", box_policy.value)
    result_policy
    return (result_policy,)


@app.cell(hide_code=True)
def hints_policy():
    L.hints(
        "pole_policy",
        [
            "De vier getallen van `obs` zijn, op volgorde, de positie en de snelheid van het karretje, en de hoek en de hoeksnelheid van de paal. Welk getal zegt welke kant de paal nu op valt, en welk teken heeft dit getal als de paal naar rechts valt?",
            "De hoek zegt waar de paal is; de hoeksnelheid zegt waar de paal heen gaat. Index 3 van `obs` is de hoeksnelheid, en is groter dan 0 als de paal naar rechts valt.",
            "`    if ___:`",
        ],
        why="`if obs[3] > 0:` duwt het karretje naar rechts precies als de paal naar rechts valt, zodat het karretje steeds onder de paal blijft komen. Over 100 episodes houdt deze ene regel de paal gemiddeld 198 stappen overeind, tot het karretje van de baan rijdt.",
    )
    return


@app.cell(hide_code=True)
def text_after_policy():
    mo.md(r"""
    In de afbeelding staat de hoek met je policy en met willekeurige duwen, in de episode met seed 0. Je policy reageert op waar de paal heen gaat, niet op waar de paal is. Over 100 episodes houdt duwen naar de kant waar de paal helt de paal gemiddeld 41 stappen overeind: deze policy blijft duwen terwijl de paal al terugzwaait, dus wordt elke zwaai groter dan de vorige, 2,7°, 4,8°, 11,4° en 13,3° met seed 0, en dan ligt de paal. De hoeksnelheid alleen houdt de paal 198 stappen overeind, en dan rijdt het karretje van de baan, in elk van de 100 episodes: de paal staat nog, maar de duwen hebben het karretje naar één kant gebracht. Beide samen, `3 * obs[2] + obs[3] > 0`, houden de paal overeind en het karretje op de baan, alle 500 stappen in elk van 100 episodes. Eén regel en vier getallen zijn genoeg voor CartPole, als je de regel kent. Een agent die leert kent de regel niet: de agent moet een policy vinden uit alleen de rewards.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bin():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De bin van een getal

        `bin_of(x, edges)` geeft de bin van het getal `x` terug: het aantal randen in `edges` onder `x`, stap 2 van het recept. `edges` is een NumPy-array van randen, van klein naar groot. Een array vergelijken met een getal geeft een array van `True` en `False`, één voor elk getal van de array, en `np.sum` telt de `True`'s, want `True` telt als 1; `int` maakt van deze telling een gewoon geheel getal. Vervang de `...` door de vergelijking waarvan de `True`'s de randen onder `x` zijn.

        De check roept `bin_of(-0.5, pos_12)` aan, met `pos_12` de 11 randen van de positie van MountainCar met 12 bins: het voorbeeld uit Het idee, bin 4. Daarna vier andere gevallen: een positie onder alle randen, een positie boven alle randen, de hoek 0,05 van CartPole met `angle_12`, de 11 randen van de hoek met 12 bins, en het getal 0,3 met `no_edges`, een lege array: een getal met één bin.
        """),
        L.task("Laat `bin_of` de randen tellen die onder `x` liggen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bin():
    box_bin = L.code_box("bin_of", lines=2)
    box_bin
    return (box_bin,)


@app.cell(hide_code=True)
def check_bin(box_bin):
    result_bin = L.check("bin_of", box_bin.value)
    result_bin
    return (result_bin,)


@app.cell(hide_code=True)
def hints_bin():
    L.hints(
        "bin_of",
        [
            "Stap 2 van het recept: tel de randen die onder `x` liggen. `np.sum` telt de `True`'s van een array van `True` en `False`, en `int` maakt een geheel getal van de telling. Welke vergelijking van `edges` en `x` is `True` voor precies de randen onder `x`?",
            "`np.array([1, 4, 9]) > 3` geeft `[False, True, True]`: één antwoord voor elk getal van de array. In het voorbeeld liggen 4 randen van `pos_12` onder −0,5: −1,05, −0,9, −0,75 en −0,6. Welke kant op moet de vergelijking, zodat deze vier de `True`'s zijn?",
            "`    return int(np.sum(___))`",
        ],
        why="`edges < x` is `True` voor elke rand onder `x`, en `np.sum` telt deze randen: 4 voor −0,5 met `pos_12`, 0 onder alle randen, 11 boven alle randen, en 0 voor een getal zonder randen.",
    )
    return


@app.cell(hide_code=True)
def text_after_bin():
    mo.md(r"""
    In de afbeelding staat je `bin_of` voor zeven posities van MountainCar, met 12 bins. Elke stip ligt tussen twee randen, en de bin van de stip telt de randen links van de stip. Gelijk verdeelde randen zijn de eenvoudigste keuze; niets houdt je tegen om meer randen te leggen waar de getallen vaak zijn, zoals bij een hoek van 0 in CartPole.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_row():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Een rij voor elke combinatie van bins

        `row_of(obs, edges)` geeft de rij van de observation `obs` in een Q-table terug: stap 3 van het recept. `edges` is een lijst met één array van randen voor elk getal van `obs`, dus `edges[i]` bevat de randen van getal `i`, en `len(edges[i])` is het aantal randen van dit getal. `bin_of` is de eigen oplossing van de course voor oefening 2. `for i in range(len(obs)):` gaat de getallen van `obs` een voor een langs; `row = 0` begint de rij bij 0, en elke ronde vermenigvuldigt `row` en telt de bin van getal `i` erbij op. Vervang de `...`: hoeveel bins getal `i` heeft.

        De check roept `row_of([-0.52, 0.004], car_12)` aan, met `car_12` de randen van MountainCar met 12 bins voor elk getal. Daarna vier andere gevallen: een observation van CartPole met `cart_12`, 1 bin voor elk getal van het karretje en 12 voor elk getal van de paal; vier getallen van 0 tot 1 met `digits`, 10 bins elk, waarin de bins de cijfers van de rij zijn; MountainCar met `car_20`, 20 bins elk; en CartPole met `cart_4`.
        """),
        L.task("Laat `row_of` de rij vermenigvuldigen met het aantal bins van elk getal voordat de functie de bin van het getal erbij optelt."),
    ])
    return


@app.cell(hide_code=True)
def exercise_row():
    box_row = L.code_box("row_of", lines=5)
    box_row
    return (box_row,)


@app.cell(hide_code=True)
def check_row(box_row):
    result_row = L.check("row_of", box_row.value)
    result_row
    return (result_row,)


@app.cell(hide_code=True)
def hints_row():
    L.hints(
        "row_of",
        [
            "Stap 3 van het recept: vermenigvuldig voor elk getal `i` van `obs` de rij `row` met het aantal bins van dit getal, en tel de bin erbij op, `bin_of(obs[i], edges[i])`. Hoeveel bins heeft een getal met `len(edges[i])` randen?",
            "11 randen verdelen een bereik in 12 bins, en een lege array van randen laat 1 bin over: altijd één bin meer dan randen. Python vermenigvuldigt voordat het optelt, dus moet je haakjes zetten om een optelling die als geheel vermenigvuldigd wordt.",
            "`        row = row * ___ + bin_of(obs[i], edges[i])`",
        ],
        why="`row = row * (len(edges[i]) + 1) + bin_of(obs[i], edges[i])` maakt van de bins de cijfers van één getal: met `car_12` geven bin 4 voor de positie en bin 6 voor de snelheid 4 × 12 + 6 = 54, en met `digits` geven de bins 3, 0, 7 en 2 rij 3072.",
    )
    return


@app.cell(hide_code=True)
def text_after_row():
    mo.md(r"""
    In de afbeelding staat je `row_of` voor het midden van elk vak van MountainCar met 4 bins voor elk getal. De rijen tellen eerst op langs de snelheid, 0 tot en met 3 in de onderste rij, dan 4 tot en met 7 in de rij erboven, met de volgende bin van de positie: de snelheid is het laatste cijfer, de positie het eerste. Elke combinatie van bins krijgt een eigen rij, en geen twee combinaties delen een rij.

    ### De Q-table van MountainCar met NumPy

    `C.course_q("MountainCar-v0", 12)` geeft de Q-table van run 1 van de course in MountainCar met 12 bins voor elk getal, na 2.000 episodes Q-learning met `row_of`. `.shape` geeft de rijen en kolommen. `np.abs(...).sum(axis=1) > 0` zegt voor elke rij of een van de Q's in de rij veranderd is, en `.sum()` telt deze rijen. `np.argmax(..., axis=1)` geeft de beste action van elke rij, en `.reshape(12, 12)` zet de 144 rijen in een raster: één regel per bin van de positie, van de linkerheuvel bovenaan tot de vlag onderaan, en één kolom per bin van de snelheid, van snel naar links tot snel naar rechts. Voer de cel uit.
    """)
    return


@app.cell
def example_numpy():
    import numpy as np

    _Q = C.course_q("MountainCar-v0", 12)   # run 1 van de course, 12 bins voor elk getal, na 2.000 episodes
    print("shape:", _Q.shape)
    print("rijen die veranderden:", (np.abs(_Q).sum(axis=1) > 0).sum())
    print("de beste action van elke rij, een regel per bin van de positie, een kolom per bin van de snelheid:")
    print(np.argmax(_Q, axis=1).reshape(12, 12))
    return (np,)


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    113 van de 144 rijen veranderden; bij de andere 31 kwam de agent nooit, zoals een hoge snelheid helemaal links, en daar is elke Q nog 0, dus neemt `np.argmax` de eerste action, 0. In de linkerhelft van de kolommen, waar de auto naar links beweegt, zeggen de meeste vakjes 0, naar links duwen; in de kolommen 6 tot en met 9, waar de auto naar rechts beweegt, zeggen de meeste 2, naar rechts duwen. Dat is de schommel uit het uitgewerkte voorbeeld, geleerd uit −1's en een vlag, met een paar vakjes die iets anders zeggen. De demo tekent dezelfde tabel als afbeelding.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Je `row_of` in een wereld naar keuze

    De demo traint de Q-learner van de course 2.000 episodes lang, met η = 0,1, γ = 0,99, ε = 0,1 en elke Q op 0, met je `row_of` en de bins die je kiest: in MountainCar voor beide getallen, in CartPole voor de hoek en de hoeksnelheid. Waar je functie de check nog niet haalt, neemt de functie van de course het over. De grafiek toont het gemiddelde aantal stappen van de laatste 50 episodes: in CartPole is meer beter, tot 500; in MountainCar is minder beter, en 200 betekent dat de auto de vlag niet bereikte. Ernaast staat de beste action in elke combinatie van bins op dit moment. Begin met MountainCar en 12 bins, en probeer daarna 4 en 20.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    world_choice = mo.ui.dropdown(options=C.world_options(L.lang), value="MountainCar", label=C.DEMO_TEXT[L.lang]["world"])
    bins_choice = mo.ui.dropdown(options=C.bins_options(L.lang), value="12", label=C.DEMO_TEXT[L.lang]["bins"])
    run_choice = mo.ui.dropdown(options=C.run_options(L.lang), value=C.DEMO_TEXT[L.lang]["run"] + " 1", label=C.DEMO_TEXT[L.lang]["run"])
    train_button = mo.ui.run_button(label="Train met mijn row_of")
    mo.hstack([world_choice, bins_choice, run_choice, train_button], justify="start", gap=1.5, wrap=True)
    return bins_choice, run_choice, train_button, world_choice


@app.cell(hide_code=True)
def demo_train(bins_choice, box_row, result_row, run_choice, train_button, world_choice):
    if not train_button.value:
        _out = mo.md("*Kies een wereld, de bins en een run, en druk dan op de knop om de agent te laten leren.*")
    else:
        _mine = result_row.state == "correct"
        _fn = run_code(box_row.value, C.R_NAMES).names["row_of"] if _mine else C.row_of
        _name, _bins, _seed = world_choice.value, bins_choice.value, run_choice.value
        _w = C.DEMO_TEXT[L.lang]
        _now = {"shown": 0}

        def _progress(_n, _run):
            if "run" not in _now:
                _now["chart"].watch(lambda: C.now_picture(_now["run"], _now["n"], L.lang))
            _now["run"], _now["n"] = _run, _n
            if _n % C.STRIDE == 0 or _n == C.EPISODES:
                _now["chart"].extend(C.chart_points(_run, _now["shown"] + 1, _n), f"**Episode {C.number(_n, 0, L.lang)} van {C.number(C.EPISODES, 0, L.lang)}**")
                _now["shown"] = _n

        _live = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"].format(w=C.WINDOW),
                        "alt": _w["alt"].format(w=C.WINDOW, world=C.SHORT[_name])},
                       C.EPISODES, role="ink", low=C.RANGE[_name][0], high=C.RANGE[_name][1], window=1, xticks=C.TICKS)
        _live.line("**Bezig met leren** …")
        _now["chart"] = _live
        _r = C.demo_run(_fn, _name, _bins, _seed, show=_progress)
        if _r == "error":
            _out = _live.stopped("*Je `row_of` gaf een fout tijdens de run, of een rij die niet in de tabel past. Haalt je functie de check nog?*")
        elif _r == "slow":
            _out = _live.stopped("*De run duurde langer dan twee minuten. Doet je functie meer werk dan nodig is?*")
        else:
            _who = "je `row_of`" if _mine else "de `row_of` van de course, omdat jouw functie de check nog niet haalt"
            _out = mo.vstack([
                L.explore("rewind", "episodes", lambda _v: C.rewind_frame(_r, int(_v), L.lang, _fn), start=C.EPISODES,
                          values=C.looks(), role="",
                          task="Ga terug naar 0, en dan weer vooruit: wanneer begint de afbeelding zich te vullen, en wat doet de lijn vanaf dat moment?"),
                mo.md(f"{C.SHORT[_name]}, {_bins} bins, run {_seed + 1}, met {_who}. {C.now_text(_r, _r['Q'], L.lang, _fn)}"),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met de functie van de course bepalen de bins of MountainCar überhaupt geleerd wordt. Met 12 bins voor elk getal bereiken alle 10 runs de vlag, de eerste keer tussen episode 190 en 429. Met 20 bins doen alle 10 runs het ook, maar later, tussen episode 404 en 694: 400 rijen in plaats van 144 laten meer over om te proberen tot de eerste vlag. Met 8 bins bereiken slechts 3 van de 10 runs ooit de vlag, allemaal na episode 1.900, en met 4 bins slechts 1. Tot de eerste vlag komt elke Q van een run uit alleen −1's. Tot dan trekt het begin de agent vooruit: met elke Q op 0 en elke stap −1 lijkt een action die geprobeerd is slechter dan een action die nog niet geprobeerd is, dus blijft de agent andere duwen proberen, zoals in *Eén interface voor elke environment: Gymnasium*. Begin elke Q op −100 in plaats van 0, en met 12 bins bereikt geen van 3 runs de vlag in 2.000 episodes.

    Met 12 bins voor de hoek en de hoeksnelheid gaat CartPole van 23 stappen per episode met willekeurige duwen naar tussen 128 en 292 aan het eind van de 10 runs, als gemiddelde van de laatste 100 episodes. Met alleen de grootste Q, zonder willekeurige actions, blijft de paal gemiddeld 49 tot 309 stappen staan: ver onder de 500 stappen van één regel met de hoek en de hoeksnelheid. In de volgende les leert een library CartPole met een neural network in plaats van een tabel.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sparse():
    mo.md(r"""
    ### Oefening 4. Welke benchmark?

    Een team heeft een nieuwe manier van exploration geschreven, bedoeld voor taken waarin de reward niets zegt tot het doel bereikt is. Het team wil op één klassieke benchmark laten zien dat de nieuwe manier helpt.
    """)
    return


@app.cell(hide_code=True)
def choice_sparse_cell():
    choice_sparse = L.choice(
        "sparse",
        question="Welke benchmark uit deze les test precies dat?",
        options={
            "cart": "CartPole: +1 voor elke stap dat de paal blijft staan",
            "right": "MountainCar: −1 voor elke stap tot de vlag",
            "atari": "Een Atari-spel, zoals Breakout: punten voor elk blokje dat de bal raakt",
            "hopper": "Hopper van MuJoCo: een reward voor elk stukje vooruit, en voor rechtop blijven",
        },
        correct="right",
        why={
            "right": "Elke stap van MountainCar geeft dezelfde −1, wat de auto ook doet, dus zegt de reward tot de vlag niets over welke duw beter was: willekeurige duwen komen er nooit. CartPole, de Atari-spellen en Hopper geven een reward die bijna elke stap verandert, en testen andere dingen: balanceren, pixels en actions die reële getallen zijn.",
            "cart": "In CartPole geeft elke stap dat de paal blijft staan +1, en een val beëindigt de episode meteen: zwijgt deze reward tot een doel?",
            "atari": "Een Atari-spel deelt het hele spel door punten uit, en test vooral leren van pixels: wacht dit spel met de reward tot een doel?",
            "hopper": "Hopper beloont elk stukje vooruitgang, en test het bewegen van een lichaam met actions die reële getallen zijn: zwijgt deze reward tot een doel?",
        },
    )
    choice_sparse
    return (choice_sparse,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Eén interface voor elke environment: Gymnasium](/course/open/rl-gymnasium?lang=nl) was elke observation een geheel getal, een rij van de tabel. Hier zijn de observations reële getallen, en maken bins er rijen van. CartPole en MountainCar testen elk iets anders: balanceren, en een reward die niets zegt tot het doel. De volgende les, *Een RL-library gebruiken: Stable-Baselines3*, traint een agent met een neural network op CartPole, met een library van geteste algoritmes, en beoordeelt de agent over meerdere seeds en veel episodes om te testen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_sparse, result_bin, result_policy, result_row):
    L.wrap_up(result_policy, result_bin, result_row, choice_sparse)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
