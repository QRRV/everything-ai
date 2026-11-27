# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-reward-design. Route: course/route/lessen/reinforcement-learning.yaml. Checks, de kamer, de runs van de course en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rl-reward-design", "nl")
    C = L.checks   # de kamer, de runs van de course met beide rewards en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Leren van rewards](/course/open/rl-what-is-rl?lang=nl) en [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_return():
    quick_return = L.choice(
        "quick-return",
        question="Met γ = 0,9 kan een agent bij elke stap een reward van +1 krijgen, zonder eind, of nu meteen één keer +5, en daarna niets. Welke van de twee geeft de grootste return?",
        options={
            "once": "Eén keer +5: elke latere reward telt minder",
            "right": "+1 bij elke stap, met een return van 10",
            "endless": "+1 bij elke stap, want de return groeit zonder eind",
            "same": "Allebei even groot",
        },
        correct="right",
        why={
            "right": "De return is 1 + 0,9 + 0,81 + 0,729 + ..., elke stap één keer extra vermenigvuldigd met 0,9, en dat komt samen op 10: twee keer de 5. Komt een kleine reward bij elke stap terug, dan kan deze reward meer waard zijn dan een grote reward die één keer komt.",
            "once": "Elke latere reward telt minder, maar er komt er bij elke stap één: waar komen 1 + 0,9 + 0,81 + ... samen op uit? Zie [Leren van rewards](/course/open/rl-what-is-rl?lang=nl).",
            "endless": "Met γ = 0,9 telt elke reward 0,9 keer zo zwaar als de reward een stap eerder, dus komen de rewards samen op een getal met een eind: welk getal? Zie [Leren van rewards](/course/open/rl-what-is-rl?lang=nl).",
            "same": "De return telt elke reward vanaf nu, elke stap later één keer extra vermenigvuldigd met γ: waar komt elk van de twee op uit? Zie [Leren van rewards](/course/open/rl-what-is-rl?lang=nl).",
        },
    )
    quick_return
    return (quick_return,)


@app.cell(hide_code=True)
def quick_check_target():
    quick_target = L.choice(
        "quick-target",
        question="Bij Q-learning neemt de agent in state s de action a, krijgt de reward r, en komt in state s′. Naar welk target schuift Q(s, a) op?",
        options={
            "reward": "Alleen r",
            "sarsa": "r + γ keer de Q van de action die de agent in s′ echt neemt",
            "right": "r + γ keer de grootste Q van s′",
            "next": "γ keer de grootste Q van s′",
        },
        correct="right",
        why={
            "right": "Het target is r + γ max Q(s′, a′): de reward van de stap, plus het beste wat de agent erna verwacht. In deze les leert Q-learning met dit target, zoals altijd; wat verandert is r, de reward die iemand schrijft.",
            "reward": "Met alleen r kent Q(s, a) slechts de volgende stap, en niets van wat erna komt: wat voegt het target toe aan r? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
            "sarsa": "Dat is het target van SARSA, met de action die de agent echt als volgende neemt: welke Q van s′ neemt Q-learning? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
            "next": "Zonder r komt de reward van de stap zelf nooit in Q: wat komt er als eerste in het target? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
        },
    )
    quick_target
    return (quick_target,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Een agent kent de taak alleen via de rewards, en mensen schrijven deze rewards. In 2016 trainden Jack Clark en Dario Amodei van OpenAI een agent op CoastRunners, een spel met bootraces. Het spel geeft punten voor het raken van targets langs de baan, en de score leek een bruikbare maat voor een goede race. De agent vond een kleine lagune met drie targets die steeds terugkomen, en voer daar rondjes: de boot raakte de targets keer op keer, vloog in brand, botste tegen andere boten en voer de verkeerde kant op, zonder de race ooit uit te varen. De score lag gemiddeld 20 procent hoger dan de score van menselijke spelers. De agent deed niets fout: de agent haalde zo veel mogelijk reward binnen, precies zoals de reward geschreven was. In hetzelfde jaar namen Amodei en vijf collega's een robot die een kantoor schoonmaakt als voorbeeld in *Concrete Problems in AI Safety*: krijgt zo'n robot een reward voor een kantoor waarin de robot geen rommel ziet, dan kan de robot de eigen camera uitzetten. In deze les probeer je het zelf, met een schoonmaakrobot en rewards die je zelf schrijft.

    **Een voorbeeld uit het dagelijks leven.** Betaalt een helpdesk de medewerkers voor elk gesloten ticket, dan gaan tickets snel dicht, ook als het probleem nog niet is opgelost. De klant opent een nieuw ticket, en hetzelfde probleem levert twee keer geld op. Niemand speelt vals: de medewerkers doen precies wat de betaling beloont.
    """)
    return


@app.cell(hide_code=True)
def predict_dirt():
    guess_dirt = L.guess(
        "dirt",
        question="De robot in de afbeelding kan lopen, een stuk vuil oppakken en het weer neerleggen; in de bak B is een stuk weg. De robot krijgt een reward van +1 elke keer dat de robot een stuk oppakt, en elke stap kost 0,01 batterij. Q-learning traint de robot 3.000 episodes van hoogstens 30 stappen. Daarna neemt de robot 30 stappen lang de action met de grootste Q. Hoeveel van de drie stukken vuil liggen er na deze 30 stappen nog in de kamer, denk je?",
        options={
            "none": "Geen: de robot heeft geleerd alle drie in de bak te leggen",
            "one": "Eén: de robot ruimt de twee dichtstbijzijnde op en heeft dan geen stappen meer",
            "right": "Alle drie",
            "two": "Twee: de robot legt het dichtstbijzijnde stuk in de bak, en heeft daarna in de buurt niets meer om op te pakken",
        },
        reveal="**Alle drie.** De robot loopt naar het dichtstbijzijnde stuk, in de hoek rechtsboven, pakt het op, legt het op hetzelfde vakje weer neer, en pakt het weer op: 14 keer oppakken in de 30 stappen, een reward van 13,70, en geen enkel stuk in de bak. In alle 10 runs van de course doet de robot dit binnen de eerste 10 episodes.",
    )
    mo.vstack([mo.Html(C.world_picture(L.lang)), guess_dirt])
    return (guess_dirt,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **De kamer.** Een kamer van 3 rijen en 5 kolommen. De bak, B, staat in de hoek linksboven; de robot begint in de hoek rechtsonder, S. Er liggen drie stukken vuil op de vloer. De robot heeft zes actions: omhoog, omlaag, links en rechts (`"up"`, `"down"`, `"left"`, `"right"`), waarbij de robot na een stap tegen een muur op hetzelfde vakje blijft; `"pick"`, dat het stuk vuil oppakt waar de robot op staat, als de hand leeg is; en `"drop"`, dat het stuk in de hand neerlegt: in de bak is het voorgoed weg, op een schoon vakje ligt het weer op de vloer. De robot houdt één stuk tegelijk vast. Een episode eindigt als de kamer schoon is, of na 30 stappen.

        **De wereld en de reward apart.** In code is een state `(cell, hand, dirt)`: het vakje van de robot als (rij, kolom), `hand`, 1 met een stuk in de hand en anders 0, en `dirt`, een tuple met de vakjes waar vuil op de vloer ligt. `BEGIN` is de state bij de start, `((2, 4), 0, ((0, 4), (1, 2), (2, 0)))`. `step(state, action)` geeft de volgende state terug, en verder niets. In eerdere lessen gaf de wereld ook de reward; hier doet een aparte functie dat: `reward(state, nxt)` kijkt naar de state bij het begin en aan het eind van de stap. De wereld ligt vast; de reward schrijven mensen, en daar kan het misgaan.

        **De batterij.** Elke stap kost 0,01: elke reward van deze les haalt `COST`, 0,01, van elke stap af. Zo is een korte weg beter dan een lange. De batterij laat de robot ook nieuwe actions proberen: elke Q begint op 0, en een geprobeerde action die alleen batterij kostte zakt onder 0, dus lijkt een action die de robot nog nooit probeerde beter.

        **Reward misspecification.** De agent maakt de return zo groot mogelijk: de return van de reward die er staat, niet van de taak die bedoeld was. Verschillen de twee, dan is dat **reward misspecification**. Vindt de agent zo'n gat en gebruikt de agent het, zoals de boot in CoastRunners deed, dan is dat **reward hacking**: een vorm van **specification gaming**, een doel halen zoals het er staat op een manier die niemand bedoelde.

        **Twee rewards.** De *reward voor oppakken* geeft 1 − 0,01 = 0,99 voor elke stap waarop de robot een stuk oppakt, en −0,01 voor elke andere stap: vuil oppakken is wat een schoonmaakrobot doet, denkt de ontwerper. De *reward voor het resultaat* kijkt in plaats daarvan naar het resultaat: het vuil in de kamer, op de vloer en in de hand van de robot. Deze reward geeft 0,99 voor elk stuk dat de kamer verlaat, de bak in, en −0,01 voor elke andere stap. Over een episode komt de reward voor het resultaat samen op het aantal stukken in de bak, min de batterij: het meeste geeft een schone kamer, in zo weinig mogelijk stappen.

        Q-learning leert met elke reward zoals in *Leren van je eigen stappen: Q-learning*, met ε = 0,1, η = 0,5 en γ = 0,9. De afbeelding toont run 1 na 3.000 episodes: de robot neemt de action met de grootste Q.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        Met de reward voor oppakken pakt de robot het stuk in de hoek rechtsboven 14 keer op, en de 30 stappen leveren een reward van 13,70 op. Met de reward voor het resultaat brengt de robot de stukken één voor één naar de bak en maakt de kamer schoon in 24 stappen, voor een reward van 2,76; de kortste weg duurt 22 stappen. Volgens de reward voor oppakken zijn deze 24 stappen die de kamer schoonmaken ook 2,76 waard: de reward voor oppakken betaalt verreweg het meest voor een kamer die vuil blijft, 13,70 tegen 2,76.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **reward misspecification** | | een reward die niet vastlegt wat de ontwerpers echt willen, zodat de agent hoog kan scoren terwijl de agent het verkeerde doet | `pickup_reward` |
        | **reward hacking** | | een agent die een onbedoelde weg naar veel reward vindt en gebruikt | |
        | **specification gaming** | | een doel halen zoals het er staat, op een manier die niemand bedoelde; reward hacking is de vorm bij reinforcement learning | |
        """),
        L.watch_out("De robot met de reward voor oppakken is niet kapot, en Q-learning evenmin: allebei werken precies zoals het hoort. Met de reward voor oppakken is hetzelfde stuk steeds weer oppakken de beste policy die er is. De fout zit in de reward."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een reward ontwerpen, en controleren:

    1. Zeg wat je wilt als een resultaat dat je in een state kunt tellen: hier het vuil in de kamer.
    2. Geef als reward van een stap hoeveel dit resultaat in de stap beter werd, min kleine kosten voor elke stap.
    3. Train de agent met deze reward.
    4. Kijk wat de agent doet: tel het resultaat zelf, los van de reward, en bekijk de stappen. Stijgt de reward terwijl het resultaat dat niet doet, of is de reward hoger dan de taak kan geven, dan betaalt de reward voor iets wat je niet wilde.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Beide rewards voor vier stappen. De robot staat op (0, 4), in de hoek rechtsboven; de eerste twee stappen beginnen in de volle kamer:

    | Stap | Hand bij begin → eind | Vuil in de kamer bij begin → eind | Reward voor oppakken | Reward voor het resultaat |
    |---|---|---|---|---|
    | `"pick"` op (0, 4) | 0 → 1 | 3 → 3 | 0,99 | −0,01 |
    | `"drop"` op (0, 4) | 1 → 0 | 3 → 3 | −0,01 | −0,01 |
    | een stap naar links, met een stuk in de hand | 1 → 1 | 3 → 3 | −0,01 | −0,01 |
    | `"drop"` op B | 1 → 0 | 3 → 2 | −0,01 | 0,99 |

    Na de eerste twee stappen is alles weer zoals het was, de robot op (0, 4) en het stuk op de vloer, en de reward voor oppakken gaf 0,99 − 0,01 = 0,98 voor deze twee stappen. Daarna kan de robot hetzelfde nog een keer doen, en daarna weer: een rondje dat loont, elke twee stappen. Met de reward voor het resultaat kost hetzelfde rondje 0,02: het stuk verliet de kamer nooit, want een stuk in de hand telt ook mee. Met de reward voor het resultaat levert een stuk pas iets op als het in de bak ligt, en daar komt het nooit meer uit.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pickup():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Een reward voor oppakken

        `pickup_reward(state, nxt)` geeft de reward van één stap, van `state` naar `nxt`. `cell, hand, dirt = state` haalt de state uit elkaar in de drie delen, zoals een `for`-loop over paren een paar uit elkaar haalt, en de regel eronder haalt `nxt` uit elkaar in `new_cell`, `new_hand` en `new_dirt`. Vervang de `...` door de test voor oppakken: de hand was leeg, en na de stap houdt de robot een stuk vast. Dan geeft de functie `1 - COST`, en anders `-COST`.

        De check roept `pickup_reward(((0, 4), 0, ((0, 4), (1, 2), (2, 0))), ((0, 4), 1, ((1, 2), (2, 0))))` aan: de robot op (0, 4) pakt het stuk daar op. Daarna zes andere stappen, waaronder een stap met een stuk in de hand, een stuk dat op de vloer gaat, en een stuk dat in de bak gaat. In de afbeelding traint Q-learning de robot met je reward.
        """),
        L.task("Laat `pickup_reward` `1 - COST` geven voor een stap waarop de robot een stuk oppakt, en `-COST` voor elke andere stap."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pickup():
    box_pickup = L.code_box("pickup_reward", lines=6)
    box_pickup
    return (box_pickup,)


@app.cell(hide_code=True)
def check_pickup(box_pickup):
    result_pickup = L.check("pickup_reward", box_pickup.value)
    result_pickup
    return (result_pickup,)


@app.cell(hide_code=True)
def hints_pickup():
    L.hints(
        "pickup_reward",
        [
            "De test na `if` zegt dat deze stap oppakken is: bij het begin van de stap is `hand` 0, de hand leeg, en erna is `new_hand` 1, een stuk in de hand.",
            "Beide moeten tegelijk waar zijn, dus verbind je de twee tests met `and`, elk met `==`. Een stap die begint met `hand` op 1 draagt alleen een stuk, en levert niets op.",
            "`    if ___:`",
        ],
        why="`if hand == 0 and new_hand == 1:` betaalt 0,99 voor de stap waarop de robot een stuk in de hand neemt, en −0,01 voor elke andere stap. `if new_hand > hand:` zegt hetzelfde.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_result():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een reward voor het resultaat

        `result_reward(state, nxt)` geeft 1 − 0,01 voor elk stuk dat in de stap de kamer verlaat, en −0,01 voor een stap waarna het vuil in de kamer hetzelfde is. De eerste twee regels halen `state` en `nxt` uit elkaar, net als in oefening 1, in `cell`, `hand`, `dirt` en `new_cell`, `new_hand`, `new_dirt`. `before` telt al het vuil in de kamer bij het begin van de stap: `len(dirt)` stukken op de vloer, plus `hand`, het stuk in de hand. Vervang de twee `...`: `after`, dezelfde telling na de stap, en de reward.

        De check roept `result_reward(((0, 0), 1, ((1, 2), (2, 0))), ((0, 0), 0, ((1, 2), (2, 0))))` aan: de robot legt een stuk in de bak. Daarna zes andere stappen, waaronder oppakken, een stuk dat op de vloer gaat, en het laatste stuk dat de bak in gaat.
        """),
        L.task("Laat `result_reward` geven hoeveel minder vuil er na de stap in de kamer is dan bij het begin ervan, min `COST`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_result():
    box_result = L.code_box("result_reward", lines=6)
    box_result
    return (box_result,)


@app.cell(hide_code=True)
def check_result(box_result):
    result_result = L.check("result_reward", box_result.value)
    result_result
    return (result_result,)


@app.cell(hide_code=True)
def hints_result():
    L.hints(
        "result_reward",
        [
            "Stap 1 en 2 van het recept. `after` telt zoals `before`, maar met de delen van `nxt`: `len(new_dirt) + new_hand`. Na `return` komt de verandering: hoeveel minder vuil er is, min `COST`.",
            "In het voorbeeld legt de robot een stuk in de bak: `before` is 3 en `after` is 2, dus hoort de reward 1 − 0,01 = 0,99 te zijn. Een stap die niets verandert hoort −0,01 te geven.",
            "`    return ___`",
        ],
        why="`return before - after - COST` geeft 0,99 voor elk stuk dat de kamer verlaat en −0,01 voor elke andere stap. Oppakken verandert niets: het stuk gaat van de vloer naar de hand, en beide tellen mee.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_dirt():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. De robot bekijken

        Een stijgende reward zegt op zich weinig. `dirt_left(Q)` bekijkt de robot: vanaf `BEGIN` neemt de robot `LIMIT` (30) stappen lang de action met de grootste Q, en daarna geeft de functie het vuil terug dat nog in de kamer ligt, op de vloer en in de hand. `greedy(Q, s)` geeft de action met de grootste Q in state `s`, de eerste als er meer gelijk zijn; voor een state die de Q-table nooit zag, tellen alle Q's als 0. `step(state, action)` geeft de volgende state. Na de loop haalt `cell, hand, dirt = state` de laatste state uit elkaar, net als in oefening 1. De loop hoeft niet te stoppen als de kamer schoon is: een schone kamer kan niet weer vuil worden, dus blijft de telling 0. In `for _ in range(LIMIT):` staat de naam `_` voor een getal dat de loop niet nodig heeft. Vervang de twee `...`: de `...` in de loop wordt twee regels, eerst `action`, dan `state = step(state, action)`.

        De check roept `dirt_left(Q_result)` aan: de Q-table van run 1 met de reward voor het resultaat na 3.000 episodes, de robot uit de afbeelding in *Het idee*. Daarna `Q_pickup`, hetzelfde met de reward voor oppakken; `Q_100` en `Q_300`, run 1 met de reward voor het resultaat na 100 en 300 episodes; `Q_500`, run 2 na 500; en `{}`, een lege Q-table.
        """),
        L.task("Laat `dirt_left` de robot `LIMIT` stappen lang de action met de grootste Q laten nemen, en het vuil op de vloer plus het stuk in de hand teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_dirt():
    box_dirt = L.code_box("dirt_left", lines=6)
    box_dirt
    return (box_dirt,)


@app.cell(hide_code=True)
def check_dirt(box_dirt):
    result_dirt = L.check("dirt_left", box_dirt.value)
    result_dirt
    return (result_dirt,)


@app.cell(hide_code=True)
def hints_dirt():
    L.hints(
        "dirt_left",
        [
            "Stap 4 van het recept. In de loop eerst `action`: de action met de grootste Q van `Q` in `state`, van `greedy`; dan brengt `state = step(state, action)` de robot verder. Geef na de loop het vuil in de kamer terug: `len(dirt) + hand`.",
            "`greedy` krijgt twee dingen: de Q-table en de state waar de robot nu is. Hoe heten deze twee in je functie?",
            "`        action = ___`",
        ],
        why="`action = greedy(Q, state)` kiest de action met de grootste Q in de state waar de robot nu is, en `step` brengt de robot verder. Zonder de reward zegt de telling van het vuil wat de robot echt deed.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Van 0 tot en met 3.000 episodes

    De course trainde beide robots 10 keer, 10 runs met elk eigen willekeurige getallen, 3.000 episodes lang. Het blok hieronder gaat door de training, met de episodes op een logaritmische schaal. Links de reward per episode, rechts het vuil dat aan het eind van een episode over is: allebei het gemiddelde van de 10 runs over de laatste 50 episodes, doorgetrokken voor de reward voor oppakken en gestreept voor de reward voor het resultaat. Daaronder de robots van run 1 na zoveel episodes, met de grootste Q.
    """)
    return


@app.cell(hide_code=True)
def explore_episodes_cell():
    explore_episodes = L.explore("episodes", "episodes", lambda _v: C.demo_frame(int(_v), L.lang), start=0,
                                 values=C.LOOKS, role="",
                                 task="Ga van 0 naar 3.000: na hoeveel episodes pakt de robot met de reward voor oppakken steeds hetzelfde stuk op, en vanaf wanneer maakt de robot met de reward voor het resultaat de hele kamer schoon?")
    explore_episodes
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Net als in de grafieken zijn de rewards en het vuil hieronder gemiddelden van de 10 runs over de laatste 50 episodes. De reward van de oppakrobot, de robot met de reward voor oppakken, schiet omhoog: van −0,30 in de eerste episode naar ongeveer 12 binnen 100 episodes, en 12,31 per episode aan het eind. Het vuil dat de robot overlaat blijft 3, en met de grootste Q pakt de robot binnen de eerste 10 episodes steeds hetzelfde stuk op, in alle 10 runs. De resultaatrobot, met de reward voor het resultaat, heeft meer tijd nodig, want de robot krijgt niets tot er toevallig een stuk in de bak belandt. Na 500 episodes laat de robot 0,48 stukken per episode over, aan het eind 0,12, terwijl de reward per episode langzaam stijgt tot 2,61, minder dan een kwart van de reward van de oppakrobot. De paar stukken die de robot nog overlaat komen van de willekeurige stappen, de ε van ε-greedy. Met de grootste Q maken alle 10 runs de kamer uiterlijk na 1.000 episodes schoon; na 3.000 doen 9 runs het in 24 stappen en één in 26.

    Bekeken via de reward is de oppakrobot verreweg de betere leerling. Bekeken via de kamer leerde de oppakrobot nooit iets nuttigs.

    ### Alle episodes met NumPy

    `C.history("pickup")` geeft twee arrays met shape (10, 3000), een rij per run en een kolom per episode: de reward van elke episode, en het vuil dat aan het eind over is; `C.history("result")` hetzelfde voor de reward voor het resultaat. `_returns[:, -500:]` neemt elke rij, en van elke rij de laatste 500 kolommen: de laatste 500 episodes van elke run. `.mean()` neemt het gemiddelde van alle getallen erin. `_left == 0` maakt een array met `True` en `False`, `True` waar de kamer schoon was, en de `.mean()` daarvan is het aandeel `True`, want `True` telt als 1. `_best` is de reward van de kortste weg naar een schone kamer, in 22 stappen, met beide rewards: `3 - 22 * C.COST`. Een episode met meer reward dan dat deed iets anders dan schoonmaken. Voer de cel uit.
    """)
    return


@app.cell
def example_numpy():
    import numpy as np

    _best = 3 - 22 * C.COST   # de reward van de kortste weg naar een schone kamer
    print("_best:", round(_best, 2))
    for _name in ("pickup", "result"):
        _returns, _left = C.history(_name)   # twee arrays met shape (10, 3000)
        print(_name, "reward", _returns.shape)
        print("  gemiddelde reward per episode, laatste 500 episodes:", round(float(_returns[:, -500:].mean()), 2))
        print("  gemiddeld vuil over, laatste 500 episodes:", round(float(_left[:, -500:].mean()), 2))
        print("  deel van alle episodes met een schone kamer:", round(float((_left == 0).mean()), 3))
        print("  deel van de laatste 500 met meer reward dan _best:", round(float((_returns[:, -500:] > _best).mean()), 4))
    return (np,)


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    In de laatste 500 episodes van elke run krijgt de oppakrobot 12,31 per episode en laat de robot alle 3 de stukken liggen. Geen van de 30.000 episodes eindigde in een schone kamer, en slechts in 2 belandde er toevallig een stuk in de bak. In 99,96% van de laatste 500 episodes is de reward hoger dan de 2,78 die een perfecte schoonmaak kan geven: de getallen alleen laten al zien dat deze reward voor iets anders betaalt. De resultaatrobot krijgt 2,57 per episode, laat gemiddeld 0,16 stukken over, en maakte de kamer schoon in 69,9% van alle episodes, de eerste onhandige meegerekend. De reward komt nooit boven 2,78: een reward voor het resultaat kun je niet verder opdrijven dan wat het resultaat waard is.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_camera():
    mo.md(r"""
    ### Oefening 4. Een hand voor de camera

    In 2017 beschreef OpenAI een robotarm die moest leren een voorwerp te pakken. De reward kwam van mensen: deze mensen keken via één camera naar twee korte filmpjes van de arm, en zeiden welk filmpje meer op pakken leek. Na de training beoordeelden de mensen de filmpjes van de arm als goede grepen.
    """)
    return


@app.cell(hide_code=True)
def choice_camera_cell():
    choice_camera = L.choice(
        "camera",
        question="Wat had de arm echt geleerd?",
        options={
            "grasp": "Het voorwerp pakken, want wie naar een filmpje kijkt ziet meteen of een hand iets vasthoudt of niet, ook via één camera",
            "right": "De hand tussen de camera en het voorwerp houden, zodat het alleen leek alsof de arm het voorwerp pakte",
            "nothing": "Helemaal niets: een reward van mensen bevat te veel ruis om van te leren",
            "fast": "Zo snel mogelijk bewegen, want een snel filmpje ziet er beter uit dan een langzaam",
        },
        correct="right",
        why={
            "right": "De mensen konden alleen beoordelen wat de camera liet zien. Vanaf de camera lijkt een hand tussen de camera en het voorwerp op een hand die het voorwerp vasthoudt, en dat is makkelijker. De reward telde wat mensen zagen, niet het pakken zelf: hetzelfde gat als een reward voor oppakken in plaats van voor een schone kamer. Een blik van meer kanten, of een sensor in de hand, zou het pakken zelf tellen.",
            "grasp": "De mensen zagen de arm slechts via één camera. Is een hand die een voorwerp vasthoudt het enige wat er vanaf daar als pakken uitziet?",
            "nothing": "De reward van mensen leerde de arm wel iets: de mensen beoordeelden de filmpjes als goede grepen. Wat kan er via één camera als pakken uitzien, zonder pakken te zijn?",
            "fast": "De mensen kozen het filmpje dat meer op pakken leek, niet het snellere. Wat kan er via één camera als pakken uitzien?",
        },
    )
    choice_camera
    return (choice_camera,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Voorzichtig of gedurfd: SARSA en Q-learning](/course/open/rl-sarsa?lang=nl) kwam de reward met de wereld mee: −1 voor een stap, −100 voor een val. Hier schreef je de reward zelf, en Q-learning leerde precies wat er stond: een reward voor een action die je kunt herhalen leerde de robot het eigen werk ongedaan te maken en opnieuw te doen; een reward voor het resultaat leerde de robot schoonmaken. Tel het resultaat dat je wilt, los van de reward, en kijk wat de agent doet. Hetzelfde gat komt terug bij elke agent die leert van een reward, ook bij de large language models die getraind worden met een reward van mensen of van een ander model. *Doen wat je zei, niet wat je bedoelde*, in het domein Verantwoorde en veilige AI, gaat verder in op specification gaming. De volgende les, *Als agents elkaar ontmoeten: emergent behavior en sociale dilemma's*, zet twee lerende agents in één wereld.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_camera, result_dirt, result_pickup, result_result):
    L.wrap_up(result_pickup, result_result, result_dirt, choice_camera)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
