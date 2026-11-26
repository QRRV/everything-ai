# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-monte-carlo-prediction. Route: course/route/lessen/reinforcement-learning.yaml. Checks, de kamer, de wandelingen, returns en schattingen van de course, en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rl-monte-carlo-prediction", "nl")
    C = L.checks   # de kamer, de wandelingen, returns en schattingen van de course, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Wat is een policy waard? Policy evaluation](/course/open/rl-policy-evaluation?lang=nl) en [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_value():
    quick_value = L.choice(
        "quick-value",
        question="Op ROOM, het grid van de robot, heeft een tegel naast het laadstation onder de random policy de waarde −24, met een reward van −1 per stap en $\\gamma$ = 1. Wat zegt deze −24?",
        options={
            "most": "Wandelingen vanaf de tegel duren hoogstens 24 stappen",
            "right": "De wandelingen vanaf de tegel duren gemiddeld 24 stappen",
            "every": "Elke wandeling vanaf de tegel duurt precies 24 stappen, omdat de policy niet verandert",
            "count": "24 van elke 100 wandelingen vanaf de tegel eindigen op het laadstation",
        },
        correct="right",
        why={
            "right": "Met −1 per stap en $\\gamma$ = 1 is de return van een wandeling min het aantal stappen, en de waarde is de return die je kunt verwachten: het gemiddelde. Eén wandeling kan veel korter of langer zijn. Deze les vindt dezelfde −24 door het gemiddelde van wandelingen te nemen.",
            "most": "De random robot kan lang rondzwerven voordat de robot bij het laadstation komt: is de waarde een grens aan het aantal stappen, of de return die je kunt verwachten? Zie [Wat is een policy waard? Policy evaluation](/course/open/rl-policy-evaluation?lang=nl).",
            "every": "De random policy kiest elke stap willekeurig, dus twee wandelingen hoeven niet gelijk te zijn: is −24 het aantal stappen van één wandeling, of van alle wandelingen samen? Zie [Wat is een policy waard? Policy evaluation](/course/open/rl-policy-evaluation?lang=nl).",
            "count": "Elke wandeling van de random robot komt vroeg of laat bij het laadstation: wat telt een reward van −1 per stap? Zie [Wat is een policy waard? Policy evaluation](/course/open/rl-policy-evaluation?lang=nl).",
        },
    )
    quick_value
    return (quick_value,)


@app.cell(hide_code=True)
def quick_check_step():
    quick_step = L.choice(
        "quick-step",
        question="Een agent die Q-learning doet in GRID ziet de tabel `T` nooit. Wat weet de agent na één stap?",
        options={
            "table": "Elke uitkomst van de action, elk met de kans uit `T`",
            "reward": "Alleen de reward, niet waar de agent belandde",
            "right": "De state, de action, de reward en waar de agent belandde",
            "values": "De waarde van elke state, vooraf uitgerekend met value iteration",
        },
        correct="right",
        why={
            "right": "Meer ziet de agent niet van de environment: één uitkomst, de uitkomst die gebeurde. Q-learning leert van zulke stappen zonder `T`, en de methode van deze les leert van hele episodes ervan.",
            "table": "De agent ziet `T` nooit: krijgt de agent de kansen van alle uitkomsten, of alleen de uitkomst die gebeurde? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
            "reward": "Het target van Q-learning gebruikt de grootste $Q$ waar de agent belandde: hoe zou de agent deze $Q$ vinden zonder te weten waar de agent belandde? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
            "values": "Value iteration heeft `T` nodig, en de agent heeft `T` niet: waar komen de $Q$-waarden van de agent dan vandaan? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
        },
    )
    quick_step
    return (quick_step,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Policy evaluation rekende uit wat een policy waard is met de tabel `T`: elke kans en elke reward. Vaak heeft niemand deze tabel. Een robotstofzuiger in een onbekend huis, een kaartspel waarin je de kaarten niet ziet, een programma dat tegen mensen speelt: geen van alle kan een sweep over een tabel doen. Wat wel kan: spelen, steeds opnieuw, en tellen wat elk potje opleverde. Monte Carlo prediction maakt van zulke hele potjes waarden, met niets dan gemiddelden, en de volgende lessen bouwen erop voort om de policy beter te maken en sneller te leren.

    **Een voorbeeld uit het dagelijks leven.** Hoe lang doe je gemiddeld over je reis naar je werk? Niemand rekent dat uit met de tijden van elk verkeerslicht. Je klokt de reis een paar weken en neemt het gemiddelde. Eén reis zegt weinig, want een file kan de reistijd verdubbelen; na veel reizen verandert het gemiddelde nauwelijks nog.
    """)
    return


@app.cell(hide_code=True)
def predict_walks():
    guess_walks = L.guess(
        "walks",
        question="Vanuit de hoek heeft de random robot van ROOM gemiddeld 36 stappen nodig om bij het laadstation te komen, zoals policy evaluation uitrekende. Zonder de tabel kun je de robot ook vaak vanuit de hoek laten lopen en het gemiddelde van de aantallen stappen nemen. Na hoeveel wandelingen zit dat gemiddelde meestal nog slechts zo'n 1 stap ernaast?",
        options={
            "ten": "Zo'n 10: de wandelingen van één robot in één kamer lijken sterk op elkaar",
            "thirty": "Zo'n 36, één wandeling voor elke stap die de robot gemiddeld nodig heeft",
            "right": "Zo'n 1.000",
            "hundred": "Zo'n 100",
        },
        reveal="**Zo'n 1.000 wandelingen.** De wandelingen verschillen sterk: zo'n 2 op de 100 nemen de kortste weg van 4 stappen, zo'n 1 op de 100 duurt meer dan 150 stappen, en de standaardafwijking is zo'n 32 stappen. Het gemiddelde van $n$ wandelingen heeft minder spreiding, zo'n 32/√n: zo'n 10 stappen bij 10 wandelingen, zo'n 3 bij 100, en zo'n 1 bij 1.000. *Het idee* laat zien waar deze √n vandaan komt.",
    )
    guess_walks
    return (guess_walks,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Monte Carlo prediction** schat de waarden $V_\pi$ van een policy zonder de tabel `T`, uit episodes die de agent met $\pi$ speelt, zoals de wandelingen van de robot. De waarde van een state is de return die je ervandaan kunt verwachten. Elke keer dat een episode in deze state is, geven de rewards erna één return die echt gebeurd is; het gemiddelde van veel zulke returns komt dicht bij de return die je kunt verwachten. Een getal schatten door het gemiddelde te nemen van veel willekeurige trekkingen is de **Monte Carlo-methode**, genoemd naar het casino van Monte Carlo. Eerst de returns van een episode: met $r_{t+1}$ de reward direct na stap $t$, zoals in [Leren van rewards](/course/open/rl-what-is-rl?lang=nl),
        """),
        L.formula(r"[[loss:G_t]] = [[data:r_{t+1}]] + [[hyper:\gamma]] \, [[loss:G_{t+1}]]"),
        mo.md(r"""
        De return vanaf stap $t$ is de reward direct erna plus $\gamma$ keer de return vanaf de volgende stap, en na de laatste stap is de return 0. De returns van een episode reken je dus het snelst achteruit uit, vanaf het eind: elke stap kost één optelling en één vermenigvuldiging. In de wandelingen van de robot, met −1 per stap en $\gamma$ = 1, is de return vanaf een stap min het aantal stappen dat nog komt. Dan de schatting van een state $s$, uit de returns $G^{(1)}, \ldots, G^{(n)}$ die op de bezoeken volgden:
        """),
        L.formula(r"[[pred:V_\pi(s)]] \approx \frac{[[loss:G^{(1)}]] + [[loss:G^{(2)}]] + \dots + [[loss:G^{(n)}]]}{n}"),
        mo.md(r"""
        Een episode kan terugkomen in een state: de robot in een hoek botst tegen de rand en blijft staan. **First-visit Monte Carlo** neemt uit elke episode alleen de return na het eerste bezoek aan een state; **every-visit Monte Carlo** neemt de return na elk bezoek. Over veel episodes komen de twee methoden allebei dicht bij $V_\pi$. Bij first visit komt elke return uit een andere episode, dus hangen de returns niet van elkaar af, en is het gemiddelde een gewoon gemiddelde van willekeurige trekkingen.

        Hoeveel episodes kost dat? De returns liggen ver uiteen: vanuit de hoek heeft de robot gemiddeld 36 stappen nodig, met een standaardafwijking van zo'n 32. Een gemiddelde van $n$ returns heeft een $\sqrt{n}$ keer zo kleine spreiding als één return: vier keer zoveel episodes halveren de spreiding. Waarom $\sqrt{n}$? De returns van verschillende episodes zijn onafhankelijk, en bij onafhankelijke getallen tellen de varianties op: $n$ returns samen hebben $n$ keer de variantie van één return. Het gemiddelde deelt door $n$, en dat deelt de variantie door $n^2$, dus het gemiddelde heeft $1/n$ van de variantie van één return, en een $\sqrt{n}$ keer zo kleine standaardafwijking.

        Elke stap krijgt de hele return erna, wat de stap zelf ook deed. Komt de robot dicht bij het laadstation en zwerft de robot dan weg, dan betalen de tegels van eerder in de wandeling ook voor de omweg. Het probleem van **credit assignment** is uitzoeken welke eerdere stappen de eer of de schuld verdienen voor een reward die later komt: volgens [Leren van rewards](/course/open/rl-what-is-rl?lang=nl) het moeilijke deel van reinforcement learning. Monte Carlo lost het slechts gemiddeld op: een tegel waarna vaak een omweg volgt krijgt een lager gemiddelde dan een tegel waarna dat zelden gebeurt, maar dat blijkt pas na veel episodes.

        ### Nieuwe woorden en tekens

        | Woord of teken | Zeg je als | Wat het betekent | In code |
        |---|---|---|---|
        | **Monte Carlo prediction** | | $V_\pi$ schatten uit episodes die met $\pi$ gespeeld zijn, door het gemiddelde te nemen van de returns die op elke state volgden | `first_visit` |
        | **Monte Carlo-methode** | | een getal schatten door het gemiddelde te nemen van veel willekeurige trekkingen | |
        | $G_t$ | G t | de return vanaf stap $t$ tot het eind van de episode | `G[t]` |
        | **first-visit Monte Carlo** | | per episode telt alleen de return na het eerste bezoek aan een state | `first_visit` |
        | **every-visit Monte Carlo** | | de return na elk bezoek aan een state telt | |
        | **credit assignment** | | uitzoeken welke eerdere stappen de eer of de schuld verdienen voor een latere reward | |
        """),
        L.watch_out("In andere vakgebieden heet elke methode die het gemiddelde neemt van willekeurige trekkingen *Monte Carlo*, zoals duizend mogelijke jaren van een belegging doorrekenen. In reinforcement learning zijn Monte Carlo-methoden de methoden die leren van hele episodes."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Monte Carlo prediction op ROOM, wandeling na wandeling. De robot begint elke wandeling in de hoek `(0, 0)` en zwerft met de random policy rond tot het laadstation. Een getal is de <span class="c-pred">schatting</span> van de tegel na de wandelingen tot nu toe: het gemiddelde van de returns na de eerste bezoeken. De grafiek ernaast volgt de hoek: first visit en every visit, na elke wandeling.
    """)
    return


@app.cell(hide_code=True)
def explore_walks_cell():
    explore_walks = L.explore("play", "wandelingen", lambda _v: C.play_frame(int(_v), L.lang), start=C.N_WALKS,
                              values=C.LOOKS, role="",
                              task="Ga naar wandeling 1: waarom geeft every visit de hoek −24,9, en first visit −59,0? Ga dan naar 10, 100 en 1.000 wandelingen: hoe ver zit de hoek elke keer van −36?")
    explore_walks
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    In wandeling 1, van 59 stappen, is de robot 9 keer in de hoek: bij de start, na botsingen tegen de rand, en als de robot later terugkomt. First visit neemt de return van het eerste bezoek, −59, de hele wandeling. Every visit neemt ook de 8 returns na de latere bezoeken, van −58 tot −6, en de negen returns samen hebben als gemiddelde −24,89. Over veel wandelingen komen de twee bij elkaar: na 10 wandelingen staat de hoek met first visit op −26,90, na 100 op −35,83, en na 1.000 op −35,89, met every visit op −36,17. Eén wandeling geeft een return aan elke tegel die de wandeling bezoekt: na 20 wandelingen heeft elke tegel een getal, en na 1.000 wandelingen ligt elke schatting binnen 1,85 van de waarde uit policy evaluation.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Monte Carlo prediction van een policy $\pi$, first visit:

    1. Speel een episode met $\pi$, van een startstate tot het eind. Noteer bij elke stap $t$ de state en de reward direct erna.
    2. Loop de episode achteruit door, vanaf de laatste stap, met een return $G$ van 0: bij elke stap wordt $G$ de reward van de stap plus $\gamma$ keer $G$, de return vanaf stap $t$.
    3. Bewaar voor elke state de return van het eerste bezoek in de episode: de eerste stap waarop de episode in deze state is.
    4. De schatting van $V(s)$ is het gemiddelde van alle returns die voor $s$ bewaard zijn, over alle episodes tot nu toe.
    5. Speel de volgende episode en begin weer bij stap 1.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De gang van policy evaluation: vier tegels, het laadstation op tegel 0 aan de linkerkant en een muur na tegel 3. De robot stapt naar links of naar rechts, elk met kans 1/2; tegen de muur blijft de robot op tegel 3. Elke stap kost −1, en $\gamma$ = 1, dus de return van een wandeling vanaf tegel 3 is min het aantal stappen. De cel laat de robot duizend keer vanaf tegel 3 lopen, met `rng.random()` als muntje, en toont het gemiddelde van de returns na 1, 10, 100 en 1.000 wandelingen. De cel gebruikt alleen de return vanaf de start, waar het eerste bezoek aan tegel 3 is, dus stap 3 is niet nodig. Policy evaluation gaf tegel 3 de waarde −12. Voer de cel uit.
    """)
    return


@app.cell
def example_corridor():
    import numpy as np

    _rng = np.random.default_rng(0)
    _returns = []
    for _i in range(1000):
        # 1. één wandeling vanaf tegel 3 met de random policy, tot het laadstation op tegel 0
        _s = 3
        _steps = 0
        while _s != 0:
            if _rng.random() < 0.5:
                _s = _s - 1
            else:
                _s = min(_s + 1, 3)   # tegen de muur blijft de robot op tegel 3
            _steps = _steps + 1
        # 2. elke stap kost -1 en gamma = 1, dus de return van de wandeling is min het aantal stappen
        _returns.append(-_steps)
        # 4. de schatting: het gemiddelde van de returns tot nu toe
        if _i + 1 in [1, 10, 100, 1000]:
            print("wandelingen:", _i + 1, "schatting:", round(sum(_returns) / len(_returns), 2))
    print("de kortste wandeling duurde", -max(_returns), "stappen, de langste", -min(_returns))
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De eerste wandeling duurde 4 stappen, dus na 1 wandeling is de schatting −4. Na 10 wandelingen is de schatting −21,2, verder van −12 dan na 1: een paar lange wandelingen trekken het gemiddelde omlaag. Na 100 wandelingen is de schatting −13,82, en na 1.000 wandelingen −12,22, dicht bij de −12 van policy evaluation. De wandelingen duurden van 3 tot 86 stappen: één wandeling zegt weinig over de waarde; het gemiddelde van veel wandelingen zegt veel.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_returns():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De returns van een episode

        `returns(episode, gamma)` geeft de return vanaf elke stap van een episode terug, als list: plek `t` bevat $G_t$. Een episode is een list van stappen, en elke stap een paar `(state, reward)`: de state waar de robot stond en de reward van de stap die de robot daar zette. `out` begint met een 0 voor elke stap, en de loop over `range(len(episode) - 1, -1, -1)` gaat met `t` van de laatste stap terug naar de eerste: van `len(episode) - 1` omlaag tot 0, steeds één stap. `s, r = episode[t]` splitst stap `t` in de state en de reward, en `G` bevat de return vanaf de stap na `t`, 0 bij het begin. Vervang de `...` door de return vanaf stap `t`, stap 2 van het recept. De check roept `returns(WALK, 0.9)` en vijf andere gevallen aan, waaronder een wandeling waarvan de laatste stap +10 oplevert. `WALK` is een wandeling in de gang van het uitgewerkte voorbeeld, met de tegels als getallen: vanaf tegel 3 botst de robot tegen de muur, gaat twee keer naar links, één keer naar rechts en twee keer naar links, het laadstation op: 6 stappen van −1.
        """),
        L.task("Laat `returns` de return vanaf elke stap uitrekenen, achteruit vanaf de laatste stap."),
    ])
    return


@app.cell(hide_code=True)
def exercise_returns():
    box_returns = L.code_box("returns", lines=10)
    box_returns
    return (box_returns,)


@app.cell(hide_code=True)
def check_returns(box_returns):
    result_returns = L.check("returns", box_returns.value)
    result_returns
    return (result_returns,)


@app.cell(hide_code=True)
def hints_returns():
    L.hints(
        "returns",
        [
            "Stap 2 van het recept, en de eerste formule van *Het idee*: de loop gaat van de laatste stap terug naar de eerste, en als de regel voor stap `t` draait, bevat `G` nog de return vanaf de stap erna. `r` is de reward van stap `t`, en `gamma` de discount factor.",
            "In `WALK`, met γ = 0,9, geeft de laatste stap −1, de voorlaatste stap −1 + 0,9 · (−1) = −1,9, en de derde stap van achteren −1 + 0,9 · (−1,9) = −2,71.",
            "`        G = ___`",
        ],
        why="`G = r + gamma * G`: de return vanaf stap `t` is de eigen reward plus $\\gamma$ keer de return vanaf de volgende stap. Achteruit kost elke stap één optelling en één vermenigvuldiging, hoe lang de episode ook is.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_first():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. First-visit Monte Carlo

        `first_visit(episodes, gamma)` geeft de schatting terug van elke state die de episodes bezoeken: een dictionary van state naar gemiddelde return. `returns` is de versie van de course van je functie uit oefening 1. Voor elke `episode` bevat `G` de returns, is `s = episode[t][0]` de state van stap `t`, en begint `seen` als een lege set, `set()`, van de states waar de episode tot nu toe geweest is; `seen.add(s)` zet `s` erin. `total[s]` is de som van de returns die voor `s` bewaard zijn, en `count[s]` telt hoeveel het er zijn. Vervang de twee `...`: wanneer een stap telt, stap 3 van het recept, en de schatting van `s`, stap 4. De check roept `first_visit(TEN_WALKS, 1.0)` aan, met `TEN_WALKS` de eerste tien wandelingen van het speelblok, en vijf andere gevallen.
        """),
        L.task("Laat `first_visit` alleen de return van het eerste bezoek aan elke state in een episode bewaren, en het gemiddelde van de bewaarde returns teruggeven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_first():
    box_first = L.code_box("first", lines=20)
    box_first
    return (box_first,)


@app.cell(hide_code=True)
def check_first(box_first):
    result_first = L.check("first", box_first.value)
    result_first
    return (result_first,)


@app.cell(hide_code=True)
def hints_first():
    L.hints(
        "first",
        [
            "Stap 3 van het recept: een stap telt alleen bij het eerste bezoek aan de state in de episode. `s` is de state van stap `t`, en `seen` bevat de states waar deze episode al geweest is. Stap 4: de schatting is een gemiddelde; `total` bevat de som van de returns van elke state en `count` hoeveel het er zijn.",
            "In wandeling 1 van het speelblok staat de robot bij stap 0 en bij stap 1 in de hoek (0, 0): alleen stap 0 telt, met een return van −59. Als stap 1 komt, staat (0, 0) al in `seen`, en zo ook bij de 7 bezoeken daarna.",
            "`            if ___:`",
        ],
        why="`if s not in seen:` laat alleen het eerste bezoek aan `s` in een episode door, en `V[s] = total[s] / count[s]` maakt van de schatting het gemiddelde van de bewaarde returns. Met `if True:` zou je every-visit Monte Carlo hebben.",
    )
    return


@app.cell(hide_code=True)
def text_after_first():
    mo.md(r"""
    Na de tien wandelingen geeft je functie de hoek −26,90, en na alle 1.000 wandelingen van het speelblok −35,89, met −23,25 op de tegel boven het laadstation, (1, 2). Schrijf `True` op de plek van de voorwaarde en je hebt every-visit Monte Carlo: na 1.000 wandelingen geeft every visit de hoek −36,17.

    ### Veel wandelingen tegelijk met NumPy

    Hoe snel krimpt de spreiding? De cel speelt 100.000 wandelingen vanuit de hoek tegelijk, als arrays. `row` en `col` bevatten waar elke wandeling is, `steps` telt de stappen, en `going` is `True` voor elke wandeling die het laadstation nog niet bereikt heeft; `dtype` zegt wat voor getallen een array bevat. In elke ronde zet elke wandeling een willekeurige stap: `rng.integers(0, 4, size=n)` trekt voor elke wandeling een richting, en een vergelijking zoals `move == 0` geeft `True`, wat telt als 1, voor de wandelingen die omhooggaan. `np.clip(x, 0, 4)` houdt elk getal van `x` tussen 0 en 4, dus een stap tegen de rand blijft staan. `steps + going` telt 1 op bij elke wandeling die nog bezig is, en `&` (en) en `|` (of) combineren vergelijkingen element voor element: een wandeling gaat door zolang de wandeling niet tegelijk op rij 2 en kolom 2 staat. Aan het eind zet `reshape(-1, k)` de wandelingen in rijen van `k`, waarbij −1 NumPy zelf het aantal rijen laat uitrekenen, en geeft `mean(axis=1)` het gemiddelde van elke rij. Voer de cel uit.
    """)
    return


@app.cell
def example_numpy(np):
    _rng = np.random.default_rng(0)
    _n = 100000
    _row = np.zeros(_n, dtype=int)
    _col = np.zeros(_n, dtype=int)
    _steps = np.zeros(_n, dtype=int)
    _going = np.ones(_n, dtype=bool)   # True voor elke wandeling die het laadstation nog niet bereikt heeft
    while _going.sum() > 0:
        _move = _rng.integers(0, 4, size=_n)   # 0 omhoog, 1 omlaag, 2 links, 3 rechts
        _row = np.clip(_row - (_move == 0) + (_move == 1), 0, 4)
        _col = np.clip(_col - (_move == 2) + (_move == 3), 0, 4)
        _steps = _steps + _going
        _going = _going & ((_row != 2) | (_col != 2))
    print("gemiddelde:", round(_steps.mean(), 2), "standaardafwijking:", round(_steps.std(), 2))
    for _k in [10, 100, 1000]:
        _averages = _steps.reshape(-1, _k).mean(axis=1)
        print(_k, "wandelingen per gemiddelde: spreiding", round(_averages.std(), 2), "verwacht:", round(31.7 / np.sqrt(_k), 2))
    return


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    De 100.000 wandelingen duren gemiddeld 35,95 stappen, met een standaardafwijking van 31,68: dicht bij de 36 en 31,7 die uit de tabel volgen. Gemiddelden van 10 wandelingen hebben een spreiding van 9,93 stappen, van 100 wandelingen 3,18, van 1.000 wandelingen 0,95: elke keer dicht bij de verwachte 31,7/√k. Er zijn slechts 100 gemiddelden van 1.000 wandelingen, dus ook de spreiding van deze gemiddelden is met wat ruis gemeten. Tien keer zoveel wandelingen maken de spreiding zo'n 3,2 keer kleiner, want √10 ≈ 3,2. Dat is de prijs van Monte Carlo: geen tabel nodig, wel veel episodes.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_credit():
    mo.md(r"""
    ### Oefening 3. Wie betaalt voor de omweg?

    De robot begint een wandeling in de hoek. In 3 stappen bereikt de robot de tegel naast het laadstation, dan zwerft de robot weg en heeft nog 57 stappen nodig tot het laadstation: 60 stappen in totaal, elk voor −1, met $\gamma$ = 1.
    """)
    return


@app.cell(hide_code=True)
def choice_credit_cell():
    choice_credit = L.choice(
        "credit",
        question="Wat bewaart first-visit Monte Carlo uit deze wandeling voor de hoek?",
        options={
            "one": "−1: alleen de reward van de stap vanuit de hoek zelf",
            "three": "−3: het deel van de wandeling vanuit de hoek ging goed tot de tegel naast het laadstation, en de omweg daarna is niet de schuld van de hoek",
            "right": "−60: de return van de hele wandeling na de hoek, ook de 57 stappen van de omweg",
            "none": "Niets: een wandeling van deze lengte is een uitzondering, en Monte Carlo laat zo'n wandeling weg",
        },
        correct="right",
        why={
            "right": "Monte Carlo geeft een state de hele return na het eerste bezoek: −60 voor de hoek, terwijl de eigen 3 stappen goed gingen en de omweg later kwam. Uitzoeken welke stappen een return verdienden is het probleem van credit assignment, en Monte Carlo lost het slechts gemiddeld op: over veel wandelingen middelen de returns vanuit de hoek, omwegen en al, uit tot −36. Zulke lange, wisselende returns zijn ook de reden dat de returns vanuit de hoek zo'n 32 stappen spreiding hebben, en dat 1.000 wandelingen nog zo'n 1 stap twijfel laten.",
            "one": "Monte Carlo prediction schat $V_\\pi$, de return die je vanuit een state kunt verwachten tot het eind van de episode: stopt de return van de hoek na één stap?",
            "three": "Monte Carlo weet niet welke stappen goed gingen en welke niet; Monte Carlo ziet alleen de rewards van de wandeling: welke rewards komen na het eerste bezoek aan de hoek?",
            "none": "Lange wandelingen horen bij wat de random robot doet: wat zou het gemiddelde van de returns worden als Monte Carlo de lange wandelingen wegliet?",
        },
    )
    choice_credit
    return (choice_credit,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Wat is een policy waard? Policy evaluation](/course/open/rl-policy-evaluation?lang=nl) rekende je de waarden van de random robot uit met de tabel `T`: −24 naast het laadstation, −36 in een hoek. Je `returns` en `first_visit` schatten dezelfde waarden alleen uit wandelingen, zoals de agent in [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl) leerde van de eigen stappen, zonder `T`. De prijs is veel episodes: een gemiddelde van $n$ returns heeft $\sqrt{n}$ keer minder spreiding dan één return, en elke stap krijgt de hele return erna, ook de omwegen. De volgende les, *Beter spelen door ervaring: Monte Carlo control*, schat $Q(s, a)$ op dezelfde manier en maakt er de policy beter mee, bij blackjack. Verderop in dit deel werkt TD learning een waarde na elke stap bij, zonder op het eind van de episode te wachten.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_credit, result_first, result_returns):
    L.wrap_up(result_returns, result_first, choice_credit)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
