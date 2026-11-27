# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-conditioning. Route: course/route/lessen/reinforcement-learning.yaml. Checks, de hond en de katten van de course, en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-conditioning", "nl")
    C = L.checks   # de hond en de katten in de puzzelkooi van de course, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Welke automaat geeft het meest? De multi-armed bandit](/course/open/rl-bandits?lang=nl) en [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_explore():
    quick_explore = L.choice(
        "quick-explore",
        question="Een greedy agent en een ε-greedy agent met ε = 0,1 spelen op dezelfde gokautomaten. Over de laatste 100 keer trekken verdient de ε-greedy agent meer per keer. Waarom?",
        options={
            "luck": "De willekeurige keren komen toevallig bij goede automaten uit",
            "right": "De agent blijft andere automaten proberen, en vindt zo vaker de beste",
            "never": "Een greedy agent trekt nooit twee keer aan dezelfde automaat",
            "more": "De agent trekt vaker dan de greedy agent, en haalt zo meer rewards binnen in dezelfde 100 stappen",
        },
        correct="right",
        why={
            "right": "Een greedy agent blijft bij de eerste automaat die goed leek: in de course probeerde greedy gemiddeld 3,3 automaten. Met ε = 0,1 gaat één keer op de tien naar een willekeurige automaat, dus ontdekt de agent welke automaat echt het meest geeft: 1,36 per keer over de laatste 100, tegen 1,04. In deze les verkent een kat zonder ε.",
            "luck": "Een willekeurige keer komt bij elke automaat uit, ook bij de slechte. Wat ontdekt de agent door af en toe aan andere automaten te trekken? Zie [Welke automaat geeft het meest? De multi-armed bandit](/course/open/rl-bandits?lang=nl).",
            "never": "Een greedy agent trekt aan de automaat met de grootste schatting, steeds weer. Wat mist de agent door de andere nooit te proberen? Zie [Welke automaat geeft het meest? De multi-armed bandit](/course/open/rl-bandits?lang=nl).",
            "more": "Beide agents trekken één keer per stap. Wat doet de ε-greedy agent met één keer op de tien? Zie [Welke automaat geeft het meest? De multi-armed bandit](/course/open/rl-bandits?lang=nl).",
        },
    )
    quick_explore
    return (quick_explore,)


@app.cell(hide_code=True)
def quick_check_update():
    quick_update = L.choice(
        "quick-update",
        question="Bij Q-learning is Q(s, a) 2, het target r + γ max Q(s′, a′) is 4, en η = 0,5. Wat is Q(s, a) na de update?",
        options={
            "right": "3: halverwege van 2 naar 4",
            "target": "4: het target zelf",
            "whole": "6: Q plus het hele target",
            "away": "1: Q beweegt van het target af",
        },
        correct="right",
        why={
            "right": "De update zet Q een deel η van de weg naar het target: 2 + 0,5 × (4 − 2) = 3. In deze les volgt dezelfde update wat een hond verwacht, met eten als target.",
            "target": "Met η = 1 zou Q naar het target springen. Wat doet η = 0,5 met de stap van 2 naar 4? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
            "whole": "Q krijgt η keer het verschil tussen het target en Q, niet het hele target. Hoeveel is 0,5 × (4 − 2)? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
            "away": "Het verschil tussen het target en Q is 4 − 2 = 2, dus gaat Q omhoog. Hoe ver, met η = 0,5? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
        },
    )
    quick_update
    return (quick_update,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Reinforcement learning begon niet bij computers. Rond 1900 bestudeerden psychologen hoe dieren leren van eten. Ivan Pavlov, die in 1904 de Nobelprijs kreeg voor onderzoek naar de spijsvertering, liet vlak voordat een hond eten kreeg een geluid horen, zoals een metronoom of een zoemer. Als dat vaak genoeg was gebeurd, liet het geluid alleen al het water de hond in de bek lopen. In 1898 zette Edward Thorndike hongerige katten in puzzelkooien, met eten buiten de kooi in zicht. Een kat kwam eruit door aan een touwtje te trekken, op een hendel te drukken of op een plankje te stappen. De eerste keer krabde, beet en duwde de kat tot de kat toevallig het goede deed. Daarna ging het steeds sneller: niet in één plotseling moment van inzicht, maar geleidelijk. In de Engelse vertaling van de lezingen van Pavlov, *Conditioned Reflexes* (1927), versterkt (*reinforces*) het eten na het geluid de reflex, en reinforcement learning dankt de naam aan deze psychologie. De twee proeven zijn nog altijd de twee helften van RL: voorspellen wat er komt, en kiezen wat je doet. In deze les bouw je van elk een model in een paar regels Python.

    **Een voorbeeld uit het dagelijks leven.** Een hond hoort de zak met hondenbrokken ritselen en rent naar de voerbak, terwijl er nog geen eten in zit: het geluid voorspelt eten. Dezelfde hond krijgt een snoepje elke keer dat de hond op verzoek gaat zitten, en gaat al snel vaker zitten: de action levert eten op. Het eerste bestudeerde Pavlov, het tweede Thorndike.
    """)
    return


@app.cell(hide_code=True)
def predict_trial():
    guess_trial = L.guess(
        "trial",
        question="De kat van de course in een puzzelkooi kan 6 dingen doen, zoals aan de tralies krabben of miauwen; alleen aan het touwtje trekken opent de deur. Elke action heeft een sterkte, 1 aan het begin, en de kat kiest een action met een kans naar verhouding van de sterkte: eerst heeft elk van de 6 dezelfde kans, dus heeft de kat gemiddeld 6 pogingen nodig om eruit te komen. Elke keer dat de kat eruit komt, groeit de sterkte van het touwtje met 1. Hoeveel pogingen heeft de kat gemiddeld nodig de 20e keer dat de kat in de kooi gaat, denk je?",
        options={
            "one": "1: na 19 ontsnappingen weet de kat dat het touwtje de deur opent",
            "right": "Zo'n 1,25",
            "half": "Zo'n 3, de helft van de eerste keer",
            "six": "Zo'n 6: willekeurig kiezen blijft willekeurig kiezen",
        },
        reveal="**Zo'n 1,25.** Na 19 ontsnappingen heeft het touwtje sterkte 20, tegen 5 voor de andere actions samen, dus trekt de kat met een kans van 20 op 25 aan het touwtje en heeft de kat gemiddeld 25 / 20 = 1,25 pogingen nodig. De 1.000 katten van de course hebben de eerste keer gemiddeld 5,88 pogingen nodig en de twintigste keer 1,27. Toch weet de kat nooit iets: één poging op de vijf gaat nog altijd naar krabben, bijten of miauwen.",
    )
    mo.vstack([mo.Html(C.cage_picture(L.lang)), guess_trial])
    return (guess_trial,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Klassieke conditionering.** In de proef van Pavlov komt het eten na het geluid, wat de hond ook doet. De hond leert wat het geluid voorspelt. Een model heeft daar één getal voor nodig: V, hoeveel eten het geluid voorspelt, 0 aan het begin. Na elke trial zet V een stap naar het eten f dat kwam, 1 voor eten en 0 zonder:
        """),
        L.formula(r"[[param:V]] \leftarrow [[param:V]] + [[hyper:\eta]] \left( [[data:f]] - [[param:V]] \right)"),
        mo.md(r"""
        Dit is de update van Q-learning, met het eten als target en zonder volgende state. Het verschil f − V is hoe ver de voorspelling ernaast zat, en η zegt hoe groot het deel is dat V daarvan neemt. Met η = 0,2 gaat V van 0 naar 0,2, dan naar 0,36 en 0,49: grote stappen zolang de voorspelling ver ernaast zit, kleinere naarmate V dichterbij komt. Als het eten wegblijft, zakt V op dezelfde manier terug. Ook dat zag Pavlov: een geluid dat steeds zonder eten komt werkt steeds minder, en Pavlov noemde dat *extinctie*.
        """),
        mo.Html(C.pavlov_picture(L.lang)),
        mo.md(r"""
        **Instrumentele conditionering.** In de puzzelkooi van Thorndike komt het eten alleen na de juiste action. De kat leert wat de kat moet doen. Thorndike vatte de waarnemingen samen in de **wet van het effect** (*law of effect*, 1911), hier vrij vertaald: "Van alles wat een dier in dezelfde situatie doet, raakt wat samengaat met voldoening voor het dier, of er kort door gevolgd wordt, als al het andere gelijk is, steviger met de situatie verbonden." Kort gezegd: een action die iets goeds oplevert wordt waarschijnlijker in dezelfde situatie. Latere psychologen noemen dit soort leren ook *operante conditionering*.

        Een model van de kat: elke action krijgt een sterkte, 1 aan het begin, en de kat kiest een action met een kans naar verhouding van de sterkte: de sterkte van de action gedeeld door de sterktes van alle actions samen. Elke keer dat de kat eruit komt, telt de wet van het effect een toename, hier 1, op bij de sterkte van de action die de deur opende. Een action met kans p kost gemiddeld 1/p pogingen, zoals een dobbelsteen gemiddeld 6 worpen nodig heeft voor een zes. Eerst heeft het touwtje een kans van 1/6, dus 6 pogingen; na k ontsnappingen heeft het touwtje sterkte 1 + k tegen 5 voor de andere, dus heeft de kat gemiddeld (6 + k)/(1 + k) pogingen nodig.
        """),
        mo.Html(C.cat_picture(L.lang)),
        mo.md(r"""
        | | Klassieke conditionering | Instrumentele conditionering |
        |---|---|---|
        | Het eten komt | na het signaal, wat het dier ook doet | alleen na de juiste action |
        | Het dier leert | wat het signaal voorspelt | wat het dier moet doen |
        | In het model | een voorspelling V | een sterkte voor elke action |
        | In RL | een value, zoals V(s) of Q(s, a) | een policy |

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **klassieke conditionering** | | leren wat een signaal voorspelt, van een reward die op het signaal volgt, wat het dier ook doet | `predictions` |
        | **instrumentele conditionering** | | leren welke action je neemt, van een reward die alleen op de juiste action volgt | `choose` |
        | **wet van het effect** | | een action waarop iets goeds volgt wordt waarschijnlijker in dezelfde situatie | `strength[C.CORD] + C.GAIN` |
        """),
        L.watch_out("Bij klassieke conditionering komt het eten toch wel: of de hond nu kwijlt of niet, aan het eten verandert niets. De hond kiest niet; de hond voorspelt alleen. Het model van de hond heeft dus één getal en geen actions."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een model van elk soort conditionering:

    1. Klassiek: begin met V = 0. Na elke trial: V ← V + η(f − V), met f = 1 als er eten kwam en 0 als er geen eten kwam.
    2. Instrumenteel: geef elke action een sterkte van 1.
    3. Laat het dier actions kiezen, elk met een kans van de eigen sterkte gedeeld door alle sterktes samen, tot het dier de action kiest die de reward oplevert. Tel de pogingen.
    4. De wet van het effect: tel de toename op bij de sterkte van deze action. Dan de volgende trial, vanaf stap 3.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel volgt de kat van de course door de eerste 10 trials, zonder actions te trekken: de cel rekent uit wat er gemiddeld gebeurt. `np.ones(C.ACTIONS)` maakt een array van 6 enen, zoals `np.zeros` er een van nullen maakt: de sterktes aan het begin. `C.CORD` is 5, de plek van het touwtje in de array, de laatste. Elke trial print de cel de kans op het touwtje, de sterkte van het touwtje gedeeld door het totaal van alle sterktes, en de pogingen die de kat gemiddeld nodig heeft, 1 gedeeld door deze kans; daarna telt de wet van het effect `C.GAIN`, 1, op bij de sterkte van het touwtje. Voer de cel uit.
    """)
    return


@app.cell
def example_strengths():
    import numpy as np

    strength = np.ones(C.ACTIONS)   # elke action begint met sterkte 1
    for _trial in range(1, 11):
        _chance = round(float(strength[C.CORD] / strength.sum()), 2)
        _tries = round(float(strength.sum() / strength[C.CORD]), 2)   # 1 / kans
        print("trial", _trial, "kans op het touwtje", _chance, "gemiddeld pogingen", _tries)
        strength[C.CORD] = strength[C.CORD] + C.GAIN   # de wet van het effect: het touwtje opende de deur
    print("sterktes na 10 ontsnappingen:", strength)
    return np, strength


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De kans op het touwtje klimt in 10 trials van 0,17 naar 0,67, en het gemiddelde aantal pogingen daalt van 6 naar 1,5: eerst snel, dan steeds langzamer. De eerste ontsnapping brengt de kat van 6 pogingen naar 3,5, de negende van 1,56 naar 1,5. Dit zijn gemiddelden. Een kat die de actions echt trekt, kan in elke trial geluk of pech hebben, en in oefening 2 trekken je katten de actions echt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_predictions():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De hond van Pavlov

        `predictions(food, eta)` volgt de V van de hond over een reeks trials en geeft V terug voor elke trial. `food` is een lijst met een getal per trial, 1 als er eten kwam na het geluid en 0 als er geen eten kwam; `eta` is de learning rate. `v` begint bij `0.0`, en `seen` verzamelt de waarden van V: in de loop bewaart `seen.append(v)` V voor de trial, en de volgende regel past `v` aan met het eten `f` van de trial. Vervang de `...` door de nieuwe V.

        De check roept `predictions(pavlov, 0.2)` aan. `pavlov` is een lijst van 40 trials: 20 met eten na het geluid, dan 20 zonder, zoals in de afbeelding bij *Het idee*. Dan drie andere gevallen: `pavlov` met η 0,5; `alternate`, eten na elk tweede geluid, 20 trials lang, met η 0,2; en `[1, 0, 1]` met η 0,5.
        """),
        L.task("Laat `predictions` `v` bij elke trial een deel `eta` van de weg naar het eten `f` zetten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_predictions():
    box_predictions = L.code_box("predictions", lines=7)
    box_predictions
    return (box_predictions,)


@app.cell(hide_code=True)
def check_predictions(box_predictions):
    result_predictions = L.check("predictions", box_predictions.value)
    result_predictions
    return (result_predictions,)


@app.cell(hide_code=True)
def hints_predictions():
    L.hints(
        "predictions",
        [
            "Stap 1 van het recept: de nieuwe `v` is de oude `v` plus een stap naar het eten `f` van deze trial, en de stap is `eta` keer het verschil tussen het eten en wat V voorspelde. Welke van `f` en `v` komt eerst in dit verschil, zodat V naar het eten toe beweegt?",
            "In de eerste trial met eten is `v` 0 en `f` 1, dus moet de nieuwe `v` 0 + 0,2 × 1 = 0,2 zijn. In de eerste trial zonder eten, na 20 met, is `v` 0,99 en `f` 0, dus moet `v` met 0,2 × 0,99 dalen. Zet het verschil tussen haakjes, zodat `eta` het hele verschil vermenigvuldigt.",
            "`        v = v + eta * ___`",
        ],
        why="`v = v + eta * (f - v)` zet V na elke trial een vijfde van de weg naar het eten: omhoog richting 1 met eten, omlaag richting 0 zonder. Het is de update van Q-learning, met het eten als target.",
    )
    return


@app.cell(hide_code=True)
def text_after_predictions():
    mo.md(r"""
    In de afbeelding je V over `pavlov`. V klimt eerst snel en bij 1 langzaam, en zodra het eten wegblijft, zakt V op dezelfde manier terug. Elke stap is η keer hoe ver de voorspelling ernaast zat: eten dat het geluid al goed voorspelt verandert V weinig, en een verrassing verandert V veel. De volgende les, *Leren van verrassing: het Rescorla-Wagner-model*, bouwt precies daarop voort, met meer dan één signaal tegelijk.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_choose():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De kat van Thorndike

        `choose(strength, rng)` kiest de action van de kat, als getal: 0 voor de eerste action, tot en met 5 voor het touwtje. `strength` is een NumPy-array met de sterkte van elke action, en `rng` is een random number generator. `rng.choice(len(strength), p=chances)` trekt een van de getallen 0 tot en met `len(strength) - 1`, elk met de eigen kans uit `chances`: met `p=` krijgt `rng.choice` de kansen, en de kansen moeten samen 1 zijn. Vervang de `...` door `chances`: de kans van elke action, de sterkte van de action gedeeld door de sterktes van alle actions samen.

        De check roept `trials(choose, 0)` aan: de functie `trials` van de course zet een kat 20 keer in de kooi, met een random number generator met seed 0, en laat je `choose` elke action kiezen; na elke ontsnapping telt de wet van het effect 1 op bij de sterkte van het touwtje. `trials` geeft de pogingen van elke trial terug. Dan drie andere gevallen: seeds 1 en 2, en seed 3 met een kat met 4 actions, `trials(choose, 3, actions=4)`.
        """),
        L.task("Laat `chances` de kans van elke action bevatten, naar verhouding van de sterkte."),
    ])
    return


@app.cell(hide_code=True)
def exercise_choose():
    box_choose = L.code_box("choose", lines=3)
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
            "Stap 3 van het recept: `chances` moet de kans van elke action bevatten, `strength` gedeeld door één getal, zodat de kansen samen 1 zijn. Welk getal is dat? NumPy deelt een hele array in één keer door een getal, zoals `strength / 2` elke sterkte halveert.",
            "Met sterktes 1, 1, 1, 1, 1 en 6 moeten de kansen vijf keer 1/11 zijn en 6/11 voor het touwtje: elke sterkte gedeeld door 11, alle sterktes opgeteld. In het uitgewerkte voorbeeld werd de kans op het touwtje ook door dit totaal gedeeld.",
            "`    chances = ___`",
        ],
        why="`chances = strength / strength.sum()` deelt elke sterkte door het totaal van alle sterktes: de kansen zijn samen 1, en een twee keer zo sterke action krijgt twee keer de kans. `strength / np.sum(strength)` zegt hetzelfde.",
    )
    return


@app.cell(hide_code=True)
def text_after_choose():
    mo.md(r"""
    In de afbeelding het gemiddelde van 100 katten die kiezen met je `choose`, seeds 0 tot en met 99, tegen de pogingen gemiddeld volgens alleen de kansen, (6 + k)/(1 + k). Het gemiddelde blijft dicht bij de gestreepte lijn, terwijl één kat alleen op en neer springt: met seed 0 heeft de kat de eerste keer 6 pogingen nodig, en daarna 2, 2, 1, 2, 2, 2, 1, 2 en 3. Ook de katten van Thorndike werden geleidelijk sneller, niet in één keer. Niets in de kat begrijpt ooit de kooi: een grotere sterkte maakt het touwtje alleen elke keer wat waarschijnlijker.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Honderd katten in de kooi

    De demo zet 100 katten 20 keer in de kooi, seeds 0 tot en met 99, met je `choose` als je `choose` de check haalt, en anders met de `choose` van de course. Kies hoeveel actions een kat heeft, 4, 6 of 10, en de toename: hoeveel sterker de wet van het effect het touwtje maakt na elke ontsnapping. De grafiek toont de gemiddelde pogingen van de 100 katten in elke trial, en gestreept de pogingen gemiddeld volgens alleen de kansen. Ernaast de sterktes van kat 1. Daarna kun je terug door de trials.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    actions_choice = mo.ui.dropdown(options=C.action_options(L.lang), value=C.number(6, 0, L.lang), label=C.DEMO_TEXT[L.lang]["actions"])
    gain_choice = mo.ui.dropdown(options=C.gain_options(L.lang), value=C.number(1, 1, L.lang), label=C.DEMO_TEXT[L.lang]["gain"])
    cats_button = mo.ui.run_button(label="Zet 100 katten in de kooi")
    mo.hstack([actions_choice, gain_choice, cats_button], justify="start", gap=1.5, wrap=True)
    return actions_choice, cats_button, gain_choice


@app.cell(hide_code=True)
def demo_cats(actions_choice, box_choose, cats_button, gain_choice, result_choose):
    if not cats_button.value:
        _out = mo.md("*Kies de actions en de toename, en druk dan op de knop om 100 katten in de kooi te zetten.*")
    else:
        _mine = result_choose.state == "correct"
        _choose = run_code(box_choose.value, C.C_NAMES).names["choose"] if _mine else C.choose
        _actions, _gain = actions_choice.value, gain_choice.value
        _w = C.DEMO_TEXT[L.lang]
        _live = L.live(C.chart_labels(L.lang), C.TRIALS, role="ink", low=0, high=C.demo_high(_actions), window=1,
                       xticks=[1, 5, 10, 15, 20])
        _live.reference(C.chance_line(_actions, _gain), _w["chance"], role="ink-2")

        def _progress(_t, _run):
            _live.watch(lambda: C.now_picture(_run, L.lang))
            _live.extend([(_t, _run.means()[-1])], _w["learning"].format(t=_t, n=C.TRIALS))

        _run = C.demo_run(_choose, _actions, _gain, show=_progress, pause=0.15)
        if _run == "error":
            _out = _live.stopped("*Je `choose` gaf een fout, of iets anders dan een action. Haalt je `choose` de check nog?*")
        else:
            _out = mo.vstack([
                L.explore("rewind", "trial", lambda _v: C.rewind_frame(_run, int(_v), L.lang), start=C.TRIALS,
                          values=list(range(C.TRIALS + 1)), role="",
                          task="Ga terug naar 0, en dan weer vooruit: hoe groeit de balk van het touwtje, en hoeveel dalen de pogingen bij elke trial?"),
                mo.md(C.verdict(_run, _mine, L.lang)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met 10 actions kost de eerste trial zo'n 10 pogingen, en met een toename van 2 daalt de lijn sneller: na k ontsnappingen heeft een kat met n actions en toename g gemiddeld (n + k·g)/(1 + k·g) pogingen nodig. Hoe groot de toename ook is, de lijn komt in 20 trials niet bij 1, omdat de andere actions de sterkte 1 houden: met 4 actions en een toename van 2 heeft een kat in de 20e trial nog gemiddeld 1,08 pogingen nodig, en met 10 actions en een toename van 0,5 zijn het er 1,86.

    De kat is een RL-agent zonder één getal over de toekomst: de kansen zijn de policy, en de wet van het effect is hoe deze policy leert. De kat verkent zonder ε. Elke action houdt een kans boven 0, dus blijft de kat af en toe de andere proberen, steeds minder naarmate het touwtje sterker wordt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_kind():
    mo.md(r"""
    ### Oefening 3. Welk soort conditionering?

    Vier soorten leren. Bij drie komt eerst een signaal, en wat erop volgt komt hoe dan ook; bij één komt de reward alleen na de eigen action van wie er leert.
    """)
    return


@app.cell(hide_code=True)
def choice_kind_cell():
    choice_kind = L.choice(
        "kind",
        question="Welke hiervan is instrumentele conditionering?",
        options={
            "bag": "Een hond rent naar de voerbak bij het geritsel van de zak met hondenbrokken",
            "right": "Een rat drukt vaker op een hendel zodra elke druk een korrel voer oplevert",
            "drill": "Iemands schouders spannen zich bij het geluid van de boor van de tandarts",
            "bowl": "Het water loopt een kat in de bek bij het zien van de eigen voerbak, die altijd vlak voor het voer komt",
        },
        correct="right",
        why={
            "right": "Op de hendel drukken is een action, en het voer komt alleen daarna: de wet van het effect maakt drukken waarschijnlijker. Bij de andere drie komt eerst een signaal, en wat erop volgt komt hoe dan ook: klassieke conditionering.",
            "bag": "Het eten komt of de hond nu rent of niet; het geritsel voorspelt het eten alleen. Welke van de vier krijgt de reward alleen na een eigen action?",
            "drill": "De boor komt of de schouders zich nu spannen of niet; het geluid voorspelt de boor alleen. Welke van de vier krijgt de reward alleen na een eigen action?",
            "bowl": "Het voer komt wat de kat ook doet; de voerbak voorspelt het voer alleen. Welke van de vier krijgt de reward alleen na een eigen action?",
        },
    )
    choice_kind
    return (choice_kind,)


@app.cell(hide_code=True)
def text_exercise_link():
    mo.md(r"""
    ### Oefening 4. De hond in RL

    De hond van Pavlov leerde wat een geluid voorspelt, met de update van Q-learning.
    """)
    return


@app.cell(hide_code=True)
def choice_link_cell():
    choice_link = L.choice(
        "link",
        question="Met welk deel van een RL-agent komt klassieke conditionering overeen?",
        options={
            "policy": "De policy: welke action je neemt in een state",
            "right": "Een value: hoeveel reward een state voorspelt",
            "explore": "Exploration: af en toe andere actions proberen, ook als een van de actions al de beste lijkt",
            "discount": "De discount: rewards later tellen minder zwaar",
        },
        correct="right",
        why={
            "right": "De hond leert wat het geluid voorspelt, zoals V(s) leert wat een state voorspelt, met dezelfde update: een stap naar wat er kwam. De kat van Thorndike komt overeen met de policy: welke action je neemt.",
            "policy": "De hond kiest niets: het eten komt toch wel. Wat leert de hond over het geluid?",
            "explore": "De hond heeft geen actions om te proberen. Wat leert de hond over het geluid?",
            "discount": "Het model van de hond heeft één stap: het geluid, en dan wel of geen eten. Wat leert V over het geluid?",
        },
    )
    choice_link
    return (choice_link,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Welke automaat geeft het meest? De multi-armed bandit](/course/open/rl-bandits?lang=nl) leerde een agent welke automaat het meest geeft; hier leerde een kat welke action de deur opent, met een regel die de psychologie in 1911 opschreef. De hond van Pavlov leert een voorspelling met de update van Q-learning; de kat van Thorndike leert een policy met de wet van het effect. RL brengt de twee samen: een agent leert values, en kiest daarmee de actions. De volgende les, *Leren van verrassing: het Rescorla-Wagner-model*, geeft de hond meer dan één signaal, en laat zien dat een signaal alleen leert voor zover het eten een verrassing is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_kind, choice_link, result_choose, result_predictions):
    L.wrap_up(result_predictions, result_choose, choice_kind, choice_link)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
