# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-rescorla-wagner. Route: course/route/lessen/reinforcement-learning.yaml. Checks, de hond van de course met een geluid en een lamp, en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-rescorla-wagner", "nl")
    C = L.checks   # de hond van de course met een geluid en een lamp, de experimenten, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [De honden van Pavlov en de katten van Thorndike: conditionering](/course/open/rl-conditioning?lang=nl) en [Leren na elke stap: TD(0)](/course/open/rl-td-prediction?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_dog():
    quick_dog = L.choice(
        "quick-dog",
        question="Bij de hond van Pavlov heeft het geluid V = 0,5, η = 0,2, en er komt eten na het geluid. Wat is V na deze trial?",
        options={
            "plus": "0,7: V plus η",
            "right": "0,6: een vijfde van de weg van 0,5 naar 1",
            "jump": "1: V springt meteen naar het eten",
            "step": "0,1: alleen de stap, dus de hond vergeet de oude V",
        },
        correct="right",
        why={
            "right": "V zet een stap van η keer hoe ver de voorspelling ernaast zat: 0,5 + 0,2 × (1 − 0,5) = 0,6. In deze les komt er een lamp bij het geluid, en delen de twee signalen dit verschil.",
            "plus": "Met η erbij groeit V elke trial met 0,2, voorbij 1 en verder. Hoe groot is het deel van het verschil tussen het eten en V dat V neemt? Zie [De honden van Pavlov en de katten van Thorndike: conditionering](/course/open/rl-conditioning?lang=nl).",
            "jump": "Met η = 1 zou V naar het eten springen. Wat doet η = 0,2 met de stap van 0,5 naar 1? Zie [De honden van Pavlov en de katten van Thorndike: conditionering](/course/open/rl-conditioning?lang=nl).",
            "step": "De stap is 0,2 × (1 − 0,5) = 0,1, en de stap komt bij de oude V. Hoeveel is 0,5 + 0,1? Zie [De honden van Pavlov en de katten van Thorndike: conditionering](/course/open/rl-conditioning?lang=nl).",
        },
    )
    quick_dog
    return (quick_dog,)


@app.cell(hide_code=True)
def quick_check_td():
    quick_td = L.choice(
        "quick-td",
        question="Bij TD(0) is V(S) = 0,5, V(S′) = 0,8, de reward R is 0 en γ = 1. Wat is de TD error δ?",
        options={
            "next": "0,8: de value van de volgende state",
            "sign": "−0,3: de oude value min het target",
            "right": "0,3: het target 0,8 min de oude 0,5",
            "reward": "0: er kwam geen reward, dus verandert er niets, wat de values van de twee states ook zijn",
        },
        correct="right",
        why={
            "right": "δ = R + γV(S′) − V(S) = 0 + 1 × 0,8 − 0,5 = 0,3: het target min de oude value. In deze les komt hetzelfde soort verschil terug in een model van dieren uit 1972, met het eten als target.",
            "next": "V(S′) is een deel van het target R + γV(S′). Wat blijft er over als de oude value V(S) eraf gaat? Zie [Leren na elke stap: TD(0)](/course/open/rl-td-prediction?lang=nl).",
            "sign": "De TD error is het target min de oude value, dus de volgorde telt. Hoeveel is 0,8 − 0,5? Zie [Leren na elke stap: TD(0)](/course/open/rl-td-prediction?lang=nl).",
            "reward": "Zonder reward kan het target R + γV(S′) toch verschillen van V(S): de volgende state kan meer waard zijn. Hoeveel is 0 + 1 × 0,8 − 0,5? Zie [Leren na elke stap: TD(0)](/course/open/rl-td-prediction?lang=nl).",
        },
    )
    quick_td
    return (quick_td,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Het model van de hond van Pavlov in *De honden van Pavlov en de katten van Thorndike: conditionering* had één signaal, een geluid, en één getal V voor wat het geluid voorspelt. Echte proeven hebben vaak meer dan één signaal tegelijk: een geluid en een lamp, een geur en een plek. In 1969 trainde de psycholoog Leon Kamin ratten met een ruisgeluid, gevolgd door een korte elektrische schok, tot de ratten bang waren voor het ruisgeluid alleen. Daarna kwam het ruisgeluid samen met een lamp, gevolgd door dezelfde schok. Getest met de lamp alleen toonden de ratten bijna geen angst: de lamp was vaak voor de schok gekomen, en toch hadden de ratten er bijna niets over geleerd. Kamin noemde dit *blocking*: wat het ruisgeluid al voorspelde, kon de lamp niet meer leren. In 1972 verklaarden Robert Rescorla en Allan Wagner het met één regel: een signaal leert alleen voor zover de uitkomst een verrassing is, en alle aanwezige signalen delen samen één verrassing. De regel werd een van de bekendste modellen in de psychologie. Later bleek de regel ook de update te zijn waarmee gradient descent een lineair model fit, en de update die TD learning over de tijd uitspreidt. In deze les bouw je de regel in twee korte functies.

    **Een voorbeeld uit het dagelijks leven.** Een hond weet dat de deurbel een bezoeker betekent, en rent naar de deur als de bel gaat. Dan komt er een lamp in de gang, en de lamp gaat steeds samen met de bel aan. Maanden later gaat de bel kapot, en gaat alleen de lamp aan als een bezoeker op de knop drukt. De hond kijkt nauwelijks op: de bel zei altijd al alles, dus had de lamp nooit iets toe te voegen.
    """)
    return


@app.cell(hide_code=True)
def predict_blocking():
    guess_blocking = L.guess(
        "blocking",
        question="De hond van de course hoort in 20 trials een geluid voor het eten, met η = 0,2, zoals in het model van de hond van Pavlov. Daarna gaat 20 trials lang samen met het geluid een lamp aan, en volgt er net als eerst eten. Beide signalen beginnen bij V = 0. Hoeveel eten voorspelt de lamp na deze 40 trials, denk je, van 0 voor geen eten tot 1 voor elke keer eten?",
        options={
            "half": "Zo'n 0,5: de lamp deelt het eten met het geluid",
            "right": "Bijna niets: zo'n 0,01",
            "full": "Zo'n 1: de lamp kwam 20 keer voor het eten, even vaak als het geluid eerst",
            "less": "Zo'n 0,3: minder dan het geluid, omdat de lamp later kwam",
        },
        reveal="**Bijna niets: zo'n 0,01.** Na de 20 trials met alleen het geluid voorspelt het geluid 0,99 van het eten. Als de lamp erbij komt, voorspellen de twee samen al 0,99, dus is het eten nauwelijks een verrassing: 1 − 0,99 = 0,01. De lamp leert alleen van deze kleine verrassing, en voorspelt na 20 trials 0,006. Een hond met geluid en lamp samen vanaf het begin leert 0,50 voor elk van de twee. Dit vond Kamin: het geluid blokkeert de lamp.",
    )
    mo.vstack([mo.Html(C.plan_picture(L.lang)), guess_blocking])
    return (guess_blocking,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Eén V per signaal, één verrassing voor allemaal.** Het Rescorla-Wagner model geeft elk signaal een eigen V: hoeveel eten het signaal voorspelt, 0 aan het begin. In een trial voorspellen de aanwezige signalen samen het eten: de V's opgeteld, geschreven als ΣV, met de Griekse hoofdletter sigma voor optellen, zoals in de mean squared error. De verrassing is het eten f dat kwam, 1 of 0, min deze voorspelling. Dan zet elk aanwezig signaal dezelfde stap, η keer de verrassing, en een signaal dat ontbreekt houdt de eigen V:
        """),
        L.formula(r"[[param:V_i]] \leftarrow [[param:V_i]] + [[hyper:\eta]] \left( [[data:f]] - \Sigma [[param:V]] \right) \quad \text{voor elk aanwezig signaal } i"),
        mo.md(r"""
        Met één signaal is ΣV de V van dat signaal, en is de regel de update van de hond van Pavlov. Met twee signalen concurreren de signalen om wat er te leren valt: zodra het geluid het eten voorspelt, is de verrassing ook voor de lamp weg. Rescorla en Wagner gaven elk signaal en elke soort eten een eigen snelheid, α en β; de course neemt de twee samen als één learning rate η. Rescorla en Wagner schreven λ voor het eten; hier is het f, zoals in het model van de hond.
        """),
        mo.Html(C.shared_picture(L.lang)),
        mo.md(r"""
        **Blocking.** Volg de hond van de course met η = 0,2. Na 20 trials met alleen het geluid is V van het geluid 0,99. Dan komt de lamp erbij: samen voorspellen de twee 0,99 + 0 = 0,99, de verrassing is 0,01, en elk signaal zet een stap van 0,2 × 0,01 = 0,002. Vanaf daar wordt de verrassing alleen kleiner, dus voorspelt de lamp na 20 trials 0,006. Een hond met geluid en lamp samen vanaf het begin ziet in de eerste trial een verrassing van 1; de twee delen elke stap, en na 20 trials voorspelt elk signaal 0,50. De afbeelding toont beide experimenten.
        """),
        mo.Html(C.runs_picture(L.lang)),
        mo.md(r"""
        **Een verrassing onder 0.** Als het eten wegblijft terwijl de aanwezige signalen wel eten voorspelden, is de verrassing onder 0, en gaan de V's omlaag. Laat het geluid zonder lamp eten brengen, en het geluid met de lamp geen eten, steeds opnieuw: het geluid klimt richting 1, en de lamp zakt onder 0, naar −0,81 na 20 trials van elke soort, zodat de twee samen bijna niets voorspellen. Een V onder 0 is een voorspelling dat het eten niet komt.

        **Dezelfde regel elders.** In *Leren na elke stap: TD(0)* ging V(S) na elke stap een stukje naar het target R + γV(S′). Het Rescorla-Wagner model doet dezelfde beweging één keer per trial: het eten f is het target, en ΣV speelt de rol van V(S). Richard Sutton en Andrew Barto bouwden hierop voort. Het TD-model van Sutton en Barto voor klassieke conditionering (1987) voorspelt op elk moment binnen een trial, niet alleen één keer per trial, zodat de timing van de signalen telt, net als bij echte dieren. En met ΣV geschreven als `present @ v`, het inproduct van een array met een 1 voor elk aanwezig signaal en de array van V's, is de regel een stap van gradient descent voor een lineair model, met de helft van de kwadratische fout als loss: `present` is de input, `v` bevat de parameters, en het eten is het label. Sutton en Barto wezen hier in 1981 op: het is de regel waarmee Bernard Widrow en Ted Hoff in 1960 een eenvoudig elektronisch neuron trainden.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **Rescorla-Wagner model** | | een model van conditionering waarin elk aanwezig signaal dezelfde stap zet, η keer de verrassing: het eten min wat alle aanwezige signalen samen voorspellen | `rw_step` |
        | **blocking** | | als een signaal het eten al voorspelt, leert een tweede signaal dat erbij komt bijna niets, omdat het eten nauwelijks nog een verrassing is | `run(blocking, 0.2)` |
        | $\Sigma V$ | sigma V | wat de aanwezige signalen samen voorspellen: de V's opgeteld | `present @ v` |
        """),
        L.watch_out("De verrassing is één getal voor de hele trial, niet één per signaal. Een signaal leert niet van hoe ver de eigen V ernaast zat: het signaal leert van hoe ver alle aanwezige signalen samen ernaast zaten. Daardoor is blocking mogelijk."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Eén trial van het Rescorla-Wagner model:

    1. Geef elk signaal een V, 0 aan het begin.
    2. Tel in een trial de V's van de aanwezige signalen op: ΣV, wat de signalen samen voorspellen.
    3. De verrassing is het eten min deze voorspelling: f − ΣV, met f = 1 als er eten kwam en 0 als er geen eten kwam.
    4. Elk aanwezig signaal zet een stap van η keer de verrassing; de andere signalen houden de eigen V. Dan de volgende trial, vanaf stap 2.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel volgt de hond van de course door het experiment van de gok, en print de eerste drie trials met de lamp. `v` bevat de V van het geluid en van de lamp, in deze volgorde. `sound` en `both` zijn arrays met een 1 voor elk aanwezig signaal. `sound @ v` is het inproduct, 1 × V van het geluid + 0 × V van de lamp: wat de aanwezige signalen samen voorspellen. Door de stap met `sound` te vermenigvuldigen krijgt de lamp een stap van 0, omdat de lamp ontbreekt. Voer de cel uit.
    """)
    return


@app.cell
def example_blocking():
    import numpy as np

    v = np.zeros(2)            # V van het geluid en van de lamp
    sound = np.array([1, 0])   # alleen het geluid
    both = np.array([1, 1])    # het geluid en de lamp samen
    for _trial in range(20):
        v = v + 0.2 * (1 - sound @ v) * sound   # er kwam eten: f = 1
    print("na 20 trials met alleen het geluid")
    print("geluid", round(float(v[0]), 4), "lamp", round(float(v[1]), 4))
    for _trial in range(21, 24):
        _surprise = 1 - both @ v
        v = v + 0.2 * _surprise * both
        print("trial", _trial, "verrassing", round(float(_surprise), 4), "V", np.round(v, 4))
    return both, np, sound, v


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Het geluid komt in 20 trials op 0,9885, en de lamp krijgt 0,0023 in de eerste trial met de lamp: een vijfde van een verrassing van 0,0115. Elke trial maakt de verrassing kleiner, tot 0,0069 en 0,0042, dus krijgt de lamp steeds minder. Beide V's groeien nog, maar alleen met wat er nog te leren valt.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rw_step():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Eén trial

        `rw_step(v, present, food, eta)` doet één trial van het Rescorla-Wagner model en geeft de nieuwe V's terug. `v` is een NumPy-array met de V van elk signaal; `present` is een array met een 1 voor elk signaal dat in de trial aanwezig is en een 0 voor elk signaal dat ontbreekt; `food` is f, 1 als er eten kwam en 0 als er geen eten kwam; `eta` is de learning rate. De regel met `return` doet stap 4 van het recept al: `eta * surprise * present` is 0 voor een signaal dat ontbreekt. Vervang de `...` door `surprise`: het eten min wat de aanwezige signalen samen voorspellen.

        De check roept `rw_step(np.array([0.5, 0.0]), np.array([1, 1]), 1, 0.2)` aan: het geluid voorspelt 0,5, de lamp 0, beide zijn aanwezig, en er komt eten. Dan vier andere gevallen: V 0,99 en 0, beide aanwezig, met eten; alleen het geluid met eten, terwijl de lamp V −0,3 heeft; beide aanwezig zonder eten, met η 0,5; en drie signalen, waarvan het eerste en het laatste aanwezig zijn.
        """),
        L.task("Maak van `surprise` het eten min wat de aanwezige signalen samen voorspellen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rw_step():
    box_rw_step = L.code_box("rw_step", lines=3)
    box_rw_step
    return (box_rw_step,)


@app.cell(hide_code=True)
def check_rw_step(box_rw_step):
    result_rw_step = L.check("rw_step", box_rw_step.value)
    result_rw_step
    return (result_rw_step,)


@app.cell(hide_code=True)
def hints_rw_step():
    L.hints(
        "rw_step",
        [
            "Stap 2 en 3 van het recept: `surprise` is het eten min ΣV, de V's van de aanwezige signalen opgeteld. In het uitgewerkte voorbeeld gaf `sound @ v` deze voorspelling voor een trial met alleen het geluid. Welke array zegt hier welke signalen aanwezig zijn?",
            "Met `v` op 0,5 en 0 en beide signalen aanwezig voorspellen de twee samen 1 × 0,5 + 1 × 0 = 0,5, en de verrassing is 1 − 0,5 = 0,5. Gebruik `food`, niet 1, zodat een trial zonder eten ook werkt.",
            "`    surprise = ___`",
        ],
        why="`surprise = food - present @ v` is het eten min wat de aanwezige signalen samen voorspellen: `present @ v` telt de V's van de aanwezige signalen op, omdat een signaal dat ontbreekt 0 keer meetelt. `food - np.dot(present, v)` zegt hetzelfde.",
    )
    return


@app.cell(hide_code=True)
def text_after_rw_step():
    mo.md(r"""
    In de afbeelding je `rw_step` in twee trials. Vanaf V 0,5 en 0 is de verrassing 0,5, en zetten beide signalen een stap van 0,1. Vanaf V 0,99 en 0 voorspelt het geluid het eten al: de verrassing is 0,01, en de lamp krijgt 0,002. De lamp is in beide trials even goed aanwezig; het verschil zit in hoeveel van het eten nog een verrassing was.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_run():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een heel experiment

        `run(trials, eta)` volgt de V's van het geluid en de lamp over de trials van een experiment, en geeft een NumPy-array terug met een rij per moment: de V's aan het begin, en na elke trial. `trials` is een lijst van trials, elk een paar van `present` en `food`. `for present, food in trials:` geeft de twee delen van elk paar een eigen naam, zoals `x, y = point` doet bij een tuple. `v` begint als `np.zeros(2)`, `seen.append(v)` bewaart de V's na elke trial in de lijst `seen`, en `np.array(seen)` maakt van deze lijst van arrays één array met een rij per moment. Vervang de `...` door de nieuwe `v`: één trial van het Rescorla-Wagner model, met `rw_step`. Het vak gebruikt de `rw_step` van de course, met dezelfde uitkomst als je eigen `rw_step`.

        De check roept `run(blocking, 0.2)` aan. `blocking` is het experiment van de gok: 20 trials met alleen het geluid en eten, dan 20 met geluid en lamp samen en eten. Dan vier andere gevallen: `control`, 20 trials zonder signalen of eten, dan dezelfde 20 met geluid en lamp; `inhibition`, 20 keer een trial met alleen het geluid en eten, gevolgd door een trial met geluid en lamp en geen eten; `blocking` met η 0,5; en `short`, drie trials, met η 0,3.
        """),
        L.task("Laat elke trial `v` bijwerken met `rw_step`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_run():
    box_run = L.code_box("run", lines=7)
    box_run
    return (box_run,)


@app.cell(hide_code=True)
def check_run(box_run):
    result_run = L.check("run", box_run.value)
    result_run
    return (result_run,)


@app.cell(hide_code=True)
def hints_run():
    L.hints(
        "run",
        [
            "Stap 4 van het recept, voor elke trial: de nieuwe `v` is wat `rw_step` geeft voor de V's tot nu toe, de signalen `present`, het `food` en de learning rate `eta`. In welke volgorde neemt `rw_step` deze vier?",
            "In de eerste trial van `blocking` is `present` alleen het geluid en `food` 1, dus moeten de V's van 0 en 0 naar 0,2 en 0 gaan. Geef `rw_step` de V's van de vorige trial, geen nieuwe array met nullen.",
            "`        v = ___`",
        ],
        why="`v = rw_step(v, present, food, eta)` doet één trial vanaf de V's tot nu toe, en `seen.append(v)` bewaart de V's daarna. Over `blocking` eindigt de lamp op 0,006: het geluid blokkeert de lamp.",
    )
    return


@app.cell(hide_code=True)
def text_after_run():
    mo.md(r"""
    In de afbeelding je `run` over `blocking` en `control`. In `blocking` blijft de lijn van de lamp plat op 0 terwijl het geluid leert, en beweegt de lijn nauwelijks als de lamp erbij komt: het geluid had alles al. In `control` liggen de twee lijnen op elkaar en stoppen op 0,50: hetzelfde eten, gedeeld. In `inhibition` eindigt de lamp op −0,81: zodra het geluid eten voorspelt, leert de lamp dat het eten wegblijft als de lamp komt.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Blocking, trial na trial

    De demo doet het experiment van de gok met je `run` als je `run` de check haalt, en anders met de `run` van de course. Kies in hoeveel trials het geluid alleen komt voordat de lamp erbij komt, 0, 5, 10 of 20, en de learning rate. Na deze trials komt de lamp 20 trials lang met het geluid, steeds gevolgd door eten. De grafiek toont V van de lamp, en gestreept V van het geluid; ernaast de V's na de laatste trial en de verrassing van deze trial. Daarna kun je terug door de trials.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    first_choice = mo.ui.dropdown(options=C.first_options(L.lang), value=C.number(20, 0, L.lang), label=C.DEMO_TEXT[L.lang]["first"])
    eta_choice = mo.ui.dropdown(options=C.eta_options(L.lang), value=C.number(0.2, 1, L.lang), label=C.DEMO_TEXT[L.lang]["eta"])
    run_button = mo.ui.run_button(label="Doe het experiment")
    mo.hstack([first_choice, eta_choice, run_button], justify="start", gap=1.5, wrap=True)
    return eta_choice, first_choice, run_button


@app.cell(hide_code=True)
def demo_blocking(box_run, eta_choice, first_choice, result_run, run_button):
    if not run_button.value:
        _out = mo.md("*Kies de trials met alleen het geluid en de learning rate, en druk dan op de knop om het experiment te doen.*")
    else:
        _mine = result_run.state == "correct"
        _run = run_code(box_run.value, C.run_names()).names["run"] if _mine else C.run
        _first, _eta = first_choice.value, eta_choice.value
        _n = _first + C.PHASE
        _w = C.DEMO_TEXT[L.lang]
        _live = L.live(C.chart_labels(L.lang), _n, role="param", low=0, high=1, window=1, xticks=C.demo_xticks(_first),
                       key=C.chart_key(L.lang))

        def _progress(_t, _demo):
            _live.watch(lambda: C.now_picture(_demo, L.lang))
            _live.reference(_demo.points(C.SOUND), "", role="param", replace=True)
            _live.extend([(_t, float(_demo.values[_t, C.LIGHT]))], _w["learning"].format(t=_t, n=_n))

        _demo = C.demo_run(_run, _first, _eta, show=_progress, pause=0.08)
        if _demo == "error":
            _out = _live.stopped("*Je `run` gaf een fout, of iets anders dan een rij van twee V's voor elk moment. Haalt je `run` de check nog?*")
        else:
            _out = mo.vstack([
                L.explore("rewind", "trial", lambda _v: C.rewind_frame(_demo, int(_v), L.lang), start=_n,
                          values=list(range(_n + 1)), role="",
                          task="Ga naar de trial waar de lamp erbij komt, en dan vooruit: hoe groot is de verrassing daar, en hoeveel leert de lamp ervan?"),
                mo.md(C.verdict(_demo, _mine, L.lang)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met η 0,2 eindigt de lamp op 0,006 na 20 trials met alleen het geluid, op 0,054 na 10, op 0,16 na 5, en op 0,50 zonder deze trials, als geluid en lamp het eten delen. Hoe langer het geluid al leert, hoe minder verrassing er voor de lamp overblijft. Met een grotere η leert het geluid sneller, en blokkeert het geluid meer: met η 0,5 en 20 trials voorspelt het geluid het eten helemaal voordat de lamp komt, en eindigt de lamp op 0,0000005. Met η 0,1 staat het geluid nog op 0,88 als de lamp erbij komt, en krijgt de lamp 0,060.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_latent():
    mo.md(r"""
    ### Oefening 3. Wat het model mist

    Elk model laat iets weg. Hier komt de lamp vaak, nog voordat er ooit eten is.
    """)
    return


@app.cell(hide_code=True)
def choice_latent_cell():
    choice_latent = L.choice(
        "latent",
        question="Voordat er ooit eten komt, ziet de hond van de course de lamp 50 keer, en er volgt niets. Wat voorspelt de lamp na deze 50 trials volgens het Rescorla-Wagner model?",
        options={
            "right": "0: de verrassing is elke keer 0 − 0",
            "below": "Onder 0: de lamp leert dat er geen eten op volgt",
            "up": "Iets boven 0: de lamp is 50 keer gezien",
            "half": "0,5: het model weet het nog niet",
        },
        correct="right",
        why={
            "right": "Met alleen de lamp en geen eten is de verrassing 0 − 0 = 0, dus beweegt V van de lamp nooit. Echte dieren leren hier wel iets: een signaal dat vaak kwam zonder dat er iets volgde wordt later langzamer geleerd, als er wel eten op gaat volgen. Psychologen noemen dit latente inhibitie (latent inhibition; Lubow en Moore, 1959), en het Rescorla-Wagner model kan het niet verklaren: een van de bekende grenzen van het model.",
            "below": "Een V gaat onder 0 als er eten voorspeld werd en het eten wegbleef. Wat voorspelde de hond voor deze trials, en wat kwam er?",
            "up": "Het zien van de lamp verandert de V alleen via de verrassing. Hoeveel is het eten min ΣV als er geen eten komt en V 0 is?",
            "half": "V van de lamp begint bij 0. Wat doet de verrassing van elke trial met deze V?",
        },
    )
    choice_latent
    return (choice_latent,)


@app.cell(hide_code=True)
def text_exercise_td():
    mo.md(r"""
    ### Oefening 4. Dezelfde regel bij TD learning

    Het Rescorla-Wagner model en TD(0) zetten allebei een voorspelling een stap naar een target.
    """)
    return


@app.cell(hide_code=True)
def choice_td_cell():
    choice_td = L.choice(
        "td",
        question="Bij TD(0) zet V(S) een stap naar R + γV(S′). Welk deel van het Rescorla-Wagner model speelt de rol van R + γV(S′)?",
        options={
            "lr": "η, hoe groot de stap van de V's is",
            "right": "f, het eten dat aan het eind van de trial komt",
            "together": "ΣV, wat de aanwezige signalen samen voorspellen, voordat het eten komt",
            "present": "Welke signalen in de trial aanwezig zijn",
        },
        correct="right",
        why={
            "right": "Beide regels zetten een voorspelling een stap naar een target. Bij TD(0) is het target R + γV(S′), na elke stap; in het Rescorla-Wagner model is het f, het eten, één keer per trial, en speelt ΣV de rol van V(S). De verrassing f − ΣV is de TD error van een trial van één stap, zonder iets erna.",
            "lr": "η is in beide regels de learning rate: hoe ver de voorspelling beweegt. Waar beweegt de voorspelling naartoe?",
            "together": "ΣV is de voorspelling zelf, zoals V(S) bij TD(0). Waar beweegt de voorspelling naartoe?",
            "present": "De aanwezige signalen zeggen welke V's de stap zetten. Waar zetten deze V's de stap naartoe?",
        },
    )
    choice_td
    return (choice_td,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [De honden van Pavlov en de katten van Thorndike: conditionering](/course/open/rl-conditioning?lang=nl) leerde één geluid wat het geluid voorspelt. Met meer signalen laat het Rescorla-Wagner model de signalen één verrassing delen, en dat verklaart blocking: een signaal leert alleen wat de andere signalen nog niet voorspellen. Dezelfde verrassing, wat er kwam min wat er voorspeld was, drijft de TD error van [Leren na elke stap: TD(0)](/course/open/rl-td-prediction?lang=nl) en de update van gradient descent voor een lineair model. De volgende les, *Dopamine als prediction error*, laat zien dat cellen in de hersenen precies zo'n verrassing lijken door te geven.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_latent, choice_td, result_run, result_rw_step):
    L.wrap_up(result_rw_step, result_run, choice_latent, choice_td)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
