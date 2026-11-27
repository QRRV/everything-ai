# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-dopamine. Route: course/route/lessen/reinforcement-learning.yaml. Checks, de trials van de course met een lamp en sap, en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-dopamine", "nl")
    C = L.checks   # de trials van de course met een lamp en sap, het TD-model van conditionering, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Leren van verrassing: het Rescorla-Wagner model](/course/open/rl-rescorla-wagner?lang=nl) en [Leren na elke stap: TD(0)](/course/open/rl-td-prediction?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_surprise():
    quick_surprise = L.choice(
        "quick-surprise",
        question="In het Rescorla-Wagner model heeft het geluid V = 0,8 en de lamp V = 0,2. Beide komen, en er volgt eten. Wat is de verrassing van deze trial?",
        options={
            "light": "0,8: het eten min de V van de lamp",
            "right": "0: samen voorspellen de twee al het eten",
            "food": "1: het eten kwam, en dat is de verrassing",
            "each": "0,2 voor het geluid en 0,8 voor de lamp: elk signaal heeft een eigen verrassing",
        },
        correct="right",
        why={
            "right": "Samen voorspellen de twee 0,8 + 0,2 = 1, dus is de verrassing 1 − 1 = 0, en blijven beide V's staan. In deze les wordt de verrassing op elk moment binnen een trial uitgerekend, niet één keer per trial.",
            "light": "De verrassing is één getal voor de hele trial: het eten min wat de aanwezige signalen samen voorspellen. Wat is 0,8 + 0,2? Zie [Leren van verrassing: het Rescorla-Wagner model](/course/open/rl-rescorla-wagner?lang=nl).",
            "food": "De verrassing is het eten min wat er voorspeld was, niet het eten alleen. Wat voorspellen het geluid en de lamp samen? Zie [Leren van verrassing: het Rescorla-Wagner model](/course/open/rl-rescorla-wagner?lang=nl).",
            "each": "In het Rescorla-Wagner model delen de aanwezige signalen één verrassing. Wat is 1 − (0,8 + 0,2)? Zie [Leren van verrassing: het Rescorla-Wagner model](/course/open/rl-rescorla-wagner?lang=nl).",
        },
    )
    quick_surprise
    return (quick_surprise,)


@app.cell(hide_code=True)
def quick_check_td():
    quick_td = L.choice(
        "quick-td",
        question="In TD(0) is V(S) = 0,9, de reward R is 0, γ = 1, en de volgende state heeft V(S′) = 0. Wat is de TD error δ?",
        options={
            "nothing": "0: er kwam geen reward, dus valt er niets te vergelijken",
            "old": "0,9: hoe ver de oude waarde ernaast zat",
            "right": "−0,9: het target 0 min de oude waarde 0,9",
            "next": "1: de volgende state is minder waard dan deze, met een hele reward",
        },
        correct="right",
        why={
            "right": "δ = R + γV(S′) − V(S) = 0 + 1 × 0 − 0,9 = −0,9: het target ligt onder de oude waarde, dus is de fout kleiner dan 0. In deze les gebeurt dat als sap dat verwacht werd wegblijft.",
            "nothing": "Zonder reward kan het target R + γV(S′) nog steeds anders zijn dan V(S). Wat is 0 + 1 × 0 − 0,9? Zie [Leren na elke stap: TD(0)](/course/open/rl-td-prediction?lang=nl).",
            "old": "De TD error is het target min de oude waarde, dus telt het teken. Wat is hier groter, het target of V(S)? Zie [Leren na elke stap: TD(0)](/course/open/rl-td-prediction?lang=nl).",
            "next": "δ = R + γV(S′) − V(S). Wat krijg je met R = 0, V(S′) = 0 en V(S) = 0,9? Zie [Leren na elke stap: TD(0)](/course/open/rl-td-prediction?lang=nl).",
        },
    )
    quick_td
    return (quick_td,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Dopamine is een stof waarmee sommige cellen in de hersenen een signaal doorgeven, en dopamine wordt vaak de stof van genot genoemd. In de jaren tachtig en negentig mat de neurowetenschapper Wolfram Schultz losse dopamineneuronen bij apen, terwijl de apen leerden dat een lampje betekent dat er ruim een seconde later een druppel vruchtensap komt. De neuronen vuurden niet gewoon bij het sap. Eerst, toen de apen nog niet wisten wat het lampje betekende, vuurden de neuronen een salvo als het sap kwam. Toen de apen het geleerd hadden, kwam het salvo bij het lampje, en reageerden de neuronen niet meer op het sap zelf. En als het sap na het lampje wegbleef, werden de neuronen stil, onder het gewone tempo, precies op het moment dat het sap had moeten komen. In 1997 lieten Schultz, de theoreticus Peter Dayan en de neurowetenschapper Read Montague zien dat dit de fout is van een TD-model dat leert het sap te voorspellen: dezelfde soort fout als in *Leren na elke stap: TD(0)*. Een idee uit reinforcement learning bleek te beschrijven wat cellen in hersenen doen, en het is nog steeds een van de duidelijkste verbanden tussen AI en neurowetenschap. In deze les bouw je dat model in twee korte functies, en zie je de fout van het sap naar het lampje schuiven.

    **Een voorbeeld uit het dagelijks leven.** Een café geeft je zonder aankondiging een gratis koekje bij de koffie. De eerste keer is het koekje een leuke verrassing. Na een paar weken komt het fijne moment al als je binnenloopt, omdat je weet dat er een koekje komt; het koekje zelf is gewoon wat je verwachtte. En op de dag dat er geen koekje is, merk je het meteen, terwijl koffie zonder koekje eerst heel gewoon was.
    """)
    return


@app.cell(hide_code=True)
def predict_shift():
    guess_shift = L.guess(
        "shift",
        question="Het model van de course ziet een lamp, en 4 stappen later komt er sap, in 60 trials achter elkaar, met η = 0,2. In trial 1 is de fout 1 op het moment van het sap en overal anders 0. Waar is de fout het grootst in trial 60, denk je?",
        options={
            "juice": "Bij het sap, net als in trial 1",
            "right": "Bij de lamp, 4 stappen voor het sap",
            "half": "Bij de lamp en bij het sap, elk ongeveer de helft",
            "none": "Nergens: na 60 trials voorspelt het model alles, dus is geen enkel moment nog een verrassing",
        },
        reveal="**Bij de lamp, 4 stappen voor het sap.** Na 60 trials zijn de V's van de lamp tot en met de stap voor het sap bijna 1: vanaf de lamp verwacht het model sap. Als het sap komt, zakt de V van 1 naar 0, en het sap van 1 maakt dat goed: geen fout. De fout van 1 zit bij de lamp, het eerste moment dat zegt dat er sap komt. Zolang de lamp uit was, voorspelde het model 0, omdat niets aankondigt wanneer de lamp komt, dus blijft de lamp zelf een verrassing. Schultz zag hetzelfde bij de dopamineneuronen van apen.",
    )
    guess_shift
    return (guess_shift,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Een V voor elke stap.** Het Rescorla-Wagner model maakte één voorspelling per trial. Om te zeggen *wanneer* iets een verrassing is, moet een model op elk moment binnen een trial voorspellen. De course telt de momenten van een trial in stappen vanaf de lamp: stap 0 is het moment dat de lamp aangaat, en het sap komt bij stap 4. Elke stap krijgt een eigen V: het sap dat na deze stap nog komt, 0 aan het begin. Zolang de lamp nog niet aan is, is de voorspelling 0, omdat niets aankondigt wanneer de lamp komt. Dit is het TD-model van conditionering van Richard Sutton en Andrew Barto, dat Read Montague, Peter Dayan en Terrence Sejnowski in 1996 voor dopamine gebruikten, met een aparte V voor elke stap na de lamp.

        **De fout bij elke stap.** Bij stap t vergelijkt het model wat het nu heeft met wat de vorige stap voorspelde:
        """),
        L.formula(r"[[loss:\delta_t]] = [[data:r_t]] + [[hyper:\gamma]] [[param:V_t]] - [[param:V_{t-1}]]"),
        mo.md(r"""
        Hier is $r_t$ het sap dat bij stap t komt, 1 of 0, en γ de discount factor. Samen is $r_t + \gamma V_t$ wat stap t over het sap zegt: wat er nu kwam, plus wat er nog komt. $V_{t-1}$ is wat de vorige stap zei. Dit is de TD error van *Leren na elke stap: TD(0)*, δ = R + γV(S′) − V(S), met S de vorige stap en S′ stap t. In deze les is γ 1, tot de demo: binnen een trial van een paar seconden verdisconteert de course niet. Om te leren zet de vorige stap een stap van η keer de fout:
        """),
        L.formula(r"[[param:V_{t-1}]] \leftarrow [[param:V_{t-1}]] + [[hyper:\eta]] \, [[loss:\delta_t]]"),
        mo.md(r"""
        **Drie trials.** Volg het model van de course met η = 0,2. In trial 1 zijn alle V's 0. Bij de lamp verandert er niets: 0 + 0 − 0 = 0. Bij stap 4 komt het sap: 1 + 0 − 0 = 1, een verrassing. Stap 3, de vorige stap, leert daarvan, en in de trials daarna leren de stappen voor stap 3 van stap 3: trial na trial schuift de fout terug. In trial 60 zijn de V's van de lamp tot en met stap 3 bijna 1. Nu zit de fout bij de lamp: 0 + 1 − 0 = 1, de sprong van 0, zolang de lamp uit was, naar 1 erna. Bij het sap is de fout 1 + 0 − 1 = 0: het sap werd verwacht. In trial 61 blijft het sap weg, en bij stap 4 is de fout 0 + 0 − 1 = −1.
        """),
        mo.Html(C.trials_picture(L.lang)),
        mo.md(r"""
        **Wat de neuronen doen.** Leg de drie trials naast de metingen van Schultz, en de trials kloppen met de metingen: een salvo bij het sap in de eerste trials, een salvo bij de lamp zodra de apen het geleerd hadden, en een dip onder het gewone tempo waar het sap had moeten komen. Dopamineneuronen vuren een paar keer per seconde, ook als er niets gebeurt, dus kunnen dopamineneuronen een fout onder 0 doorgeven door minder te vuren. Schultz, Dayan en Montague (1997) concludeerden dat dopamineneuronen een reward prediction error doorgeven: de reward die kwam plus wat er nu verwacht wordt, min wat er eerder verwacht werd. Het sap zelf laat de neuronen niet vuren: zodra het sap voorspeld is, drinkt de aap hetzelfde sap en blijven de neuronen stil. Dopamine is niet het plezier van het sap; dopamine is de verrassing erover.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **dopamine** | | een stof waarmee sommige cellen in de hersenen een signaal doorgeven; dopamineneuronen vuren zoals de fout van een TD-model dat reward voorspelt | |
        | **reward prediction error** (RPE) | | de reward die kwam plus wat er nu verwacht wordt, min wat er eerder verwacht werd; dopamineneuronen lijken deze fout door te geven | `errors` |
        | $\delta_t$ | delta t | de fout bij stap t van een trial: $r_t + \gamma V_t - V_{t-1}$ | `deltas[t]` |
        """),
        L.watch_out("De fout bij stap t hoort bij de voorspelling van de vorige stap: de fout zegt hoe ver $V_{t-1}$ ernaast zat. Als het sap bij stap 4 komt, leert dus $V_3$, niet $V_4$."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Eén trial van het TD-model van conditionering:

    1. Tel de stappen vanaf de lamp, stap 0, en geef elke stap een V, 0 aan het begin. Zolang de lamp nog niet aan is, is de voorspelling 0.
    2. Reken bij elke stap t de fout uit: $\delta_t = r_t + \gamma V_t - V_{t-1}$, met het sap $r_t$ van deze stap.
    3. Elke stap behalve de laatste zet een stap van η keer de fout van de stap erna: $V_{t-1} \leftarrow V_{t-1} + \eta \, \delta_t$. De laatste stap heeft geen stap erna, en de V van de laatste stap blijft 0.
    4. Dan de volgende trial, vanaf stap 2.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    De cel neemt de V's die het model van de course na veel trials heeft, 1 van de lamp tot en met stap 3 en daarna 0, en rekent de fout bij de eerste zes stappen uit met γ = 1: één keer met het sap bij stap 4, en één keer zonder sap. `old` is de V van de vorige stap, 0 bij stap 0, omdat er nog niets voorspeld wordt zolang de lamp uit is. Voer de cel uit.
    """)
    return


@app.cell
def example_errors():
    import numpy as np

    learned = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])   # V van elke stap na veel trials
    juice = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])     # het sap komt bij stap 4
    for t in range(6):
        if t == 0:
            old = 0   # zolang de lamp uit is, wordt niets voorspeld
        else:
            old = learned[t - 1]
        print("stap", t, "met sap", juice[t] + learned[t] - old, "zonder", learned[t] - old)
    return juice, learned, np, old, t


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Bij stap 0 springt de V van 0, zolang de lamp uit was, naar 1: een fout van 1, met of zonder sap, omdat de lamp in beide gevallen komt. Van stap 1 tot en met stap 3 blijft de V 1 en komt er niets: geen fout. Bij stap 4 zakt de V naar 0. Komt het sap, dan maakt de 1 van het sap de daling goed, en is de fout 0; zonder sap is de daling het enige, en is de fout −1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_errors():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. De fout bij elke stap

        `errors(V, rewards, gamma)` geeft de fout δ bij elke stap van één trial terug. `V` is een NumPy-array met de V van elke stap, vanaf de lamp; `rewards` is een array met het sap bij elke stap; `gamma` is γ. `np.zeros(len(V))` maakt `deltas`, een array met een 0 voor elke stap, en de loop `for t in range(len(V)):` vult de array stap voor stap in. `before` houdt de V van de vorige stap vast: 0 zolang de lamp nog niet aan is, en de laatste regel van de loop schuift `before` door naar `V[t]`. Vervang de `...` door de fout bij stap t.

        De check roept `errors(np.array([0.5, 0.6, 0.7, 0.8, 0.0, 0.0]), np.array([0, 0, 0, 0, 1, 0]), 1)` aan: een trial van zes stappen waarin het model een deel van de weg geleerd heeft, met het sap bij stap 4. Dan vijf andere gevallen: de geleerde V's met en zonder sap, V's van 0, het eerste geval met γ 0,9, en een trial met twee keer zoveel sap bij stap 2.
        """),
        L.task("Maak `deltas[t]` de fout bij stap t."),
    ])
    return


@app.cell(hide_code=True)
def exercise_errors():
    box_errors = L.code_box("errors", lines=7)
    box_errors
    return (box_errors,)


@app.cell(hide_code=True)
def check_errors(box_errors):
    result_errors = L.check("errors", box_errors.value)
    result_errors
    return (result_errors,)


@app.cell(hide_code=True)
def hints_errors():
    L.hints(
        "errors",
        [
            "Stap 2 van het recept: `deltas[t]` is het sap van stap `t`, dat in `rewards` staat, plus `gamma` keer de V van stap `t`, min `before`, de V van de vorige stap.",
            "In het eerste geval van de check heeft stap 1 V 0,6, de vorige stap 0,5, en komt er geen sap: 0 + 1 × 0,6 − 0,5 = 0,1. Bij stap 0 is `before` nog 0, dus is de fout de V van de lamp zelf: 0,5.",
            "`        deltas[t] = ___`",
        ],
        why="`deltas[t] = rewards[t] + gamma * V[t] - before` is $\\delta_t = r_t + \\gamma V_t - V_{t-1}$: wat stap t over het sap zegt, min wat de vorige stap zei. Met de geleerde V's geeft de functie 1 bij de lamp, 0 bij het sap, en −1 daar als het sap wegblijft.",
    )
    return


@app.cell(hide_code=True)
def text_after_errors():
    mo.md(r"""
    De afbeelding toont je `errors` bij de drie trials van het idee. In trial 1 staat de fout bij het sap, in trial 60 bij de lamp, en in trial 61 wijst de balk bij stap 4 omlaag: het sap dat de vorige stap beloofde kwam niet.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_learn():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Leren van de fout

        `learn(V, rewards, eta, gamma)` doet één trial leren en geeft de nieuwe V's terug. De eerste regel rekent `deltas` uit met `errors`; het vak gebruikt de `errors` van de course, en deze doet hetzelfde als je eigen `errors`. Binnen één trial wordt elke V één keer gebruikt voordat de V verandert, dus zijn de fouten van de oude V's de fouten om van te leren. `new = np.zeros(len(V))` maakt een array voor de nieuwe V's, en `for t in range(len(V) - 1):` gaat langs elke stap behalve de laatste: de laatste stap heeft geen stap erna, en de V van de laatste stap blijft 0. `eta` is de learning rate η. Vervang de `...` door de fout waar `V[t]` van leert.

        De check roept `learn(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.array([0, 0, 0, 0, 1, 0]), 0.2, 1)` aan: de eerste trial, vanaf V's van 0, met het sap bij stap 4. Dan vier andere gevallen: de V's die een deel van de weg geleerd hebben; de geleerde V's in een trial zonder sap; η 0,5 met γ 0,9; en een trial met twee keer zoveel sap bij stap 2, met η 0,1 en γ 0,8.
        """),
        L.task("Laat elke `new[t]` van de juiste fout leren."),
    ])
    return


@app.cell(hide_code=True)
def exercise_learn():
    box_learn = L.code_box("learn", lines=6)
    box_learn
    return (box_learn,)


@app.cell(hide_code=True)
def check_learn(box_learn):
    result_learn = L.check("learn", box_learn.value)
    result_learn
    return (result_learn,)


@app.cell(hide_code=True)
def hints_learn():
    L.hints(
        "learn",
        [
            "Stap 3 van het recept: `new[t]` is `V[t]` plus `eta` keer de fout die zegt hoe ver `V[t]` ernaast zat. `deltas` houdt de fout van elke stap vast. De fout bij stap t vergelijkt stap t met de vorige stap. Welke stap heeft dus de fout over de voorspelling van stap `t`?",
            "In de eerste trial is alleen het sap bij stap 4 een verrassing: δ is 1 bij stap 4. De vorige stap, stap 3, moet leren: `new` moet 0,2 worden bij stap 3 en 0 blijven bij stap 4.",
            "`        new[t] = V[t] + eta * ___`",
        ],
        why="`new[t] = V[t] + eta * deltas[t + 1]` is $V_{t-1} \\leftarrow V_{t-1} + \\eta \\, \\delta_t$, één stap later geschreven: de fout bij stap t + 1 zegt hoe ver de voorspelling van stap t ernaast zat. In 60 trials schuift de fout van het sap naar de lamp.",
    )
    return


@app.cell(hide_code=True)
def text_after_learn():
    mo.md(r"""
    De afbeelding toont de fout bij elke stap nadat je `learn` zoveel trials geleerd heeft. In trial 1 staat de fout bij het sap. In de trials daarna spreidt de fout zich uit over de stappen voor het sap, omdat elke stap leert van de stap erna, en schuift de fout stap voor stap naar de lamp; in trial 60 is alleen de lamp over. De laatste rij, trial 61 zonder sap, heeft de dip bij stap 4.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### De fout schuift, trial na trial

    De demo doet 60 trials met je `learn` als je `learn` de check haalt, en anders met de `learn` van de course, met η 0,2, en daarna één trial zonder sap. Kies bij welke stap na de lamp het sap komt, 2, 4 of 6, en de discount factor γ. De grafiek toont de fout op het moment van het sap in elke trial, en gestreept de fout bij de lamp; ernaast de V's en de fouten van de laatste trial. Daarna kun je terug door de trials.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    delay_choice = mo.ui.dropdown(options=C.delay_options(L.lang), value=C.number(4, 0, L.lang), label=C.DEMO_TEXT[L.lang]["delay"])
    gamma_choice = mo.ui.dropdown(options=C.gamma_options(L.lang), value=C.number(1, 0, L.lang), label=C.DEMO_TEXT[L.lang]["gamma"])
    run_button = mo.ui.run_button(label="Doe de trials")
    mo.hstack([delay_choice, gamma_choice, run_button], justify="start", gap=1.5, wrap=True)
    return delay_choice, gamma_choice, run_button


@app.cell(hide_code=True)
def demo_trials(box_learn, delay_choice, gamma_choice, result_learn, run_button):
    if not run_button.value:
        _out = mo.md("*Kies de stap van het sap en γ, en druk dan op de knop om de trials te doen.*")
    else:
        _mine = result_learn.state == "correct"
        _learn = run_code(box_learn.value, C.learn_names()).names["learn"] if _mine else C.learn
        _delay, _gamma = delay_choice.value, gamma_choice.value
        _n = C.TRIALS + 1
        _w = C.DEMO_TEXT[L.lang]
        _live = L.live(C.chart_labels(L.lang), _n, role="loss", low=-1, high=1, window=1, xticks=C.demo_xticks(),
                       key=C.chart_key(L.lang))

        def _progress(_t, _demo):
            _live.watch(lambda: C.now_picture(_demo, L.lang))
            _live.reference(_demo.points(0), "", role="loss", replace=True)
            _live.extend([(_t, float(_demo.deltas[_t - 1, _demo.delay]))], _w["learning"].format(t=_t, n=_n))

        _demo = C.demo_run(_learn, _delay, _gamma, show=_progress, pause=0.08)
        if _demo == "error":
            _out = _live.stopped("*Je `learn` gaf een fout, of iets anders dan een getal voor elke stap. Haalt je `learn` de check nog?*")
        else:
            _out = mo.vstack([
                L.explore("rewind", "trial", lambda _v: C.rewind_frame(_demo, int(_v), L.lang), start=_n,
                          values=list(range(_n + 1)), role="",
                          task="Ga naar trial 1, en dan vooruit: bij welke stap is de fout het grootst in trial 10, en in trial 20?"),
                mo.md(C.verdict(_demo, _mine, L.lang)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met γ 1 eindigt de fout bij de lamp na 60 trials op 0,99 of meer, waar het sap ook komt; hoe later het sap, hoe meer trials de fout nodig heeft om terug te schuiven naar de lamp. Met γ onder 1 telt sap verder weg minder, dus voorspelt de lamp er minder van: met γ 0,9 eindigt de fout bij de lamp op 0,81 met het sap bij stap 2, op 0,66 bij stap 4 en op 0,52 bij stap 6, en met γ 0,8 op 0,64, 0,41 en 0,26. Dopamineneuronen doen iets vergelijkbaars: Shunsuke Kobayashi en Schultz (2008) vonden dat het salvo bij een signaal kleiner is naarmate het sap langer op zich laat wachten. In elke instelling zakt de trial zonder sap naar −1 op het moment van het sap. In het model schuift de fout stap voor stap terug naar de lamp; in metingen was dat lang moeilijk te zien, tot een studie bij muizen in 2022 zo'n geleidelijke verschuiving vond (Ryunosuke Amo en collega's).
    """)
    return


@app.cell(hide_code=True)
def text_exercise_more():
    mo.md(r"""
    ### Oefening 3. Meer sap dan verwacht

    Het model van de course heeft in 60 trials geleerd dat de lamp sap bij stap 4 betekent.
    """)
    return


@app.cell(hide_code=True)
def choice_more_cell():
    choice_more = L.choice(
        "more",
        question="In trial 61 komt er twee keer zoveel sap als anders: r = 2 bij stap 4. Wat is de fout bij stap 4?",
        options={
            "none": "0: het sap kwam op het moment dat het model verwachtte",
            "right": "1: één meer dan de vorige stap voorspelde",
            "all": "2: al het sap dat kwam",
            "dip": "−1: het sap kwam, maar niet de hoeveelheid die beloofd was",
        },
        correct="right",
        why={
            "right": "$\\delta_4 = r_4 + \\gamma V_4 - V_3$ = 2 + 0 − 1 = 1: de fout is hoeveel meer er kwam dan voorspeld was, niet hoeveel er kwam. Dopamineneuronen doen hetzelfde: meer sap dan voorspeld geeft een salvo, minder geeft een dip (Philippe Tobler, Christopher Fiorillo en Schultz, 2005). De fout bij de lamp blijft 1, omdat de lamp nog steeds de gewone hoeveelheid sap voorspelt.",
            "none": "Het sap kwam op tijd, maar was het de hoeveelheid die de vorige stap voorspelde? Wat is 2 + 0 − 1?",
            "all": "De fout is wat er kwam min wat er voorspeld was. Wat voorspelde stap 3?",
            "dip": "Een fout onder 0 betekent dat er minder kwam dan voorspeld was. Is 2 minder of meer dan de 1 die door stap 3 voorspeld werd?",
        },
    )
    choice_more
    return (choice_more,)


@app.cell(hide_code=True)
def text_exercise_unblock():
    mo.md(r"""
    ### Oefening 4. Blocking in de hersenen

    Bij blocking, uit *Leren van verrassing: het Rescorla-Wagner model*, voorspelt een geluid het eten al als er een lamp bij komt, en leert de lamp bijna niets. Pascale Waelti, Anthony Dickinson en Schultz (2001) vonden dat dopamineneuronen van apen nauwelijks vuren bij zo'n geblokkeerd signaal, en ook niet bij het eten, omdat het eten verwacht werd.
    """)
    return


@app.cell(hide_code=True)
def choice_unblock_cell():
    choice_unblock = L.choice(
        "unblock",
        question="In een experiment met ratten (Elizabeth Steinberg en collega's, 2013) werden dopamineneuronen aan het vuren gebracht op het moment van het eten, in de trials waarin de lamp bij het geluid kwam. Wat leerden de ratten over de lamp?",
        options={
            "right": "Dat de lamp eten betekent: het salvo werkte als een verrassing",
            "nothing": "Niets: het geluid voorspelde het eten al",
            "sound": "Het geluid verloor de eigen V, omdat het salvo ook na het geluid kwam",
            "below": "Een V onder 0 voor de lamp: het salvo kwam op een moment dat het model niet verwachtte, dus telt het salvo tegen de lamp",
        },
        correct="right",
        why={
            "right": "Het salvo bij het eten is het foutsignaal, dus werkt een salvo dat de onderzoekers daar opwekken als een verrassing: de lamp leerde dat er eten op volgt, hoewel het geluid het eten nog steeds voorspelde. De neuronen werden aangezet met licht via een dunne glasvezel, een techniek die optogenetica heet. Het is een van de sterkste aanwijzingen dat de hersenen niet alleen een reward prediction error maken, maar er ook van leren.",
            "nothing": "Zonder salvo zou de lamp niets leren, zoals bij blocking. Wat vertelt een salvo bij het eten de hersenen meestal?",
            "sound": "Het salvo kwam bij het eten, zoals na een verrassing. Welk signaal had nog iets te leren?",
            "below": "Het salvo kwam bij het eten, het moment waarop een verrassing zou komen. Betekent een salvo meer of minder dan voorspeld was?",
        },
    )
    choice_unblock
    return (choice_unblock,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Leren van verrassing: het Rescorla-Wagner model](/course/open/rl-rescorla-wagner?lang=nl) leerden signalen van de verrassing van een hele trial. Het TD-model van conditionering maakt van deze verrassing een TD error bij elke stap, zoals in [Leren na elke stap: TD(0)](/course/open/rl-td-prediction?lang=nl), en deze fout schuift van het sap naar de lamp terwijl het model leert, net als de salvo's van dopamineneuronen. De volgende les, *RL buiten games*, verlaat de hersenen en kijkt waar RL in de wereld gebruikt wordt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_more, choice_unblock, result_errors, result_learn):
    L.wrap_up(result_errors, result_learn, choice_more, choice_unblock)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
