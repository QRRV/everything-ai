# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-td-prediction. Route: course/route/lessen/reinforcement-learning.yaml. Checks, de willekeurige wandeling van de course, de runs van TD(0) en Monte Carlo, en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-td-prediction", "nl")
    C = L.checks   # de willekeurige wandeling van de course, de runs van TD(0) en Monte Carlo, en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Leren van hele potjes: Monte Carlo prediction](/course/open/rl-monte-carlo-prediction?lang=nl) en [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mc():
    quick_mc = L.choice(
        "quick-mc",
        question="Wanneer kan een agent met Monte Carlo prediction de schatting $V(s)$ bijwerken voor de states van een episode?",
        options={
            "step": "Na elke stap, met alleen de reward van deze ene stap",
            "right": "Pas aan het einde van de episode, als de return van elke state bekend is",
            "ten": "Na elke tien stappen, met de rewards van deze tien stappen, zodat de agent een lange episode in delen leert",
            "never": "Pas als alle episodes gespeeld zijn, allemaal tegelijk",
        },
        correct="right",
        why={
            "right": "De return van een state telt elke reward tot het einde van de episode, dus Monte Carlo moet wachten op het einde. Deze les werkt in plaats daarvan na elke stap bij.",
            "step": "De return telt elke reward vanaf de state tot het einde van de episode, niet alleen de volgende: wanneer is de laatste bekend? Zie [Leren van hele potjes: Monte Carlo prediction](/course/open/rl-monte-carlo-prediction?lang=nl).",
            "ten": "Tien stappen bevatten niet de rewards die daarna komen: van welk getal neemt Monte Carlo het gemiddelde, en wanneer is dat getal compleet? Zie [Leren van hele potjes: Monte Carlo prediction](/course/open/rl-monte-carlo-prediction?lang=nl).",
            "never": "Monte Carlo kan na elke episode bijwerken, zodra de returns van de episode bekend zijn: wanneer is dat? Zie [Leren van hele potjes: Monte Carlo prediction](/course/open/rl-monte-carlo-prediction?lang=nl).",
        },
    )
    quick_mc
    return (quick_mc,)


@app.cell(hide_code=True)
def quick_check_q():
    quick_q = L.choice(
        "quick-q",
        question="Een agent in Q-learning kiest in state $s$ action $a$, krijgt reward $r$ en komt in $s'$. Naar welke waarde schuift $Q(s, a)$ op?",
        options={
            "return": "De return van de hele episode, als de episode voorbij is",
            "reward": "Alleen de reward $r$, omdat $Q$ telt wat de action meteen oplevert",
            "right": "$r + \\gamma \\max_{a'} Q(s', a')$: de reward plus de verdisconteerde grootste $Q$ in $s'$",
            "mean": "Het gemiddelde van alle rewards die de agent tot nu toe in state $s$ kreeg, over alle actions en alle episodes",
        },
        correct="right",
        why={
            "right": "Q-learning leert na elke stap, met de eigen schatting van de state waar de agent terechtkomt. Deze les doet hetzelfde voor $V$ van een vaste policy, zonder de max.",
            "return": "Q-learning wacht niet op het einde van de episode: welke schatting staat er in de plaats van de rewards na de stap? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
            "reward": "$Q(s, a)$ telt ook wat er na de stap komt, verdisconteerd met $\\gamma$: welke schatting zegt dat? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
            "mean": "$Q(s, a)$ gaat over één action in één state, en het target gebruikt één stap: welke getallen gaan erin? Zie [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl).",
        },
    )
    quick_q
    return (quick_q,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Monte Carlo prediction wacht tot een episode voorbij is, want pas dan is de return bekend. Voor een wandeling van een paar stappen is dat prima, maar sommige taken duren heel lang, en sommige houden nooit op, zoals een robot die de hele dag een magazijn opgeruimd houdt. Een andere aanpak is temporal-difference learning, kort TD learning: deze methode werkt de schatting na elke stap bij, met de reward van de stap en de eigen schatting van de volgende state. Veel latere methoden van RL bouwen voort op dit idee, van Q-learning, dat je kent, tot programma's die backgammon en videogames leerden spelen door zelf te spelen. Het idee bleek ook te passen bij onderzoek naar het brein: dopaminecellen geven een signaal dat sterk lijkt op de TD error van deze les.

    **Een voorbeeld uit het dagelijks leven.** Je rijdt naar huis en verwacht dat de rit 30 minuten duurt. Na 5 minuten begint het te regenen en rijdt het verkeer langzamer; je verwacht nu nog 35 minuten, 40 in totaal. Wacht je tot je thuis bent om te leren dat 30 te optimistisch was? Je weet het al: je nieuwe schatting zegt het. Je leert van het verschil tussen twee schattingen, de ene een paar minuten na de andere, niet alleen van de uitkomst aan het einde.
    """)
    return


@app.cell(hide_code=True)
def predict_closer():
    guess_closer = L.guess(
        "closer",
        question="Een wandeling begint in het midden van vijf states op een rij, A tot en met E, en zet willekeurig stappen naar links of rechts tot de wandeling de rij verlaat; rechts eruit levert 1 op, links 0. TD learning en Monte Carlo schatten allebei de waarden van A tot en met E uit dezelfde 100 wandelingen, met dezelfde learning rate, η = 0,1. Welke schattingen zitten na de 100 wandelingen, gemiddeld over 100 runs, dichter bij de echte waarden?",
        options={
            "mc": "De schattingen van Monte Carlo: Monte Carlo leert van returns die echt gebeurd zijn, terwijl TD leert van eigen schattingen, die in het begin niet kloppen",
            "same": "Ongeveer even dichtbij: beide methoden schuiven de schattingen naar dezelfde waarden, dus na 100 wandelingen zitten beide er even dicht bij",
            "right": "De schattingen van TD, zo'n drie keer dichterbij",
            "turn": "Eerst de schattingen van Monte Carlo, omdat TD wandelingen nodig heeft om het nieuws state voor state terug te brengen, later de schattingen van TD",
        },
        reveal="**De schattingen van TD, zo'n drie keer dichterbij.** Na 100 wandelingen met η = 0,1 zitten de schattingen van TD learning gemiddeld op 0,048 van de echte waarden, en de schattingen van Monte Carlo op 0,142; TD zit in 96 van de 100 runs dichterbij, en gemiddeld na elk aantal wandelingen. De schattingen van TD kloppen in het begin niet, maar veranderen weinig van de ene wandeling op de andere, terwijl elke return van Monte Carlo 0 of 1 is.",
    )
    guess_closer
    return (guess_closer,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **De willekeurige wandeling.** Vijf states op een rij, A tot en met E, met aan beide kanten een einde. Elke wandeling begint in C, en gaat bij elke stap één state naar links of naar rechts, elk met een kans van 1/2, tot de wandeling bij een einde komt. De stap naar het rechtereinde levert een reward van 1 op, elke andere stap 0, en $\gamma$ = 1. De waarde van een state is dan de kans om van daaruit rechts te eindigen: vanaf C liggen beide einden even ver weg, dus $V(C) = 1/2$, en elke state verder naar rechts telt er 1/6 bij op, van $V(A) = 1/6$ tot $V(E) = 5/6$. Dit zijn de echte waarden waarmee de course de schattingen vergelijkt; de agent ziet deze waarden nooit. Beide methoden beginnen met een schatting van 0,5 voor elke state. In code zijn de states de getallen 1 tot en met 5, het linkereinde 0 en het rechtereinde 6, waar de waarde 0 blijft, want na een einde komt niets meer.

        **Monte Carlo met een vaste learning rate.** Monte Carlo prediction maakte $V(S)$ het gemiddelde van de returns na $S$; een return $G$ bevat de rewards vanaf $S$ tot het einde van de episode. Hier schuift $V(S)$ bij elke nieuwe $G$ een vast deel $\eta$ van de afstand op naar $G$, zoals $Q$ in Q-learning opschoof naar het target:
        """),
        L.formula(r"[[param:V(S)]] \leftarrow [[param:V(S)]] + [[hyper:\eta]] \left( [[data:G]] - [[param:V(S)]] \right)"),
        mo.md(r"""
        **TD(0).** TD learning wacht niet op $G$. Na elke stap vanuit een state $S$, met reward $R$, naar de volgende state $S'$, neemt TD learning de reward van de stap plus de eigen schatting van de rest, $R + \gamma V(S')$, als target:
        """),
        L.formula(r"[[param:V(S)]] \leftarrow [[param:V(S)]] + [[hyper:\eta]] \left( [[data:R]] + [[hyper:\gamma]] [[param:V(S')]] - [[param:V(S)]] \right)"),
        mo.md(r"""
        Het deel tussen haakjes is de **TD error**, $\delta = R + \gamma V(S') - V(S)$: hoeveel beter of slechter de stap ging dan $V(S)$ voorspelde. Een positieve $\delta$ duwt $V(S)$ omhoog, een negatieve omlaag, en $\eta$ zegt hoe ver: boeken over TD schrijven er α voor. Leren van je eigen schatting van de volgende state heet **bootstrapping**. Q-learning doet aan bootstrapping met $\max_{a'} Q(s', a')$, en policy evaluation deed het met de waarden van alle mogelijke volgende states, met de kansen uit de tabel `T`; TD(0) doet het voor $V$ van een vaste policy, hier de willekeurige stappen van de wandeling, met de ene volgende state die de wandeling echt bereikte, zonder tabel. Alle methoden die leren van het verschil tussen schattingen op twee momenten heten **temporal-difference learning**; **TD(0)** is de methode die één stap vooruitkijkt.

        Bij de start is elke TD error 0, behalve bij de einden: als $V(S)$ en $V(S')$ allebei 0,5 zijn en de reward 0 is, verandert de stap niets. Alleen de laatste stap van een wandeling brengt nieuws, en van daaruit reist het nieuws in latere wandelingen state voor state terug.

        ### Nieuwe woorden en tekens

        | Woord of teken | Zeg je als | Wat het betekent | In code |
        |---|---|---|---|
        | **temporal-difference learning** | | schattingen leren van het verschil tussen een schatting en een betere schatting een stap later | |
        | **TD(0)** | T D nul | TD learning met één stap: het target is $R + \gamma V(S')$ | `td_update` |
        | **TD error** | | hoeveel beter of slechter een stap ging dan voorspeld: $\delta = R + \gamma V(S') - V(S)$ | `delta` |
        | **bootstrapping** | | in RL een schatting bijwerken met behulp van een andere schatting, zoals $V(S')$, in plaats van de echte return | |
        | **willekeurige wandeling** | | een pad van willekeurige stappen, hier elk naar links of rechts met een kans van 1/2 | `walk` |
        | $\delta$ | delta | de TD error | `delta` |
        | $S'$ | S accent | de state na de stap | `s_next` |
        """),
        L.watch_out("TD(0) leert wat een vaste policy waard is, zoals Monte Carlo prediction deed; de agent kiest hier niets. Kiezen met TD komt in de volgende les."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    TD(0) en Monte Carlo op de willekeurige wandeling, allebei met $\eta$ = 0,1 en met dezelfde wandelingen, 100 runs van elk 100 wandelingen. Links de schattingen van run 1 na een aantal wandelingen, in <span class="c-param">oranje</span>: TD(0) doorgetrokken, Monte Carlo gestreept, afgezet tegen de echte waarden van 1/6 tot 5/6. Rechts de gemiddelde afstand tussen de schattingen en de echte waarden over de 100 runs, in <span class="c-loss">paars</span>.
    """)
    return


@app.cell(hide_code=True)
def explore_walks_cell():
    explore_walks = L.explore("play", "wandelingen", lambda _v: C.play_frame(int(_v), L.lang), start=C.N_WALKS,
                              values=C.LOOKS, role="",
                              task="Ga naar 1: welke schattingen veranderde elke methode na de eerste wandeling, en waarom slechts één bij TD(0)? Ga dan stap voor stap naar 100: welke schattingen in run 1 springen heen en weer van stop tot stop?")
    explore_walks
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    De eerste wandeling van run 1 komt langs A, B, C en D en eindigt links. Monte Carlo schuift elke bezochte state op naar de return, 0: A naar 0,365 en B naar 0,266, een tiende van de afstand bij elk bezoek. TD(0) verandert alleen A, naar 0,45: bij elke andere stap waren beide schattingen 0,5 en de reward 0, dus was de TD error 0. Over de 100 runs zit TD(0) na 10 wandelingen gemiddeld op 0,113 van de echte waarden, en Monte Carlo op 0,161; na 100 wandelingen 0,048 en 0,142. In run 1 zit Monte Carlo na 5 wandelingen op 0,073 en na 100 op 0,19: een return van 1 trekt elke state van de wandeling omhoog, een return van 0 trekt alle states omlaag, en $\eta$ = 0,1 schuift de states elke keer een tiende op. Bij TD(0) ligt het target $R + V(S')$ dicht bij de schatting, dus de schattingen van TD(0) bewegen beetje bij beetje.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    TD(0) voor de waarden van een policy:

    1. Geef elke state een schatting $V$, zoals 0,5, en de einden 0.
    2. Begin een wandeling, en zet een stap met de policy: vanuit $S$, met reward $R$, naar $S'$.
    3. Reken de TD error uit, $\delta = R + \gamma V(S') - V(S)$.
    4. Schuif $V(S)$ op met $\eta\,\delta$.
    5. Is $S'$ een einde, dan is de wandeling voorbij: begin de volgende bij stap 2. Anders zet je de volgende stap vanuit $S'$ en ga je verder bij stap 3.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Eén wandeling, C → D → E → rechtereinde, met elke schatting op 0,5, $\eta$ = 0,1 en $\gamma$ = 1. TD(0) werkt na elke stap bij:

    - C naar D, reward 0: $\delta = 0 + 0{,}5 - 0{,}5 = 0$, dus $V(C)$ blijft 0,5.
    - D naar E, reward 0: $\delta = 0 + 0{,}5 - 0{,}5 = 0$, dus $V(D)$ blijft 0,5.
    - E naar het rechtereinde, reward 1: $\delta = 1 + 0 - 0{,}5 = 0{,}5$, dus $V(E) = 0{,}5 + 0{,}1 \cdot 0{,}5 = 0{,}55$.

    Monte Carlo wacht op het einde. De return na C, D en E is telkens 1, dus elke schatting schuift een tiende van de afstand op naar 1: $0{,}5 + 0{,}1 \cdot (1 - 0{,}5) = 0{,}55$, alle drie.

    Dezelfde wandeling nog een keer. TD(0): bij C weer $\delta = 0 + 0{,}5 - 0{,}5 = 0$; bij D $\delta = 0 + 0{,}55 - 0{,}5 = 0{,}05$, dus $V(D) = 0{,}505$: D leert van de nieuwe schatting van E, niet van het einde van de wandeling; bij E $\delta = 1 + 0 - 0{,}55 = 0{,}45$, dus $V(E) = 0{,}595$. Monte Carlo schuift C, D en E elk van 0,55 naar 0,595. De echte waarde van C is 0,5: na twee wandelingen met geluk heeft Monte Carlo C van de echte waarde weggeschoven, en TD(0) liet C met rust, want niets wat TD(0) tot nu toe leerde zegt iets nieuws over C.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_td():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Eén stap van TD(0)

        `td_update(V, s, r, s_next, eta, gamma)` doet stap 3 en 4 van het recept voor één stap. `V` is een lijst van zeven schattingen: het linkereinde, A tot en met E, en het rechtereinde. `s` en `s_next` zijn de states voor en na de stap, als getallen van 0 tot en met 6, `r` is de reward van de stap, en `eta` en `gamma` zijn $\eta$ en $\gamma$. De functie verandert `V[s]` en geeft niets terug: verandert een functie een lijst, dan is de lijst ook veranderd voor de code die de functie aanriep. Vervang de `...` door de TD error `delta`.

        De check begint bij `START`, 0,5 voor A tot en met E en 0 aan beide einden, en roept `td_update` aan voor elke stap van `WALK + WALK`, de twee wandelingen van het uitgewerkte voorbeeld, met $\eta$ = 0,1 en $\gamma$ = 1. Dan vier andere gevallen, elk vanaf `START`: `LEFT_WALK`, C, B, A en links eruit; `WALK + LEFT_WALK + WALK` met $\eta$ = 0,5 en $\gamma$ = 0,9; `LOOP_WALK`, C, D, C, D, E, D, E en rechts eruit, met $\eta$ = 0,2; en `LOOP_WALK + LEFT_WALK` met $\gamma$ = 0,9.
        """),
        L.task("Laat `td_update` de waarde `V[s]` opschuiven met `eta` keer de TD error."),
    ])
    return


@app.cell(hide_code=True)
def exercise_td():
    box_td = L.code_box("td_update", lines=5)
    box_td
    return (box_td,)


@app.cell(hide_code=True)
def check_td(box_td):
    result_td = L.check("td_update", box_td.value)
    result_td
    return (result_td,)


@app.cell(hide_code=True)
def hints_td():
    L.hints(
        "td_update",
        [
            "Stap 3 van het recept en de formule van *Het idee*: `delta` is het target min de oude schatting. Het target is de reward `r` plus `gamma` keer de schatting van de state waar je terechtkwam; de oude schatting is de schatting van de state waar je vandaan kwam. Welke van `V[s]` en `V[s_next]` is welke?",
            "Bij de tweede wandeling van het uitgewerkte voorbeeld, bij D: de reward is 0, de schatting van E is 0,55 en de schatting van D is 0,5, dus de TD error is 0 + 1 · 0,55 − 0,5 = 0,05. In code is D state 4 en E state 5.",
            "`    delta = ___`",
        ],
        why="`delta = r + gamma * V[s_next] - V[s]` is de TD error: de reward plus de verdisconteerde schatting van de volgende state, min wat `V[s]` eerst zei. De regel eronder schuift `V[s]` op met `eta` keer dat verschil.",
    )
    return


@app.cell(hide_code=True)
def text_after_td():
    mo.md(r"""
    De afbeelding laat zien wat je `td_update` leert van de 100 wandelingen van run 1. Na 10 wandelingen beginnen de schattingen te kantelen, laag links en hoog rechts; na 100 liggen de schattingen dicht bij de echte waarden, gemiddeld 0,047 ervandaan.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_mc():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Monte Carlo na een hele wandeling

        Om te vergelijken: Monte Carlo met een vaste learning rate. `mc_update(V, walk, eta, gamma)` krijgt de hele wandeling als de wandeling voorbij is: `walk` is de lijst van de stappen, elk een tuple `(s, r, s_next)`. De loop gaat met `reversed(walk)` van het einde naar het begin door de wandeling; `reversed` geeft de elementen van een lijst in omgekeerde volgorde zonder de lijst te veranderen. Zo rekent de loop de return `G` vanaf elke stap uit, zoals in Monte Carlo prediction: de reward van de stap plus `gamma` keer de return vanaf de volgende stap. Elk bezoek aan een state telt. Vervang de `...` door de nieuwe `V[s]`: opgeschoven naar `G` met `eta` keer de afstand.

        De check begint bij `START` en roept `mc_update` aan na elke wandeling van `[WALK, WALK]`, met $\eta$ = 0,1 en $\gamma$ = 1. Dan vier andere gevallen met `LEFT_WALK` en `LOOP_WALK`, zoals bij oefening 1, ook met $\gamma$ = 0,9.
        """),
        L.task("Laat `mc_update` elke bezochte `V[s]` opschuiven naar de return `G`, met `eta` keer de afstand."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mc():
    box_mc = L.code_box("mc_update", lines=7)
    box_mc
    return (box_mc,)


@app.cell(hide_code=True)
def check_mc(box_mc):
    result_mc = L.check("mc_update", box_mc.value)
    result_mc
    return (result_mc,)


@app.cell(hide_code=True)
def hints_mc():
    L.hints(
        "mc_update",
        [
            "De eerste formule van *Het idee*: `V[s]` schuift op naar de return `G` met `eta` keer de afstand ertussen, en `G` bevat al elke reward vanaf deze stap tot het einde. Wat is deze afstand, en welke van de twee staat voor het minteken?",
            "De eerste wandeling van het uitgewerkte voorbeeld: `G` is 1 na C, D en E, en elke schatting is 0,5, dus de afstand is 1 − 0,5 = 0,5, en een tiende daarvan brengt elke schatting op 0,55.",
            "`        V[s] = ___`",
        ],
        why="`V[s] = V[s] + eta * (G - V[s])` schuift `V[s]` een deel `eta` van de afstand op naar de return `G`: de update van TD(0), met de hele return als target in plaats van de reward plus de schatting van de volgende state.",
    )
    return


@app.cell(hide_code=True)
def text_after_mc():
    mo.md(r"""
    In de afbeelding leert je `mc_update` van dezelfde 100 wandelingen van run 1. Na 100 wandelingen zitten de schattingen van je functie gemiddeld op 0,19 van de echte waarden, vier keer zo ver weg als bij TD(0). De schattingen volgen de laatste paar wandelingen: na 90 wandelingen stond C op 0,94, na 95 op 0,09, en na de laatste twee wandelingen, die allebei rechts eindigen, op 0,65.

    ### Alle runs tegelijk met NumPy

    Eén run zegt weinig. `C.course_errors()` geeft de fout van elke run na elk aantal wandelingen, voor TD(0) en voor Monte Carlo, allebei met $\eta$ = 0,1, als twee arrays met shape `(100, 101)`: een rij per run, een kolom per aantal wandelingen van 0 tot en met 100. `np.mean(_td, axis=0)` neemt het gemiddelde van elke kolom over de 100 runs: de lijn uit het speelblok. `_td[:, 100]` is de kolom na 100 wandelingen, alle runs tegelijk, en `np.std` zegt hoe ver de getallen uiteenlopen. `_td[:, 100] < _mc[:, 100]` is `True` in de runs waar TD(0) dichterbij zit, en `.sum()` telt de `True`s. Voer de cel uit.
    """)
    return


@app.cell
def example_numpy():
    import numpy as np

    _td, _mc = C.course_errors()        # shape (100, 101): een rij per run, een kolom per aantal wandelingen
    _curve_td = np.mean(_td, axis=0)    # het gemiddelde van elke kolom: de lijn uit het speelblok
    _curve_mc = np.mean(_mc, axis=0)
    print("shape:", _td.shape)
    print("na 10 wandelingen, TD(0):", round(float(_curve_td[10]), 3), "Monte Carlo:", round(float(_curve_mc[10]), 3))
    print("na 100 wandelingen, TD(0):", round(float(_curve_td[100]), 3), "Monte Carlo:", round(float(_curve_mc[100]), 3))
    print("spreiding na 100 wandelingen, TD(0):", round(float(np.std(_td[:, 100])), 3), "Monte Carlo:", round(float(np.std(_mc[:, 100])), 3))
    print("runs waarin TD(0) na 100 wandelingen dichterbij zit:", int((_td[:, 100] < _mc[:, 100]).sum()))
    return (np,)


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    Na 100 wandelingen hebben de fouten van de runs een spreiding van 0,023 bij TD(0) en 0,069 bij Monte Carlo, drie keer zo veel, en TD(0) zit in 96 van de 100 runs dichterbij. Na 10 wandelingen is dat in 67 runs zo: in het begin beslist een wandeling met geluk of pech veel, bij beide methoden.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Je functies op 100 runs

    De demo speelt de 100 runs uit het speelblok met je `td_update` en `mc_update`, op dezelfde wandelingen. Waar je functie de check nog niet haalt, neemt de functie van de course het over. Kies $\eta$ voor elke methode: met 0,1 voor beide zie je het speelblok terug; probeer daarna een kleinere $\eta$ voor Monte Carlo, zoals 0,02, en 0,05 voor TD(0), en daarna 0,01 voor beide. De grafiek toont de gemiddelde afstand tot de echte waarden na elke wandeling, TD(0) doorgetrokken en Monte Carlo gestreept; ernaast de huidige schattingen van run 1.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    eta_td = mo.ui.dropdown(options=C.eta_options(L.lang), value=C.eta_text(0.1, L.lang), label=C.DEMO_TEXT[L.lang]["eta_td"])
    eta_mc = mo.ui.dropdown(options=C.eta_options(L.lang), value=C.eta_text(0.1, L.lang), label=C.DEMO_TEXT[L.lang]["eta_mc"])
    play_button = mo.ui.run_button(label="Speel 100 runs met mijn td_update en mc_update")
    mo.hstack([eta_td, eta_mc, play_button], justify="start", gap=1.5, wrap=True)
    return eta_mc, eta_td, play_button


@app.cell(hide_code=True)
def demo_play(box_mc, box_td, eta_mc, eta_td, play_button, result_mc, result_td):
    walked = None
    if not play_button.value:
        _out = mo.md("*Kies η voor elke methode, en druk dan op de knop om beide methoden van de wandelingen te laten leren.*")
    else:
        _mine = [_n for _n, _r in (("td_update", result_td), ("mc_update", result_mc)) if _r.state == "correct"]
        _td = run_code(box_td.value).names["td_update"] if "td_update" in _mine else C.td_update
        _mc = run_code(box_mc.value).names["mc_update"] if "mc_update" in _mine else C.mc_update
        _w = C.DEMO_TEXT[L.lang]
        _now = {}

        def _progress(_n, _run):
            if "run" not in _now:
                _now["chart"].watch(lambda: C.now_picture(_now["run"], _now["n"], L.lang))
            _now["run"], _now["n"] = _run, _n
            _now["chart"].reference([(_k, _run["mc"][_k]) for _k in range(_n + 1)], "", role="loss", replace=True)
            _now["chart"].extend([(_k, _run["td"][_k]) for _k in ([0, 1] if _n == 1 else [_n])], f"**Wandeling {_n} van {C.N_WALKS}**")

        _live = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"], "alt": _w["alt"]}, C.N_WALKS, role="loss",
                       low=0.0, high=0.25, window=1, key=C.demo_key(eta_td.value, eta_mc.value, L.lang))
        _live.line("**Bezig met wandelen** …")
        _now["chart"] = _live
        _r = C.run(_td, _mc, eta_td.value, eta_mc.value, show=_progress)
        if _r == "td":
            _out = _live.stopped("*Je `td_update` gaf een fout tijdens de run, of liet `V` achter zonder getal voor elke state. Haalt je functie de check nog?*")
        elif _r == "mc":
            _out = _live.stopped("*Je `mc_update` gaf een fout tijdens de run, of liet `V` achter zonder getal voor elke state. Haalt je functie de check nog?*")
        elif _r == "slow":
            _out = _live.stopped("*De run duurde langer dan een minuut. Doet je functie meer werk dan nodig is?*")
        else:
            walked = _r
            _theirs = [_n for _n in ("td_update", "mc_update") if _n not in _mine]
            if len(_mine) == 2:
                _who = "je `td_update` en je `mc_update`"
            elif len(_mine) == 1:
                _who = f"je `{_mine[0]}` en de `{_theirs[0]}` van de course"
            else:
                _who = "de `td_update` en `mc_update` van de course, omdat je eigen functies de check nog niet halen"
            _out = mo.vstack([
                L.explore("rewind", "wandelingen", lambda _v: C.rewind_frame(_r, int(_v), L.lang), start=C.N_WALKS,
                          values=C.LOOKS, role="",
                          task="Ga terug naar 0, en dan weer vooruit: na hoeveel wandelingen komt de doorgetrokken lijn van TD(0) los van de gestreepte lijn, en hoe bewegen de schattingen van run 1 onderweg?"),
                mo.md(f"Gespeeld met {_who}, $\\eta$ = {C.eta_text(eta_td.value, 'nl')} voor TD(0) en {C.eta_text(eta_mc.value, 'nl')} voor Monte Carlo. Na 100 wandelingen zitten de schattingen van TD(0) over de 100 runs gemiddeld op {C.number(_r['td'][C.N_WALKS], 3, 'nl')} van de echte waarden, en de schattingen van Monte Carlo op {C.number(_r['mc'][C.N_WALKS], 3, 'nl')}."),
            ])
    _out
    return (walked,)


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met de functies van de course en $\eta$ = 0,02 voor Monte Carlo springen de schattingen veel minder heen en weer, en eindigt Monte Carlo na 100 wandelingen op 0,076; TD(0) met $\eta$ = 0,05 eindigt op 0,031. Een kleinere $\eta$ ruilt snelheid in voor rust, bij beide methoden: met 0,01 voor beide ligt Monte Carlo na 100 wandelingen zelfs voor, 0,088 tegen 0,113, omdat TD(0) het nieuws dan heel langzaam terugbrengt. Maar met elke methode op de eigen beste $\eta$ uit het menu wint TD(0) op deze wandeling, 0,031 tegen 0,076. Niemand heeft bewezen dat TD learning altijd minder episodes nodig heeft dan Monte Carlo, maar bij taken met willekeurige uitkomsten, zoals deze wandeling, is dat meestal zo.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_predictor():
    mo.md(r"""
    ### Oefening 3. Dezelfde episodes, twee antwoorden

    Een agent heeft alleen deze acht episodes gezien, met $\gamma$ = 1: één keer A, reward 0, dan B, reward 0, en het einde; zes keer B, reward 1, en het einde; één keer B, reward 0, en het einde. Beide methoden schatten $V(B)$ = 0,75: zes van de acht episodes vanuit B eindigden met 1. Monte Carlo schat $V(A)$ = 0: de enige return na A was 0. Nu gaat de agent steeds opnieuw door de acht episodes met TD(0) en een kleine $\eta$, tot de schattingen niet meer bewegen.
    """)
    return


@app.cell(hide_code=True)
def choice_predictor_cell():
    choice_predictor = L.choice(
        "predictor",
        question="Waar eindigt $V(A)$ met TD(0)?",
        options={
            "zero": "Op 0, zoals bij Monte Carlo: de enige return na A was 0, en uiteindelijk leert TD(0) dat ook",
            "half": "Op 0,375, halverwege tussen de return van 0 en $V(B)$, omdat TD(0) beide meeweegt",
            "right": "Op 0,75: na A kwam altijd B, met reward 0, dus TD(0) schuift $V(A)$ naar $0 + V(B)$",
            "one": "Op 1, omdat B meestal met 1 eindigt",
        },
        correct="right",
        why={
            "right": "TD(0) ziet vanuit A alleen één stap: naar B, met reward 0. Het target voor A is $0 + V(B)$, en $V(B)$ leert van alle acht episodes. Monte Carlo volgt de ene return na A; TD(0) volgt wat de stappen over elkaar zeggen: komt na A weer B, dan is B volgens alles wat de agent zag 0,75 waard. Zo is de schatting van TD de betere gok voor nieuwe episodes, zolang de taak werkt zoals de stappen laten zien.",
            "zero": "De enige return na A was 0, maar TD(0) kijkt nooit naar returns. Wat is het target van TD(0) voor de stap vanuit A, en waar eindigt de schatting?",
            "half": "TD(0) heeft voor een stap één target, niet twee om tegen elkaar af te wegen: wat is het target voor de stap van A naar B?",
            "one": "Het target van TD(0) voor A is de reward van de stap plus $V(B)$: hoe groot zijn deze twee?",
        },
    )
    choice_predictor
    return (choice_predictor,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Leren van hele potjes: Monte Carlo prediction](/course/open/rl-monte-carlo-prediction?lang=nl) schatte je $V$ uit hele episodes. TD(0) schat dezelfde waarden na elke stap, met bootstrapping: het target is de reward plus de eigen schatting van de volgende state, zoals het target van [Leren van je eigen stappen: Q-learning](/course/open/rl-q-learning?lang=nl), en zoals [Wat is een policy waard? Policy evaluation](/course/open/rl-policy-evaluation?lang=nl) deed met elke mogelijke volgende state en de tabel `T`. TD(0) zegt alleen wat een vaste policy waard is. De volgende les, *Voorzichtig of gedurfd: SARSA en Q-learning*, gebruikt TD om te kiezen: SARSA leert $Q(s, a)$ na elke stap en verbetert de policy onderweg, en je zet SARSA naast Q-learning aan de rand van een klif.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_predictor, result_mc, result_td):
    L.wrap_up(result_td, result_mc, choice_predictor)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
