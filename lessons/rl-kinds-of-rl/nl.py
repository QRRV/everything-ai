# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-kinds-of-rl. Route: course/route/lessen/reinforcement-learning.yaml. Checks, de kaart, de willekeurige wandeling, de drie automaten en de afbeeldingen: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-kinds-of-rl", "nl")
    C = L.checks   # de kaart, de willekeurige wandeling, de drie automaten en de afbeeldingen


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Voorzichtig of gedurfd: SARSA en Q-learning](/course/open/rl-sarsa?lang=nl) en [Beter spelen door ervaring: Monte Carlo control](/course/open/rl-monte-carlo-control?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sarsa():
    quick_sarsa = L.choice(
        "quick-sarsa",
        question="Een agent leert met ε-greedy, ε = 0,1. Welke methode leert de waarden van altijd de beste action nemen, terwijl de agent nog willekeurige stappen zet?",
        options={
            "sarsa": "SARSA, want SARSA kiest $A'$ met ε-greedy",
            "right": "Q-learning, want het target neemt de grootste $Q$",
            "both": "Allebei, want na genoeg episodes leren de twee dezelfde waarden",
            "none": "Geen van beide: de willekeurige stappen komen altijd in de waarden",
        },
        correct="right",
        why={
            "right": "Q-learning is off-policy: het target $R + \\gamma \\max_{a'} Q(S', a')$ bevat de beste action van $S'$, wat de agent ook als volgende doet. SARSA is on-policy: SARSA leert de waarden van ε-greedy zelf.",
            "sarsa": "SARSA zet de $Q$ van de action die de agent echt als volgende neemt in het target, willekeurige stappen meegeteld: van welke policy leert SARSA dan de waarden? Zie [Voorzichtig of gedurfd: SARSA en Q-learning](/course/open/rl-sarsa?lang=nl).",
            "both": "Op het klif lopen de twee na 500 episodes nog steeds verschillende wegen: welke methode leerde de weg van een agent die altijd de beste action neemt? Zie [Voorzichtig of gedurfd: SARSA en Q-learning](/course/open/rl-sarsa?lang=nl).",
            "none": "Een van de twee targets neemt de grootste $Q$ van de volgende state, wat de agent ook als volgende doet: welke? Zie [Voorzichtig of gedurfd: SARSA en Q-learning](/course/open/rl-sarsa?lang=nl).",
        },
    )
    quick_sarsa
    return (quick_sarsa,)


@app.cell(hide_code=True)
def quick_check_mc():
    quick_mc = L.choice(
        "quick-mc",
        question="Monte Carlo control leert $Q(s, a)$ uit gespeelde handen blackjack. Wanneer werkt de methode de $Q$ van de stappen van een hand bij?",
        options={
            "step": "Na elke kaart, vanuit de schatting van de volgende state",
            "right": "Na afloop van de hand, met de return vanaf elke stap",
            "start": "Bij de start, vanuit de regels van blackjack",
            "all": "Pas na alle handen, met het gemiddelde over elke gespeelde hand tot nu toe",
        },
        correct="right",
        why={
            "right": "Monte Carlo wacht tot de episode voorbij is: pas dan is de return $G$ van elke stap bekend, en schuift elke $Q(s, a)$ van de hand richting de eigen $G$.",
            "step": "Na elke stap bijwerken vanuit een schatting van de volgende state doet TD learning: waar wacht Monte Carlo op? Zie [Beter spelen door ervaring: Monte Carlo control](/course/open/rl-monte-carlo-control?lang=nl).",
            "start": "Monte Carlo control krijgt nooit de regels: de methode ziet alleen de handen die de agent speelt. Wat heeft de methode van een hand nodig om te kunnen bijwerken? Zie [Beter spelen door ervaring: Monte Carlo control](/course/open/rl-monte-carlo-control?lang=nl).",
            "all": "De agent verbetert de policy tijdens het spelen, dus kan de methode niet op alle handen wachten: na welk deel is elke return al bekend? Zie [Beter spelen door ervaring: Monte Carlo control](/course/open/rl-monte-carlo-control?lang=nl).",
        },
    )
    quick_mc
    return (quick_mc,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Tot nu toe leerde je acht RL-methoden kennen, van value iteration tot SARSA, en RL heeft er nog veel meer. Papers, libraries en vacatures delen methoden in met een paar woorden: model-free of model-based, value-based of policy-based, on-policy of off-policy. Deze woorden zeggen wat een methode nodig heeft en wat een methode kan: of kennis van de werking van de environment nodig is, of leren van stappen van een oudere policy lukt, en of de methode moet wachten tot het einde van een episode. Met vier vragen plaats je een methode die je nog nooit zag, en weet je wat je kunt verwachten voordat je de details leest.

    **Een voorbeeld uit het dagelijks leven.** Drie manieren om de weg te vinden in een nieuwe stad. Met een plattegrond plan je de hele route aan tafel, voordat je een stap zet. Zonder plattegrond loop je, en onthoud je welke straten je ergens brachten, en hoe goed elke straat was. Of je onthoudt helemaal geen straten, alleen gewoontes: bij een groot plein linksaf. De eerste manier plant met een model van de stad, de tweede leert waarden, en de derde leert direct een policy.
    """)
    return


@app.cell(hide_code=True)
def predict_walks():
    guess_walks = L.guess(
        "walks",
        question="De willekeurige wandeling uit *Leren na elke stap: TD(0)*: vijf states A tot E op een rij; elke wandeling begint in C en stapt naar links of rechts, elk met kans 1/2, tot de wandeling de rij verlaat, en de rij rechts verlaten levert 1 op. Policy evaluation kent deze kansen: de methode heeft het model. Vanaf 0,5 voor elke state, zoals bij TD(0), hoeveel wandelingen zijn nodig voordat elke waarde binnen 0,01 van de echte waarde zit?",
        options={
            "hundred": "Ongeveer 100, zoveel als in de les over TD(0)",
            "thousand": "Ongeveer 1.000: met het model leert elke wandeling meer, maar 0,01 is dichtbij",
            "right": "Geen: de methode heeft geen wandelingen nodig",
            "many": "Meer dan 10.000, want de methode gaat elke mogelijke wandeling langs",
        },
        reveal="**Geen: de methode heeft geen wandelingen nodig.** Policy evaluation wandelt helemaal niet. Met het model weet de methode dat de wandeling vanuit C naar B of naar D gaat, elk met kans 1/2, dus werkt de methode $V(C)$ direct bij vanuit $V(B)$ en $V(D)$. Na 6 sweeps over de vijf states, 30 updates, zit elke waarde binnen 0,01 van de echte waarde, van $V(A) = 1/6$ tot $V(E) = 5/6$. In de les over TD(0) zat TD learning, dat alleen wandelingen ziet, na 100 wandelingen nog gemiddeld op 0,048 van de echte waarden. En elke wandeling langsgaan: alle wandelingen vanuit C van hooguit 33 stappen, 43.046.720 stuks, leggen $V(C)$ slechts vast tussen 0,495 en 0,505.",
    )
    guess_walks
    return (guess_walks,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Vier vragen.** Elk van de acht methoden beantwoordt vier vragen.

        1. **Gebruikt de methode een model van de environment?** Een model zegt voor een state en een action welke volgende states kunnen volgen, met welke kans, en welke reward: de tabel `T` uit policy evaluation is er een. Een methode met een model, gegeven of geleerd, is **model-based RL**. Een methode die alleen leert van stappen die gebeurden, en nooit een volgende state voorspelt, is **model-free RL**.
        2. **Leert de methode waarden of de policy?** Een **value-based methode** leert $V$ of $Q$, en de policy volgt daaruit, zoals ε-greedy op $Q$. Een **policy-based methode** leert de policy zelf: voor elke state de kans op elke action, en kiest met deze kansen, niet vanuit waarden.
        3. **On-policy of off-policy?** Leert de methode over de policy die de agent volgt, zoals SARSA, of over een andere policy, zoals Q-learning?
        4. **Hele episodes of de eigen schattingen?** Monte Carlo wacht tot het einde van de episode en gebruikt de return. TD learning werkt bij na één stap, vanuit de eigen schatting van de volgende state: bootstrapping.

        **De kaart.** Twee vragen over één update tekenen een kaart. Horizontaal: hoe ver kijkt een update, één stap en dan een schatting, of de hele episode? Verticaal: kijkt de update naar alle volgende states met de kansen, waar een model voor nodig is, of naar de ene volgende state die gebeurde? Policy evaluation, policy iteration en value iteration kijken één stap vooruit, naar alle volgende states; samen heten deze drie **dynamic programming**. TD learning kijkt één stap vooruit naar de volgende state die gebeurde, en Monte Carlo naar de hele episode die gebeurde. De vierde hoek, elke volgende state tot helemaal aan het einde, is *alle paden*: zo kom je dicht bij de exacte waarde, maar zelfs de willekeurige wandeling heeft daar miljoenen wandelingen voor nodig.

        **Een policy-based agent.** Alle acht methoden op de kaart zijn value-based. Hier is een policy-based agent, op een taak zonder states. Drie automaten, 0, 1 en 2, betalen 1 munt met kans 0,2, 0,5 en 0,8, en anders niets; de agent kent deze kansen niet. De agent houdt een lijst `p` bij met de kans om aan elke automaat te trekken, bij de start 1/3 elk, en trekt elke beurt aan een automaat die met deze kansen wordt gekozen. Levert een keer trekken een munt op, dan schuift elke kans $\eta$ van de afstand richting het target $t_a$ op: 1 voor de automaat waaraan de agent trok, en 0 voor de andere:
        """),
        L.formula(r"[[param:p_a]] \leftarrow [[param:p_a]] + [[hyper:\eta]] \left( [[data:t_a]] - [[param:p_a]] \right)"),
        mo.md(r"""
        Levert een keer trekken niets op, dan verandert er niets. De targets zijn samen 1, dus blijven de kansen samen 1. De agent trekt steeds vaker aan een automaat die vaak betaalt; zodra de kans van een automaat dicht bij 1 ligt, trekt de agent bijna alleen nog aan deze automaat. De agent schat nooit wat een automaat betaalt: de agent heeft geen waarden, alleen de policy.

        ### Nieuwe woorden en tekens

        | Woord of teken | Zeg je als | Wat het betekent | In code |
        |---|---|---|---|
        | **model-based RL** | | RL die een model van de environment gebruikt, gegeven of geleerd, om te plannen of bij te werken | |
        | **model-free RL** | | RL die alleen leert van stappen die gebeurden, zonder model | |
        | **value-based methode** | | een methode die $V$ of $Q$ leert en daaruit de actions kiest | |
        | **policy-based methode** | | een methode die de policy zelf leert: de kans op elke action | `p` |
        | **dynamic programming** | | policy evaluation, policy iteration en value iteration: updates over alle volgende states, met het model | |
        | $p_a$ | p a | de kans dat de agent aan automaat $a$ trekt | `p[a]` |
        | $t_a$ | t a | het target van $p_a$: 1 voor de automaat waaraan de agent trok, 0 voor de andere | `target` |
        """),
        L.watch_out("Een kans van precies 1 verandert nooit meer: de agent trekt alleen nog aan deze automaat, en een keer trekken aan deze automaat schuift de kans richting 1, waar de kans al is. Gebeurt dat vroeg, bij een automaat die niet de beste is, dan komt de agent daar nooit achter."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    De acht methoden die je tot nu toe kent, één voor één. Links de kaart, met de gekozen methode in <span class="c-param">oranje</span>. Rechts de willekeurige wandeling, met C oranje omlijnd: één update van de gekozen methode verandert de waarde van C, en in inkt staat waar de update naar kijkt: beide volgende states met de kansen, de ene stap die gebeurde, of een hele wandeling.
    """)
    return


@app.cell(hide_code=True)
def explore_map_cell():
    explore_map = L.explore("map", "methode", lambda _v: C.play_frame(int(_v), L.lang), start=1, values=C.STOPS,
                            role="", task="Ga de acht methoden langs. Welke methoden kijken in één update naar B én D, en welke naar slechts één van de twee? Welke methode is als enige off-policy?")
    explore_map
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Alle acht zijn value-based: de methoden leren $V$ of $Q$. De drie methoden van dynamic programming kijken naar B én D, elk met de kans uit het model; de andere vijf kijken alleen naar wat gebeurde, hier de stap naar D. Monte Carlo wacht tot de wandeling de rij heeft verlaten, en TD learning gebruikt meteen de schatting van D. Q-learning is als enige van de acht off-policy. De vierde hoek blijft leeg: *alle paden* langsgaan kan alleen bij heel kleine taken.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een methode op de kaart plaatsen:

    1. Model: gebruikt de methode kennis van welke volgende states op een action volgen, en met welke kans, gegeven of geleerd? Dan is het model-based RL; anders model-free RL.
    2. Waarden of de policy: houdt de methode $V$ of $Q$ bij en kiest de methode daaruit? Dan is het een value-based methode. Houdt de methode een kans per action bij en verandert de methode deze kansen direct? Dan is het een policy-based methode.
    3. Van welke policy de waarden zijn: van de policy die de agent volgt? Dan is de methode on-policy. Van een andere policy? Dan is de methode off-policy.
    4. Wanneer de methode bijwerkt: na de episode, met de return? Hele episodes. Na een stap, met de eigen schatting? De eigen schattingen.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    **Eén keer trekken van de policy-based agent.** De kansen zijn `p` = `[0.2, 0.3, 0.5]` en $\eta$ = 0,1, en de agent trekt aan automaat 2; de automaat betaalt een munt. De targets zijn 0, 0 en 1:

    - automaat 0: $0{,}2 + 0{,}1 \cdot (0 - 0{,}2) = 0{,}2 - 0{,}02 = 0{,}18$
    - automaat 1: $0{,}3 + 0{,}1 \cdot (0 - 0{,}3) = 0{,}3 - 0{,}03 = 0{,}27$
    - automaat 2: $0{,}5 + 0{,}1 \cdot (1 - 0{,}5) = 0{,}5 + 0{,}05 = 0{,}55$

    Samen nog steeds 1: 0,18 + 0,27 + 0,55 = 1. Had automaat 2 niets betaald, dan was `p` gelijk gebleven aan `[0.2, 0.3, 0.5]`.

    **De agent op de kaart.** De agent gebruikt geen model: model-free RL. De agent leert kansen op actions, geen waarden: een policy-based methode. De agent trekt met de kansen die worden bijgewerkt, dus leert de agent over de policy die de agent volgt: on-policy. En elke keer trekken is een hele episode, met de reward van deze keer trekken als return: de agent werkt nooit bij vanuit een eigen schatting.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Met of zonder model

    Zeven agents. Zet elke agent in een groep: model-based met een model dat de agent kreeg, model-based met een model dat de agent leerde uit de eigen pogingen, of model-free.
    """)
    return


@app.cell(hide_code=True)
def exercise_kinds():
    sort_kinds = L.sort("kinds")
    sort_kinds
    return (sort_kinds,)


@app.cell(hide_code=True)
def check_kinds(sort_kinds):
    result_kinds = L.check("kinds", sort_kinds)
    result_kinds
    return (result_kinds,)


@app.cell(hide_code=True)
def hints_kinds():
    L.hints(
        "kinds",
        [
            "Stel bij elke agent vraag 1 van het recept: gebruikt de agent ooit kennis van welke volgende states op een action volgen, en met welke kans? Zo ja, waar komt deze kennis vandaan?",
            "Een opgeschreven tabel, kaart of spelregels zijn een model dat de agent kreeg. Een model kan ook geleerd worden: de agent meet eerst wat de eigen actions doen, en plant dan met wat de agent mat.",
            "De routeplanner, het bordspelprogramma en policy iteration kregen het model. Welke twee van de andere agents leren er een, en welke twee voorspellen nooit waar een action toe leidt?",
        ],
        why="**Gekregen:** de routeplanner (de wegen), het bordspelprogramma (de regels), policy iteration op HALL (de tabel `T`). **Geleerd:** de verwarming, de robotarm. **Model-free:** de blackjack-agent met Monte Carlo control, en Q-learning op het klif: de twee leren alleen wat de eigen pogingen waard waren.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_nudge():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Een policy-based agent

        `nudge(p, action, reward, eta)` geeft de nieuwe lijst met kansen terug na één keer trekken. `p` bevat de kans van elke automaat, `action` is de automaat waaraan de agent trok, `reward` is 1 of 0, en `eta` is $\eta$. Na een reward van 0 geeft de functie `p` onveranderd terug. Anders bouwt de functie de lijst `new`: voor elke automaat `a` is `target` gelijk aan 1 als `a` de automaat is waaraan de agent trok, en anders 0, en `new.append(x)` zet het getal `x` achteraan in `new`. Vervang de `...` door de nieuwe kans van automaat `a`.

        De check roept `nudge([0.2, 0.3, 0.5], 2, 1, 0.1)` aan, de keer trekken uit het uitgewerkte voorbeeld. Daarna vier andere gevallen: een keer trekken zonder munt, $\eta$ = 0,5, vier automaten, en $\eta$ = 0,05.
        """),
        L.task("Laat `nudge` de kans van elke automaat η van de afstand richting het target opschuiven, 1 voor de automaat waaraan de agent trok en 0 voor de andere."),
    ])
    return


@app.cell(hide_code=True)
def exercise_nudge():
    box_nudge = L.code_box("nudge", lines=11)
    box_nudge
    return (box_nudge,)


@app.cell(hide_code=True)
def check_nudge(box_nudge):
    result_nudge = L.check("nudge", box_nudge.value)
    result_nudge
    return (result_nudge,)


@app.cell(hide_code=True)
def hints_nudge():
    L.hints(
        "nudge",
        [
            "Kijk naar de formule uit *Het idee* en de tabel met nieuwe woorden: welk teken is de oude kans, en welk het target? In de code staan de kansen in `p`, is `a` de automaat die aan de beurt is, is $\\eta$ gelijk aan `eta`, en zet `append` een getal achteraan in de lijst `new`.",
            "In het uitgewerkte voorbeeld had automaat 0 kans 0,2 en target 0: $0{,}2 + 0{,}1 \\cdot (0 - 0{,}2)$. Welke twee namen bevatten binnen de loop het target en de oude kans van automaat `a`?",
            "`        new.append(p[a] + eta * (___))`",
        ],
        why="`new.append(p[a] + eta * (target - p[a]))` schuift de kans van automaat `a` $\\eta$ van de afstand richting het target op: omhoog voor de automaat waaraan de agent trok, omlaag voor de andere. De targets zijn samen 1, dus zijn de nieuwe kansen ook samen 1.",
    )
    return


@app.cell(hide_code=True)
def text_after_nudge():
    mo.md(r"""
    In de afbeelding speelt je `nudge` run 1 van de demo, met $\eta$ = 0,1. Na 50 keer trekken is de kans op de automaat die met kans 0,8 betaalt al 0,95, en na 100 keer 0,999: vanaf dan trekt de agent alleen nog aan deze automaat. De demo laat zien hoe vaak dat goed gaat over 200 runs.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Je `nudge` naast een value-based agent

    De demo speelt 200 runs van 1.000 keer trekken met je `nudge` voor de policy-based agent; waar je functie de check nog niet haalt, neemt de functie van de course het over. Ernaast speelt een value-based agent: deze agent houdt het gemiddelde bij van wat elke automaat tot nu toe betaalde, en kiest met ε-greedy, ε = 0,1. De schattingen beginnen allemaal op 0, en bij gelijke schattingen neemt de agent de eerste automaat, de automaat met kans 0,2: bij de start trekt de agent aan deze automaat, tot de willekeurige keuzes een betere automaat vinden. Kies $\eta$ voor de policy-based agent: met 0,1 zie je de run van de course; probeer daarna 0,01, 0,2 en 0,5. De grafiek toont voor elke keer trekken het deel van de 200 runs dat aan de beste automaat trok, de automaat met kans 0,8, als gemiddelde over de laatste 10 keer: policy-based doorgetrokken, value-based gestreept. Ernaast run 1 op dit moment: de kansen van de policy-based agent gevuld, en de schattingen van de value-based agent open.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    eta = mo.ui.dropdown(options=C.eta_options(L.lang), value=C.eta_text(0.1, L.lang), label=C.DEMO_TEXT[L.lang]["eta"])
    play_button = mo.ui.run_button(label="Speel 200 runs met mijn nudge")
    mo.hstack([eta, play_button], justify="start", gap=1.5, wrap=True)
    return eta, play_button


@app.cell(hide_code=True)
def demo_play(box_nudge, eta, play_button, result_nudge):
    pulled = None
    if not play_button.value:
        _out = mo.md("*Kies η, en druk dan op de knop om beide agents aan de drie automaten te laten trekken.*")
    else:
        _mine = result_nudge.state == "correct"
        _fn = run_code(box_nudge.value).names["nudge"] if _mine else C.nudge
        _w = C.DEMO_TEXT[L.lang]
        _now = {"shown": 0}

        def _progress(_n, _run):
            if "run" not in _now:
                _now["chart"].watch(lambda: C.now_picture(_now["run"], _now["n"], L.lang))
            _now["run"], _now["n"] = _run, _n
            if _n % 50 == 0 or _n == 1:
                _now["chart"].reference(C.chart_points(_run, "value", 1, _n), "", role="ink", replace=True)
                _now["chart"].extend(C.chart_points(_run, "policy", _now["shown"] + 1, _n), f"**Beurt {C.number(_n, 0, 'nl')} van {C.number(C.PULLS, 0, 'nl')}**")
                _now["shown"] = _n

        _live = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"], "alt": _w["alt"]}, C.PULLS, role="ink",
                       low=0.0, high=1.0, window=1, key=C.curve_key(L.lang))
        _live.line("**Trekken** …")
        _now["chart"] = _live
        _r = C.run(_fn, eta.value, show=_progress)
        if _r == "nudge":
            _out = _live.stopped("*Je `nudge` gaf een fout tijdens de run, of gaf iets anders dan kansen die samen 1 zijn. Haalt je functie de check nog?*")
        elif _r == "slow":
            _out = _live.stopped("*De run duurde langer dan een minuut. Doet je functie meer werk dan nodig?*")
        else:
            pulled = _r
            _who = "je `nudge`" if _mine else "de `nudge` van de course, omdat je eigen functie de check nog niet haalt"
            _s = C.settled(_r)
            _out = mo.vstack([
                L.explore("rewind", "keer trekken", lambda _v: C.rewind_frame(_r, int(_v), L.lang), start=C.PULLS,
                          values=C.LOOKS, role="",
                          task="Ga terug naar 10, en dan weer vooruit: waar komen de twee lijnen het dichtst bij elkaar, en wanneer loopt de doorgetrokken lijn voorgoed weg? Hoe zien de balken van run 1 er dan uit?"),
                mo.md(f"Run met {_who} en η = {C.eta_text(eta.value, 'nl')}. Na 1.000 keer trekken heeft de beste automaat in {_s[C.BEST]} van de 200 runs een kans boven 0,99, de automaat met kans 0,5 in {_s[1]}, de automaat met kans 0,2 in {_s[0]}, en nog geen automaat in {_s[None]}. Over alle beurten kreeg de policy-based agent {C.number(_r['reward']['policy'], 3, 'nl')} munt per beurt, en de value-based agent {C.number(_r['reward']['value'], 3, 'nl')}."),
            ])
    _out
    return (pulled,)


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Met de `nudge` van de course en $\eta$ = 0,1 eindigen 198 van de 200 runs met een kans boven 0,99 voor de beste automaat, en 2 voor de automaat met kans 0,5. Over de laatste 100 beurten trekt de policy-based agent in 99,0% van de beurten aan de beste automaat, en de value-based agent in 93,1%: met ε = 0,1 kiest de value-based agent in één op de tien beurten een willekeurige automaat. Over alle 1.000 beurten krijgt de policy-based agent 0,782 munt per beurt, en de value-based agent 0,752. Met een grotere $\eta$ kiest de agent sneller vast, en vaker een automaat die niet de beste is: met $\eta$ = 0,5 eindigen 50 runs bij de automaat met kans 0,5 en 7 bij de automaat met kans 0,2, met 0,701 munt per beurt. Met $\eta$ = 0,01 hebben 196 van de 200 runs na 1.000 beurten nog geen vaste keuze. De value-based agent blijft met ε verkennen zolang de agent speelt; de policy-based agent verkent vanzelf steeds minder, terwijl de kansen naar 0 en 1 schuiven. Later in de course komen policy-based methoden voor taken met states; deze methoden leren met gradients.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_paper():
    mo.md(r"""
    ### Oefening 3. Een methode uit een paper

    Een paper beschrijft een agent die videospellen leert spelen vanaf de pixels op het scherm. Niemand geeft de agent de regels van een spel, en de agent leert nooit voorspellen wat een stap doet. De agent bewaart de laatste miljoen stappen in een geheugen. Om de paar zetten kiest de agent willekeurig stappen uit dit geheugen, veel van lang geleden, toen de agent met een oudere policy speelde, en schuift de $Q(s, a)$ van deze stappen richting $r + \gamma \max_{a'} Q(s', a')$.
    """)
    return


@app.cell(hide_code=True)
def choice_paper_cell():
    choice_paper = L.choice(
        "paper",
        question="Waar staat deze agent op de kaart?",
        options={
            "model": "Model-based, value-based, off-policy, vanuit de eigen schattingen: het geheugen met oude stappen is een model van het spel",
            "right": "Model-free, value-based, off-policy, vanuit de eigen schattingen",
            "on": "Model-free, value-based, on-policy, vanuit hele episodes",
            "policy": "Model-free, policy-based, off-policy, vanuit de eigen schattingen",
        },
        correct="right",
        why={
            "right": "De agent voorspelt nooit een volgende state: model-free RL. De agent leert $Q$ en kiest daaruit: een value-based methode. Het target neemt de grootste $Q$ van $s'$, en de agent leert van stappen van oudere policies: off-policy, zoals Q-learning. En de agent werkt bij tijdens het spelen, vanuit de eigen schatting van $s'$, niet vanuit hele episodes. Dit is de agent van Mnih en anderen (2015); de agent leerde 49 Atari-spellen spelen, met Q-learning en een netwerk in plaats van de tabel.",
            "model": "Het geheugen bevat stappen die gebeurden. Kan het geheugen zeggen wat er zou volgen na een action die de agent in deze state nooit nam?",
            "on": "Het target neemt de grootste $Q$ van $s'$, wat de oudere policy daar ook deed, en de agent wacht niet op het einde van een spel: welke methode op de kaart doet hetzelfde?",
            "policy": "Wat leert de agent: een kans voor elke action, of $Q$?",
        },
    )
    choice_paper
    return (choice_paper,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In dit deel van de course bouwde je dynamic programming in [Wat is een policy waard? Policy evaluation](/course/open/rl-policy-evaluation?lang=nl) en [Steeds beter: policy iteration](/course/open/rl-policy-iteration?lang=nl), Monte Carlo in [Beter spelen door ervaring: Monte Carlo control](/course/open/rl-monte-carlo-control?lang=nl), en TD learning in [Leren na elke stap: TD(0)](/course/open/rl-td-prediction?lang=nl) en [Voorzichtig of gedurfd: SARSA en Q-learning](/course/open/rl-sarsa?lang=nl). Deze les zet de methoden met vier vragen op één kaart, en voegt er een policy-based agent aan toe. De volgende les, *Project: een agent voor je eigen grid world*, laat Monte Carlo control, SARSA en Q-learning naast elkaar een grid world leren die je zelf ontwerpt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_paper, result_kinds, result_nudge):
    L.wrap_up(result_kinds, result_nudge, choice_paper)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
