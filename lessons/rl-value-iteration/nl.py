# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les rl-value-iteration. Route: course/route/lessen/overige-takken.yaml. Checks, de werelden en het speelblok: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rl-value-iteration", "nl")
    C = L.checks   # de grid worlds, hun tabellen T, en de eigen value iteration van de course die het speelblok tekent


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [A*: zoeken met richtinggevoel](/course/open/search-astar?lang=nl) en [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_path():
    quick_path = L.choice(
        "quick-path",
        question="Wat geeft A\\* terug als hij het doel uit zijn frontier pakt?",
        options={
            "every": "Een action voor elke state van het doolhof",
            "right": "Een pad: de states van de start naar het doel, één action per stap",
            "far": "Het aantal stappen van elke state naar het doel",
        },
        correct="right",
        why={
            "right": "A\\* geeft het pad terug dat hij eruit pakte: vanaf de start, één action per stap, naar het doel.",
            "every": "A\\* weet alleen wat te doen op het pad dat hij vond; een state buiten dat pad krijgt geen action. Zie [A*: zoeken met richtinggevoel](/course/open/search-astar?lang=nl).",
            "far": "A\\* schat de stappen naar het doel met $h$, maar hij geeft maar één pad terug. Zie [A*: zoeken met richtinggevoel](/course/open/search-astar?lang=nl).",
        },
    )
    quick_path
    return (quick_path,)


@app.cell(hide_code=True)
def quick_check_expected():
    quick_expected = L.choice(
        "quick-expected",
        question="Een spel betaalt 10 punten met kans 0,8, en niets met kans 0,2. Wat is de verwachtingswaarde?",
        options={
            "middle": "5",
            "right": "8",
            "most": "10",
        },
        correct="right",
        why={
            "right": "Elke uitkomst keer zijn kans, opgeteld: 0,8 · 10 + 0,2 · 0 = 8.",
            "middle": "5 is het gemiddelde van 10 en 0, alsof ze even vaak voorkomen. Weeg elke uitkomst met zijn kans. Zie [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
            "most": "10 is de uitkomst die het vaakst voorkomt, maar één spel op de vijf betaalt niets. Zie [Verwachtingswaarde en variantie](/course/open/prob-expectation?lang=nl).",
        },
    )
    quick_expected
    return (quick_expected,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In het doolhof van A\* deed elke action wat hij zei, dus was een pad een plan. Een robot op een natte vloer kan daar niet op rekenen: een stap kan uitglijden. Na een uitglijder staat hij naast zijn pad, en het pad zegt niet meer wat hij moet doen. Hij heeft een keuze nodig voor elke state waar hij terecht kan komen. En door het uitglijden is de ene weg riskanter dan de andere.

    **Een voorbeeld uit het dagelijks leven.** Een glad pad loopt langs een gracht. De kortste weg ligt vlak langs de kant, maar elke stap kan opzij glijden. Je loopt wat verder van het water, ook al is die weg langer.
    """)
    return


@app.cell(hide_code=True)
def predict_way():
    guess_way = L.guess(
        "way",
        question="In de grid world van deze les ligt een rij kuilen onder de weg van S naar het doel G: rechtdoor langs de kuilen kost 5 stappen, over de bovenste rij 7. Elke stap glijdt met kans 0,2 opzij uit, 0,1 naar elke kant, en de discount factor is 0,9. Wat denk je dat beter is?",
        options={
            "straight": "De weg rechtdoor: minder stappen, dus minder verdisconteerd",
            "top": "De bovenste rij: verder van de kuilen",
            "same": "Ze zijn ongeveer even goed",
        },
        reveal=f"**De bovenste rij.** Langs de kuilen valt elke stap er met kans 0,1 in. Vanaf S heeft de weg rechtdoor een verwachte return van {C.number(C.STRAIGHT[C.START], 2, 'nl')}; de bovenste rij {C.number(C.V_STAR[C.START], 2, 'nl')}. Value iteration, hieronder, vindt dat zelf.",
    )
    guess_way
    return (guess_way,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een taak waarin actions kunnen uitglijden, schrijf je op als **Markov decision process**, een MDP: de states, de actions, een discount factor $\gamma$, en de **transition function**, die voor elke state en action zegt wat er kan gebeuren, met zijn kans en zijn reward. In deze grid world gaat een action met kans 0,8 de kant op die hij aangeeft, en glijdt hij met 0,1 naar elke kant ervan uit; een stap tegen de rand blijft waar hij is. Op G landen geeft +1, in een kuil −1, en daar is de taak voorbij; elke andere stap geeft 0. De course schrijft het op als tabel `T`: `T[s][a]` is de lijst met uitkomsten van action `a` in state `s`, elk een tuple `(kans, volgende state, reward)`.

        Wat er daarna kan gebeuren, hangt alleen af van de state en de action, niet van hoe je daar kwam: dat is de *Markov* in de naam. Daarom is één tabel genoeg, en één getal per state. De **value function** $V(s)$ is de return die je vanuit state $s$ mag verwachten als je een [policy](/course/open/rl-what-is-rl?lang=nl) volgt; in deze les altijd de beste, dus: als je zo goed mogelijk handelt. $Q(s, a)$ is hetzelfde als je eerst action $a$ neemt. De **Bellman equation** verbindt ze:
        """),
        L.formula(r"[[pred:Q(s, a)]] = \sum [[data:p]] \left( [[data:r]] + [[hyper:\gamma]] \, [[pred:V(s')]] \right) \qquad [[pred:V(s)]] = \max_a [[pred:Q(s, a)]]"),
        mo.md(r"""
        De som gaat over de uitkomsten van $a$ in $s$: elke uitkomst telt mee met zijn kans $p$, zijn reward $r$ en de waarde van de state $s'$ waar hij landt, een stap later en dus keer $\gamma$. Dat is een verwachtingswaarde, zoals in de instapcheck. **Value iteration** begint overal met $V = 0$ en gebruikt de vergelijking als update, voor elke state, ronde na ronde, tot de waarden bijna niet meer veranderen. De beste policy neemt dan in elke state $s$ de action $\pi(s)$ met de grootste $Q(s, a)$.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **Markov decision process**, MDP | | states, actions, een transition function en een discount factor; wat er daarna gebeurt, hangt alleen af van de state en de action | `T`, `gamma` |
        | **transition function** | | voor elke state en action: wat er kan gebeuren, met zijn kans en reward | `T[s][a]` |
        | **value function**, $V(s)$ | V van s | de return die je vanuit state $s$ mag verwachten als je een policy volgt; hier de beste | `V[s]` |
        | **action-value function**, $Q(s, a)$ | Q van s a | hetzelfde, als je eerst action $a$ neemt | `q` |
        | **Bellman equation** | | $Q$ uit de rewards en de waarden een stap verder; $V$ is de grootste $Q$ | |
        | **value iteration** | | de Bellman equation als update, ronde na ronde, vanaf $V = 0$ | `value_iteration` |
        """),
        L.watch_out("*Waarde* is ook een gewoon woord. Hier is de waarde van een state de return die je ervan mag verwachten: de value function $V(s)$."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Value iteration op GRID, ronde voor ronde, zonder uitglijden en met een kans van 0,2 op uitglijden. Een getal is de <span class="c-pred">waarde</span> van die state na de ronde; een driehoekje wijst naar elke action met de grootste $Q$, geen waar alle vier even goed zijn. De kuilen zijn grijs, G is +1. De lijn eronder laat $V$(S) in elke ronde zien, gestippeld zonder uitglijden, met een streep bij de ronde die je koos.
    """)
    return


@app.cell(hide_code=True)
def explore_rounds_cell():
    _titles = ("Zonder uitglijden", "Met uitglijden 0,2")
    _line = {"title": "V(S) per ronde", "x": "ronde", "y": "V(S)", "still": "zonder uitglijden", "slip": "met uitglijden",
              "alt": "V(S) na elke ronde, van 0 tot {n}: zonder uitglijden 0 tot en met ronde {still_to} en {still_end} vanaf ronde {still_from}; "
                     "met uitglijden onder 0,01 tot en met ronde {tiny_to}, boven 0,1 vanaf ronde {above}, en {slip_end} vanaf ronde {near}. "
                     "De streep staat bij ronde {k}: {a} zonder uitglijden en {b} met uitglijden."}

    def _say(k):
        _a = C.number(C.STEPS["still"][k]["V"][C.START], 2, "nl")
        _b = C.number(C.STEPS["slip"][k]["V"][C.START], 2, "nl")
        if k == 0:
            return "Ronde 0: elke waarde is 0. De driehoekjes laten al de beste action zien waar de volgende stap beslist: naast G de stap erop, naast een kuil de actions die er niet in kunnen vallen."
        if k < 5:
            return f"Ronde {k}: alleen states op hooguit {k} {'stap' if k == 1 else 'stappen'} van G hebben een waarde. S ligt 5 stappen weg: $V$(S) is in beide nog 0."
        return f"Ronde {k}: $V$(S) is {_a} zonder uitglijden en {_b} met uitglijden."

    def _frame(k):
        k = int(k)
        _alt = (f"GRID twee keer na ronde {k} van value iteration, 3 rijen bij 6 kolommen, met de kuilen langs de onderste rij: "
                f"zonder uitglijden is V(S) {C.number(C.STEPS['still'][k]['V'][C.START], 2, 'nl')}, "
                f"met uitglijden {C.number(C.STEPS['slip'][k]['V'][C.START], 2, 'nl')}.")
        if k == C.ROUNDS:
            _alt += " Zonder uitglijden gaat de policy rechtdoor langs de kuilen; met uitglijden gaat hij omhoog naar de bovenste rij."
        return C.side_by_side(k, _titles, _alt, decimal=",").text + C.value_of_s(k, _line, "nl"), _say(k)

    explore_rounds = L.explore("rounds", "ronde", _frame, start=C.ROUNDS, low=0, high=C.ROUNDS, step=1, role="",
                               task="Schuif vanaf ronde 0 naar rechts. In welke ronde komt de waarde van S voor het eerst boven 0,1, zonder en met uitglijden?")
    explore_rounds
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(rf"""
    Zonder uitglijden komt $V$(S) in ronde {C.ABOVE["still"]} boven 0,1: S ligt 5 stappen van G, en na $k$ rondes is een waarde de beste verwachte return van de volgende $k$ stappen, hier 0,9⁴ ≈ 0,66. Met uitglijden pas in ronde {C.ABOVE["slip"]}: de weg rechtdoor is bijna niets waard, en de bovenste rij kost 7 stappen. En de policies verschillen: zonder uitglijden rechtdoor langs de kuilen, met uitglijden omhoog en over de bovenste rij.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Value iteration op een MDP:

    1. Schrijf het MDP op als tabel `T`: voor elke state en action zijn uitkomsten `(kans, volgende state, reward)`.
    2. Geef elke state de waarde 0.
    3. Eén ronde: reken voor elke state de $Q$ van elke action uit, de $p \cdot (r + \gamma \, V(s'))$ van de uitkomsten opgeteld, en maak de grootste $Q$ zijn nieuwe waarde.
    4. Herhaal stap 3 tot de waarden bijna niet meer veranderen.
    5. De policy: in elke state de action met de grootste $Q$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    SMALL is een wereld van 2 rijen bij 3 kolommen: S linksonder, G rechtsboven, en een kuil rechtsonder. `C.table(C.SMALL)` maakt zijn tabel `T`; de eerste `print` laat de uitkomsten van omhoog in `(1, 1)` zien. `T[s][a]` zoekt eerst de state op en dan de action: `T` is een dictionary van dictionaries. `for p, nxt, r in T[s][a]` geeft de drie waarden van elke uitkomst een eigen naam, zoals `x, y = point` dat doet. `max(qs)` is het grootste getal van de lijst `qs`, en `np.argmax(qs)` zijn plek, zoals `.argmax()` die geeft bij een array; `round(x, 3)` rondt `x` af op 3 decimalen, om hem te printen. Voer de cel uit.
    """)
    return


@app.cell
def example_value_iteration():
    import numpy as np

    # 1. het MDP van SMALL als tabel: T[state][action] is de lijst met uitkomsten,
    #    elk als (kans, volgende state, reward)
    _T = C.table(C.SMALL)
    print(_T[(1, 1)]["up"])

    def _value_iteration(T, gamma, rounds):
        # 2. elke state begint met de waarde 0
        V = {}
        for s in T:
            V[s] = 0
        pi = {}
        for k in range(rounds):
            new = {}
            for s in T:
                # 3. de Q van elke action: p * (r + gamma * V[nxt]) van zijn uitkomsten, opgeteld
                actions = []
                qs = []
                for a in T[s]:
                    q = 0
                    for p, nxt, r in T[s][a]:
                        q = q + p * (r + gamma * V[nxt])
                    actions.append(a)
                    qs.append(q)
                # 3. de nieuwe waarde is de grootste Q; 5. de policy neemt de action die hem geeft
                new[s] = max(qs)
                pi[s] = actions[np.argmax(qs)]
            V = new
            # 4. een ronde is klaar: de waarden van (0, 0), (0, 1), (1, 0) en (1, 1)
            print(k + 1, round(V[(0, 0)], 3), round(V[(0, 1)], 3),
                  round(V[(1, 0)], 3), round(V[(1, 1)], 3))
        return V, pi

    _V, _pi = _value_iteration(_T, 0.9, 12)
    print(_pi[(0, 0)], _pi[(0, 1)], _pi[(1, 0)], _pi[(1, 1)])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Omhoog in `(1, 1)` landt met kans 0,8 op `(0, 1)`, glijdt met 0,1 naar links uit naar S, en met 0,1 naar rechts de kuil in, reward −1. In ronde 1 krijgt alleen `(0, 1)` een waarde: rechts landt met kans 0,8 op G, dus 0,8. Elke ronde komen de waarden een state verder en veranderen ze minder; na 12 rondes minder dan 0,001. De policy is rechts, rechts, omhoog, en in `(1, 1)` links: terug naar S, niet omhoog. Omhoog komt vaker op `(0, 1)`, maar kan de kuil in glijden; links kan dat niet. Met de waarden na 12 rondes heeft links een $Q$ van 0,66 en omhoog 0,64.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_vi():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Je eigen value iteration

        `value_iteration(T, gamma, rounds)` hieronder is de functie van het uitgewerkte voorbeeld, zonder de `print`. De check roept `value_iteration(T, 0.9, 30)` aan op de tabel `T` van GRID, waar $V$(S) op 0,42 moet uitkomen, en ook op `T_SMALL` en `T_BRIDGE`, de twee werelden hieronder, met andere discount factors en minder rondes. Waar twee actions even goed zijn, telt elk van de twee als goed.
        """),
        C.worlds("Twee kleine grid worlds, elk met de nummers van zijn rijen en kolommen. SMALL: 2 rijen bij 3 kolommen, S op rij 1, kolom 0, G op rij 0, kolom 2, een kuil op rij 1, kolom 2. BRIDGE: 2 rijen bij 4 kolommen, S op rij 0, kolom 0 en G op rij 0, kolom 3, met kuilen langs de hele onderste rij.", decimal=","),
        L.task("Laat `value_iteration` de p · (r + γ V(s′)) van elke uitkomst bij `q` optellen, en neem de action met de grootste Q als policy."),
    ])
    return


@app.cell(hide_code=True)
def exercise_vi():
    box_vi = L.code_box("value-iteration", lines=21)
    box_vi
    return (box_vi,)


@app.cell(hide_code=True)
def check_vi(box_vi):
    result_vi = L.check("value-iteration", box_vi.value)
    result_vi
    return (result_vi,)


@app.cell(hide_code=True)
def hints_vi():
    L.hints(
        "value-iteration",
        [
            "Stap 3 en 5 van het recept, en de Bellman equation van *Het idee*.",
            "Elke uitkomst telt mee met zijn kans, keer zijn reward plus de verdisconteerde waarde van `nxt`. Voor de policy: `np.argmax(qs)` is de plek van de grootste Q, en `actions` heeft op elke plek de action.",
            "`q = q + p * (___ + gamma * V[___])` en `pi[s] = actions[np.___(qs)]`.",
        ],
        why="`q` telt de uitkomsten van één action op, elk gewogen met zijn kans: een verwachtingswaarde. De grootste `q` is de nieuwe waarde van de state; `actions[np.argmax(qs)]` is de action die hem geeft, de policy.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_neighbor():
    mo.md(r"""
    ### Oefening 2. Naar de grootste waarde?
    """)
    return


@app.cell(hide_code=True)
def choice_neighbor_cell():
    _q = C.Q_STAR[C.START]
    choice_neighbor = L.choice(
        "neighbor",
        question=f"In GRID met uitglijden heeft de state rechts van S de waarde {C.number(C.V_STAR[(1, 1)], 2, 'nl')}, de state boven S {C.number(C.V_STAR[(0, 0)], 2, 'nl')}. Welke action neemt de policy in S?",
        options={
            "right": "Rechts: naar de grootste waarde",
            "up": "Omhoog: zijn Q is groter, omdat een stap naar rechts een kuil in kan glijden",
            "left": "Links, tegen de rand: daar kan hij niet vallen",
        },
        correct="up",
        why={
            "up": f"$Q$(S, omhoog) is {C.number(_q['up'], 2, 'nl')}, $Q$(S, rechts) maar {C.number(_q['right'], 2, 'nl')}. De policy vergelijkt de $Q$ van elke action, met al zijn uitkomsten, niet de waarden van de buren.",
            "right": f"De waarde van een buur telt alleen voor de uitkomst die daar landt. Rechts glijdt ook met kans 0,1 omlaag een kuil in: $Q$(S, rechts) is {C.number(_q['right'], 2, 'nl')}, tegen {C.number(_q['up'], 2, 'nl')} voor omhoog. Zie de Bellman equation in *Het idee*.",
            "left": f"Links glijdt omhoog of omlaag uit, en omlaag is een kuil: $Q$(S, links) is {C.number(_q['left'], 2, 'nl')}. Reken de uitkomsten van elke action uit, zoals in *Het idee*.",
        },
    )
    choice_neighbor
    return (choice_neighbor,)


@app.cell(hide_code=True)
def text_exercise_markov():
    mo.md(r"""
    ### Oefening 3. Nog steeds een MDP?
    """)
    return


@app.cell(hide_code=True)
def choice_markov_cell():
    choice_markov = L.choice(
        "markov",
        question="Stel dat een vakje van GRID gladder wordt naarmate je er vaker op staat: vanaf de derde keer glijdt een stap vanaf dat vakje uit met kans 0,5 in plaats van 0,2. Is de state (rij, kolom) dan nog genoeg voor een MDP?",
        options={
            "yes": "Ja: elke action heeft nog steeds zijn kansen",
            "count": "Nee: wat er daarna gebeurt, hangt ook af van hoe vaak je eerder op het vakje stond, dus dat hoort in de state",
            "never": "Nee: een taak waarin actions kunnen uitglijden, is nooit een MDP",
        },
        correct="count",
        why={
            "count": "Een MDP vraagt dat wat er daarna gebeurt alleen afhangt van de state en de action. Hier hangt het ook af van de weg waarlangs je kwam, dus moet de state bijhouden hoe vaak je op elk vakje stond.",
            "yes": "De kansen zijn er, maar ze hangen af van hoe vaak je op het vakje stond, en dat zegt (rij, kolom) niet. Zie het MDP in *Het idee*.",
            "never": "In GRID glijden stappen uit, en het is een MDP: de kansen hangen alleen af van de state en de action. Zie het MDP in *Het idee*.",
        },
    )
    choice_markov
    return (choice_markov,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Als stappen kunnen uitglijden, is een pad van [A*: zoeken met richtinggevoel](/course/open/search-astar?lang=nl) geen plan meer: de Bellman equation geeft elke state een waarde, en de policy een action voor elke state, ook naast de beste weg. Value iteration heeft de hele tabel `T` nodig: elke kans en elke reward. In de volgende les, *Leren van je eigen stappen: Q-learning*, kent de agent `T` niet, en leert hij $Q(s, a)$ van de stappen die hij zet.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_markov, choice_neighbor, result_vi):
    L.wrap_up(result_vi, choice_neighbor, choice_markov)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
