# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les search-astar. Route: course/route/lessen/overige-takken.yaml. Checks, de doolhoven en het speelblok: checks.py.
# termen:allow A ster

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("search-astar", "nl")
    C = L.checks   # de doolhoven, en de eigen A* en breadth-first search van de course die het speelblok tekenen

    def steps(n):
        """Een aantal stappen, in woorden."""
        return f"{n} stap" if n == 1 else f"{n} stappen"


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een doolhof doorzoeken: breadth-first search](/course/open/search-bfs?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_order():
    quick_order = L.choice(
        "quick-order",
        question="In welke volgorde pakt breadth-first search de paden uit zijn wachtrij?",
        options={
            "newest": "Het nieuwste eerst: het pad dat het laatst binnenkwam",
            "right": "Het oudste eerst: het pad dat het eerst binnenkwam",
            "closest": "Eerst het pad dat het dichtst bij het doel eindigt",
        },
        correct="right",
        why={
            "right": "Zijn frontier is een wachtrij: `popleft()` geeft de paden af in de volgorde waarin ze binnenkwamen, dus op volgorde van hun aantal stappen.",
            "newest": "Het nieuwste pad eerst pakken is depth-first search, en die vindt niet altijd een kortste weg. Zie [de vorige les](/course/open/search-bfs?lang=nl).",
            "closest": "Breadth-first search weet niet waar het doel ligt: hij ziet alleen of een state het doel is. Zie [de vorige les](/course/open/search-bfs?lang=nl).",
        },
    )
    quick_order
    return (quick_order,)


@app.cell(hide_code=True)
def quick_check_expanded():
    quick_expanded = L.choice(
        "quick-expanded",
        question="Breadth-first search pakt een pad naar het doel van 10 stappen uit zijn wachtrij. Welke states heeft hij dan uitgebreid?",
        options={
            "path": "Alleen de states op dat pad",
            "towards": "Alleen de states aan de kant van het doel",
            "right": "Elke state die minder dan 10 stappen van de start ligt, in elke richting",
        },
        correct="right",
        why={
            "right": "Hij probeert de paden op volgorde van hun aantal stappen, dus elke state die dichter bij de start ligt dan het doel, komt er eerder uit.",
            "path": "Hij kent het pad niet voordat hij het gevonden heeft. Zie het speelblok van [de vorige les](/course/open/search-bfs?lang=nl).",
            "towards": "Hij weet niet aan welke kant het doel ligt. Zie het speelblok van [de vorige les](/course/open/search-bfs?lang=nl).",
        },
    )
    quick_expanded
    return (quick_expanded,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Breadth-first search zoekt in elke richting, ook van het doel af. Een routeplanner die zo een heel land doorzocht, zou elke weg bekijken die dichter bij je ligt dan je bestemming. Toch weet je ongeveer waar het doel ligt: kan de search dat gebruiken?

    **Een voorbeeld uit het dagelijks leven.** Je zoekt het station in een stad die je niet kent, en je ziet de toren ervan. Je loopt niet in ringen elke straat om je heen af: je loopt naar de toren toe, en slaat pas af als een straat ophoudt.
    """)
    return


@app.cell(hide_code=True)
def predict_how_many():
    guess_how_many = L.guess(
        "how-many",
        question=f"Denk aan een search die eerst het pad pakt met de minste stappen tot nu toe plus een schatting van de stappen die nog komen. In MAZE breidde breadth-first search {len(C.BFS_DONE)} states uit om een pad van {steps(len(C.BFS_PATH) - 1)} te vinden. Hoeveel denk je dat deze uitbreidt?",
        options={
            "same": "Ongeveer evenveel: zo'n 50",
            "half": "Ongeveer de helft: zo'n 30",
            "path": "Alleen de states van het pad: zo'n 17",
        },
        reveal=f"**{len(C.DONE)}, en hij vindt nog steeds een pad van {steps(len(C.SHORTEST) - 1)}.** Deze search is *A\\**. Door de muren breidt hij meer uit dan de states van het pad: die ziet de schatting niet.",
    )
    guess_how_many
    return (guess_how_many,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Breadth-first search is **uninformed search**: hij weet niets over waar het doel ligt. **Informed search** gebruikt een schatting van hoe ver een state van het doel ligt: een **heuristische functie**, kortweg de heuristiek $h$. In het doolhof telt $h$ de stappen naar het doel alsof er geen muren waren: het aantal rijen plus het aantal kolommen ertussen. Muren kunnen de weg alleen langer maken, dus $h$ zegt nooit meer dan het echte aantal stappen.

        **A\*** is de search van de vorige les met één verandering: hij pakt uit zijn frontier het pad met de kleinste $f$, de som van twee getallen.
        """),
        L.formula(r"f = [[loss:g]] + [[pred:h]]"),
        mo.md(r"""
        $g$ is de path cost tot nu toe, en $h$ de schatting van de stappen die nog komen vanaf waar het pad eindigt, dus $f$ schat de hele weg via dit pad. De frontier is nu een **prioriteitswachtrij**, die altijd het element met het kleinste getal eerst afgeeft. De module `heapq` van Python houdt een lijst als zo'n wachtrij bij: `heapq.heappush(frontier, (f, path))` zet een paar erin, en `heapq.heappop(frontier)` pakt het paar met de kleinste `f` eruit: Python vergelijkt paren eerst op hun eerste waarde.

        Komt er een pad naar het doel uit, dan is zijn $f$ zijn echte lengte, want $h$ is 0 bij het doel. Elk pad dat nog in de wachtrij staat, heeft een $f$ die minstens even groot is, en omdat $h$ nooit te veel zegt, haalt geen ervan het doel in minder stappen: het pad is een kortste. Een state in `done` overslaan kost niets, want één stap verandert deze $h$ hooguit 1, dus het eerste pad naar een state dat eruit komt, is een kortste pad ernaartoe. Een state ver opzij heeft een grote $h$: paden erdoorheen krijgen een $f$ boven die lengte, en komen er nooit uit.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **uninformed search** | | search die niets weet over waar het doel ligt | |
        | **informed search** | | search die schat hoe ver elke state van het doel ligt | |
        | **heuristische functie** | | een schatting van de path cost van een state naar het doel; kortweg de heuristiek | `h(state, goal)` |
        | **prioriteitswachtrij** | | een wachtrij die het element met het kleinste getal eerst afgeeft | `frontier` |
        | `heapq` | heap-q | de prioriteitswachtrij van Python, op een lijst | `heapq.heappop(frontier)` |
        | **A\*** | A ster | search die het pad met de kleinste $f$ eerst pakt | `astar` |
        | $g$ | g | de path cost tot nu toe | `g` |
        | $h$ | h | de schatting van de stappen die nog komen | `h(nxt, goal)` |
        | $f$ | f | $g + h$, de geschatte lengte van de hele weg | `f` |

        Hier staat <span class="c-pred">h</span> voor elke state van MAZE: één stap verandert hem precies 1.
        """),
        C.heuristic("MAZE met h, het aantal rijen plus kolommen tot G, in elk open vakje: 16 linksboven, 14 boven S, 12 rechts van en onder S, 1 boven en onder G."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Beide searches lopen naast elkaar door MAZE, één uitgebreide state per stap. Een grijs vakje is een state die een search uitbreidde, een gestippelde ring een state in zijn frontier, en de dikke ring de state die hij als laatste eruit pakte. Een search die het doel eruit pakte, laat zijn pad zien. Het blok begint bij de laatste stap.
    """)
    return


@app.cell(hide_code=True)
def explore_both_cell():
    _a_end = len(C.ASTAR_STEPS) - 1
    _b_end = len(C.BFS_STEPS) - 1

    def _far(s):
        return max(s["done"].values()) if s["done"] else 0

    def _say(k):
        _b, _a = C.at_step(C.BFS_STEPS, k), C.at_step(C.ASTAR_STEPS, k)
        if k == 0:
            return "Stap 0: beide frontiers bevatten één pad, de start alleen."
        if k < _a_end:
            return (f"Stap {k}: beide searches breidden {k} states uit. Het verst: breadth-first search "
                    f"{steps(_far(_b))} vanaf de start, A\\* {steps(_far(_a))}.")
        if k == _a_end:
            return f"Stap {k}: A\\* pakt het doel eruit. Breadth-first search kwam niet verder dan {steps(_far(_b))} vanaf de start."
        if k < _b_end:
            return f"Stap {k}: A\\* is klaar. Breadth-first search breidde {k} states uit, de verste {steps(_far(_b))} vanaf de start."
        return (f"Stap {k}: breadth-first search pakt het doel nu ook: een pad van {steps(len(_b['path']) - 1)}, na "
                f"{len(_b['done'])} states uit te breiden. A\\* had er {len(_a['done'])} nodig.")

    def _frame(k):
        k = int(k)
        _b, _a = C.at_step(C.BFS_STEPS, k), C.at_step(C.ASTAR_STEPS, k)
        _alt = (f"MAZE twee keer, na stap {k}: breadth-first search met {len(_b['done'])} states uitgebreid, "
                f"A* met {len(_a['done'])}.")
        for _name, _s in (("Breadth-first search", _b), ("A*", _a)):
            if _s["path"] is not None:
                _alt += f" {_name} vond een pad van {steps(len(_s['path']) - 1)}, getekend."
        return C.side_by_side(k, ("Breadth-first search", "A*"), _alt).text, _say(k)

    explore_both = L.explore("both", "stap", _frame, start=_b_end, low=0, high=_b_end, step=1, role="",
                             task="Schuif naar de stap waarop A\\* het doel eruit pakt. Hoe ver was breadth-first search toen?")
    explore_both
    return


@app.cell(hide_code=True)
def text_after_play():
    _k = len(C.ASTAR_STEPS) - 1
    _b = C.at_step(C.BFS_STEPS, _k)["done"]
    mo.md(rf"""
    A\* pakt het doel eruit bij stap {_k}, met een pad even kort als dat van breadth-first search. Die is dan niet verder gekomen dan {steps(max(_b.values()))} vanaf de start. A\* ging op het doel af, en week alleen uit waar een muur in de weg stond.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    A\*, van een start naar een doel:

    1. Zet één paar in de prioriteitswachtrij: het pad met alleen de start, en zijn $f$, $h$ van de start. `done` begint leeg.
    2. Pak het paar met de kleinste $f$ eruit.
    3. Goal test: is de state waar zijn pad eindigt het doel? Dan is dit pad een kortste: stop.
    4. Zit de state nog niet in `done`? Breid hem dan uit: zet hem in `done`, en zet elk pad dat één action langer is in de prioriteitswachtrij, met zijn $f = g + h$: zijn stappen plus $h$ van waar het eindigt.
    5. Ga terug naar stap 2 zolang de prioriteitswachtrij niet leeg is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    SMALL en `_neighbors` zijn die van het uitgewerkte voorbeeld in de vorige les. `_h` is de heuristiek, en `_astar` volgt het recept, met de stappen als commentaar. Daarin laat een `print` elke state zien die hij uitbreidt, na zijn $f$, zijn $g$ en zijn $h$. Voer de cel uit.
    """)
    return


@app.cell
def example_astar():
    import heapq

    # SMALL: '#' is een muur; de start S en het doel G zijn vakjes waar je op kunt staan
    _maze = [
        "#######",
        "#S..#.#",
        "#.#...#",
        "#...#G#",
        "#######",
    ]

    def _neighbors(maze, state):
        # de actions: een stap omhoog, omlaag, naar links of naar rechts, maar niet tegen een muur in
        r, c = state
        result = []
        for nxt in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if maze[nxt[0]][nxt[1]] != "#":
                result.append(nxt)
        return result

    def _h(state, goal):
        # de stappen naar het doel als er geen muren waren: het aantal rijen plus het aantal kolommen ertussen
        return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

    def _astar(maze, start, goal):
        # 1. één paar in de prioriteitswachtrij: f van de start, en het pad met alleen de start
        frontier = [(_h(start, goal), [start])]
        done = set()
        while len(frontier) > 0:
            # 2. het paar met de kleinste f
            f, path = heapq.heappop(frontier)
            state = path[-1]
            # 3. de goal test
            if state == goal:
                return path, done
            # 4. een nieuwe state: breid hem uit, en zet elk langer pad erin, met zijn f
            if state not in done:
                done.add(state)
                print(f, len(path) - 1, _h(state, goal), state)
                for nxt in _neighbors(maze, state):
                    g = len(path)
                    f = g + _h(nxt, goal)
                    heapq.heappush(frontier, (f, path + [nxt]))
        # 5. de prioriteitswachtrij is leeg: er is geen weg naar het doel
        return None, done

    _path, _done = _astar(_maze, (1, 1), (3, 5))
    print("pad:", _path)
    print(len(_path) - 1, "stappen,", len(_done), "states uitgebreid")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `g = len(path)` is het aantal stappen van het langere pad: één meer dan dat van `path`. Hier blijft $f$ 6: $h$ klopt precies op het hele pad. Onder de muur heeft `(2, 1)` ook een $f$ van 6, 1 stap plus een $h$ van 5; bij een gelijke $f$ vergelijkt Python de paden, en het pad langs de bovenkant komt eerst. Dus breidt A\* alleen de 6 states van het pad uit, op G na, waar breadth-first search er 11 uitbreidde.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_astar():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Van breadth-first search naar A\*

        `astar(maze, start, goal)` hieronder is de search van het uitgewerkte voorbeeld zonder de `print`. Vervang de twee `...`: stap 2 van het recept, en $f$ voor het langere pad van stap 4. De check roept `astar(MAZE, (4, 1), (5, 13))` aan, dan de drie doolhoven hieronder, en een start die het doel is. Elk kortste pad telt als goed, ook met andere uitgebreide states waar twee paden dezelfde $f$ hebben.
        """),
        C.mazes("SMALL, 5 rijen bij 7 kolommen: S op (1, 1), G op (3, 5). CLOSED, 6 bij 7: G ingemetseld op (4, 4). TRAP, 7 bij 11: S op (5, 1), G op (3, 9), achter een muur in kolom 7 met alleen een opening in rij 5, dus wie recht op G afgaat, loopt ertegenaan."),
        L.task("Laat `astar` het paar met de kleinste f eruit pakken, en geef elk langer pad zijn f = g + h."),
    ])
    return


@app.cell(hide_code=True)
def exercise_astar():
    box_astar = L.code_box("astar", lines=25)
    box_astar
    return (box_astar,)


@app.cell(hide_code=True)
def check_astar(box_astar):
    result_astar = L.check("astar", box_astar.value)
    result_astar
    return (result_astar,)


@app.cell(hide_code=True)
def hints_astar():
    L.hints(
        "astar",
        [
            "Stap 2 van het recept, en de formule uit *Het idee*.",
            "`heapq.heappop(frontier)` geeft het paar terug dat erin ging, `(f, path)`: geef het twee namen. Voor het langere pad is `g` zijn aantal stappen, en `h(nxt, goal)` zijn schatting.",
            "`f, path = heapq.___(frontier)` en `f = g + ___`.",
        ],
        why="`heapq.heappop` geeft het paar met de kleinste $f$ af: het pad dat in de minste stappen bij het doel lijkt te komen. $f = g + h$ telt de stappen tot nu toe op bij de schatting vanaf `nxt`, waar het langere pad eindigt.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_f():
    mo.md(r"""
    ### Oefening 2. Eén f
    """)
    return


@app.cell(hide_code=True)
def choice_f_cell():
    _s, _g, _h = C.FAR_STATE, C.FAR_G, C.FAR_H
    choice_f = L.choice(
        "f-value",
        question=f"In MAZE eindigt een pad van {steps(_g)} in `{_s}`. G staat op `{C.GOAL}`. Wat is $f$ voor dit pad?",
        options={
            "g": str(_g),
            "h": str(_h),
            "right": str(_g + _h),
        },
        correct="right",
        why={
            "right": f"$g$ = {_g} stappen tot nu toe, en $h$ = {abs(_s[0] - C.GOAL[0])} rij plus {abs(_s[1] - C.GOAL[1])} kolommen ertussen = {_h}. Dus $f$ = {_g} + {_h} = {_g + _h}: de hele weg via dit pad kost minstens {_g + _h} stappen.",
            "h": f"{_h} is alleen $h$. $f$ telt ook de {_g} stappen die het pad al zette.",
            "g": f"{_g} is alleen $g$. $f$ telt ook $h$ mee, de schatting van de stappen die nog komen.",
        },
    )
    choice_f
    return (choice_f,)


@app.cell(hide_code=True)
def text_exercise_large():
    mo.md(r"""
    ### Oefening 3. Een schatting die te veel zegt
    """)
    return


@app.cell(hide_code=True)
def choice_large_cell():
    choice_large = L.choice(
        "too-large",
        question="Om nog minder states uit te breiden, maakt iemand $h$ drie keer zo groot: `3 * h(nxt, goal)`. Wat kan er misgaan?",
        options={
            "none": "De search vindt geen weg naar het doel meer",
            "longer": "De search kan een weg teruggeven die geen kortste is",
            "nothing": "Niets: een grotere schatting is altijd beter",
        },
        correct="longer",
        why={
            "longer": f"Dan weegt $h$ zwaarder dan de gezette stappen, dus de search gaat verder met paden dicht bij het doel, ook na een omweg, terwijl een korter pad in de wachtrij blijft wachten. In TRAP geeft hij dan een weg van {steps(len(C.TOO_LARGE) - 1)} terug, waar {len(C.TRAP_PATH) - 1} genoeg is.",
            "none": "Elk pad gaat nog steeds de wachtrij in, dus er wordt nog steeds een weg naar het doel gevonden; alleen welke er het eerst uit komt, verandert.",
            "nothing": "Een kortste pad is alleen zeker als $h$ nooit meer zegt dan het echte aantal stappen. Zie *Het idee*.",
        },
    )
    choice_large
    return (choice_large,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Met $h$ = 0 is $f$ gewoon $g$, en pakt A\* de paden op volgorde van hun aantal stappen, zoals [breadth-first search](/course/open/search-bfs?lang=nl). Zoeken komt terug in een tak die nog komt, *Zoeken, logica, planning en optimalisatie*, met als project je eigen zoeker en planner. In de volgende les, *Een language model voorspelt het volgende token*, leert een model van alleen tekst welk token er volgt.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_f, choice_large, result_astar):
    L.wrap_up(result_astar, choice_f, choice_large)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
