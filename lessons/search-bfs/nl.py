# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les search-bfs. Route: course/route/lessen/overige-takken.yaml. Checks, de doolhoven en het speelblok: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("search-bfs", "nl")
    C = L.checks   # de doolhoven, en de eigen breadth-first search van de course die het speelblok tekent

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

    Twee korte vragen over [Leren van rewards](/course/open/rl-what-is-rl?lang=nl) en [De volledige backward pass, in de goede volgorde](/course/open/nn-micrograd-backward?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_action():
    quick_action = L.choice(
        "quick-action",
        question="Een robot in een doolhof kan een stap omhoog, omlaag, naar links of naar rechts zetten, en krijgt een reward als hij bij de uitgang komt. Wat is \"een stap naar links\" in de woorden van *Leren van rewards*?",
        options={
            "reward": "Een reward",
            "environment": "De environment",
            "right": "Een action",
        },
        correct="right",
        why={
            "right": "De robot kiest hem, en hij verandert waar de robot is: een action.",
            "reward": "Een reward is het getal dat de environment na een stap teruggeeft; de stap zelf is wat de robot kiest. Zie [Leren van rewards](/course/open/rl-what-is-rl?lang=nl).",
            "environment": "De environment is het doolhof om de robot heen; de stap is wat de robot erin doet. Zie [Leren van rewards](/course/open/rl-what-is-rl?lang=nl).",
        },
    )
    quick_action
    return (quick_action,)


@app.cell(hide_code=True)
def quick_check_seen():
    quick_seen = L.choice(
        "quick-seen",
        question="Wat laat dit programma zien?\n\n```python\nseen = set()\nfor v in [\"a\", \"b\", \"a\", \"c\", \"b\"]:\n    seen.add(v)\nprint(len(seen))\n```",
        options={
            "none": "`0`",
            "right": "`3`",
            "all": "`5`",
        },
        correct="right",
        why={
            "right": "`seen.add(v)` zet `v` in de set, en een waarde die er al in zit, blijft er één keer in: `a`, `b` en `c`.",
            "all": "Een set bevat elke waarde maar één keer: de tweede `a` en de tweede `b` zitten er al in. Zie de set `seen` in [De volledige backward pass, in de goede volgorde](/course/open/nn-micrograd-backward?lang=nl).",
            "none": "`seen` begint leeg, maar `seen.add(v)` zet elke waarde erin. Zie de set `seen` in [De volledige backward pass, in de goede volgorde](/course/open/nn-micrograd-backward?lang=nl).",
        },
    )
    quick_seen
    return (quick_seen,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    In [Leren van rewards](/course/open/rl-what-is-rl?lang=nl) weet de agent niet waar een action toe leidt, dus moet hij het proberen. Vaak weet je het wel: je hebt de plattegrond van een stad, of de regels van een puzzel. Dan kan een agent nadenken voordat hij iets doet: de actions eerst in zijn hoofd uitproberen, en pas dan de beste weg nemen. Dat is **zoeken**, een van de oudste ideeën van AI, en wat een routeplanner doet.

    **Een voorbeeld uit het dagelijks leven.** Giet water in de ingang van een doolhof. Het stroomt alle gangen tegelijk in, elke seconde een stukje verder. Het eerste water dat bij de uitgang komt, kwam langs de kortste weg.
    """)
    return


@app.cell(hide_code=True)
def predict_first_path():
    guess_first_path = L.guess(
        "first-path",
        question="Een search probeert eerst elk pad van 1 stap, dan elk pad van 2 stappen, dan van 3, enzovoort, tot een pad bij het doel komt. Wat geldt volgens jou voor dat eerste pad naar het doel?",
        options={
            "luck": "Het is de weg die de search toevallig als eerste probeerde; er kan een kortere zijn",
            "fewest": "Om het te vinden, bekeek de search zo weinig mogelijk vakjes",
            "shortest": "Het is een kortste weg naar het doel",
        },
        reveal="**Het is een kortste weg.** Elk korter pad kwam ervoor, en geen daarvan kwam bij het doel. Deze search is *breadth-first search*, de search van deze les.",
    )
    guess_first_path
    return (guess_first_path,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Om te zoeken, schrijf je de taak eerst op als **search problem**: de **states** waarin hij kan zijn, de state waarin je begint, de actions in elke state en waar elke action naartoe leidt, een **goal test**, en de **path cost**. In het doolhof is een state het vakje waarin je staat, als tuple `(rij, kolom)`; de actions zijn een stap omhoog, omlaag, naar links of naar rechts, maar niet tegen een muur in; en de path cost van een pad is zijn aantal stappen, waar een routeplanner minuten zou kunnen tellen. Een oplossing is een pad van de start naar een doel, één action per stap.

        Een search houdt de paden bij die hij vond maar nog niet verder volgde: de **frontier**. Steeds weer pakt hij één pad eruit om de state waar het eindigt **uit te breiden**: hij zet elk pad dat één action langer is in de frontier. Welk pad hij als eerste pakt, is wat de ene search van de andere onderscheidt. **Breadth-first search** pakt het oudste, het pad dat het eerst binnenkwam: zijn frontier is een **wachtrij**, zoals bij de kassa van een winkel. Een set `done` onthoudt welke states hij uitbreidde, zodat hij er nooit een twee keer uitbreidt en nooit in rondjes loopt; `state not in done` vraagt of een state er nog niet in zit, het omgekeerde van `in`.

        ### Nieuwe woorden en tekens

        | Woord | Wat het betekent | In code |
        |---|---|---|
        | **state** | één situatie van de taak; in het doolhof het vakje waarin je staat | `state = (4, 1)` |
        | **search problem** | een taak als states, een start, de actions in elke state, een goal test en een path cost | |
        | **goal test** | de vraag of een state een doel is | `state == goal` |
        | **path cost** | wat een pad kost; hier het aantal stappen | `len(path) - 1` |
        | **frontier** | de gevonden paden die nog niet zijn uitgebreid | `frontier` |
        | **uitbreiden** | elk pad dat één action langer is in de frontier zetten | |
        | **wachtrij** | een rij waarin wat het eerst binnenkomt, er ook het eerst uit gaat | |
        | **breadth-first search** | search die het oudste pad uit zijn frontier eerst pakt | `bfs` |
        | `deque` | de wachtrij van Python, uit `collections`: `append` zet achteraan, `popleft()` pakt van voren | `frontier.popleft()` |

        Een pad is een lijst van zijn states, de start meegeteld, dus zijn aantal stappen is `len(path) - 1`.
        """),
        L.watch_out("*State* betekent elders iets anders, zoals de hidden state van een recurrent neural network, een vector van getallen. Hier is een state, net als in reinforcement learning, één situatie van de taak."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Breadth-first search in het doolhof van deze les, stap voor stap. S is de start en G het doel; de getallen boven en links zijn de kolommen en de rijen, zoals de code ze telt. Een vakje met een getal is een uitgebreide state, en het getal is zijn <span class="c-loss">aantal stappen</span> vanaf de start; een gestippelde ring is een state in de frontier, en de dikke ring de state die er als laatste uit is gepakt.
    """)
    return


@app.cell(hide_code=True)
def explore_steps_cell():
    _last = len(C.STEPS) - 1

    def _say(k):
        _s = C.STEPS[k]
        if k == 0:
            return "Stap 0: in de wachtrij staat één pad, met alleen de start erin."
        if _s["path"] is not None:
            return (f"Stap {k}: het pad dat hij eruit pakt, eindigt in het doel: {steps(len(_s['path']) - 1)}. Elk korter pad kwam "
                    f"er eerder uit, dus dit is een kortste. De search breidde {len(_s['done'])} states uit.")
        _far = sorted(set(_s["frontier"].values()))
        _ring = steps(_far[0]) if len(_far) == 1 else ", ".join(str(_d) for _d in _far[:-1]) + f" of {steps(_far[-1])}"
        return (f"Stap {k}: de search breidt `{_s['mark']}` uit, {steps(_s['done'][_s['mark']])} vanaf de start, en zet de paden "
                f"naar zijn buren achter in de wachtrij. In de frontier staan {len(_s['frontier'])} states, {_ring} vanaf de start.")

    def _frame(k):
        k = int(k)
        _s = C.STEPS[k]
        _alt = (f"Het doolhof van deze les, {len(C.MAZE)} rijen bij {len(C.MAZE[0])} kolommen, na stap {k} van breadth-first search: "
                f"{len(_s['done'])} {'state' if len(_s['done']) == 1 else 'states'} uitgebreid, {len(_s['frontier'])} in de frontier.")
        if _s["path"] is not None:
            _alt += f" Het pad van {steps(len(_s['path']) - 1)} van S naar G is getekend."
        return C.picture(k, _alt).text, _say(k)

    explore_steps = L.explore("steps", "stap", _frame, start=20, low=0, high=_last, step=1, role="",
                              task="Schuif naar de laatste stap. Hoe lang is het pad dat de search vindt, en hoeveel states breidde hij uit om het te vinden?")
    explore_steps
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(rf"""
    De frontier is een ring die naar buiten groeit: zijn states liggen allemaal *d* of *d* + 1 stappen van de start. Om het pad van {steps(len(C.SHORTEST) - 1)} te vinden, breidde de search {len(C.DONE)} van de {C.open_cells(C.MAZE)} states uit, ook die aan de kant die van het doel af ligt.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Breadth-first search, van een start naar een doel:

    1. Zet één pad in de wachtrij: de start alleen. De set `done` begint leeg.
    2. Pak het oudste pad eruit, van voren uit de wachtrij. Zijn laatste state is waar het eindigt.
    3. Goal test: is die state het doel? Dan is dit pad een kortste: stop.
    4. Zit de state nog niet in `done`? Breid hem dan uit: zet hem in `done`, en zet elk pad dat één action langer is achter in de wachtrij.
    5. Ga terug naar stap 2 zolang de wachtrij niet leeg is. Een lege wachtrij betekent dat er geen weg naar het doel is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een klein doolhof, als lijst van strings: `#` is een muur, `S` de start en `G` het doel. `_neighbors` geeft de states die één action verderop liggen, en `_bfs` volgt het recept; zijn `print` laat van elk pad dat hij vóór het doel eruit pakt, het aantal stappen en de laatste state zien. `path + [nxt]` maakt een nieuwe lijst, het pad met `nxt` aan het eind, zoals `+` twee strings aan elkaar plakt; `path` zelf blijft zoals hij was. Aan het eind doorzoekt `_bfs` een tweede doolhof, waarin een muur tussen S en G staat. Voer de cel uit.
    """)
    return


@app.cell
def example_bfs():
    from collections import deque

    # het doolhof: '#' is een muur; de start S en het doel G zijn vakjes waar je op kunt staan
    _maze = [
        "#######",
        "#S..#.#",
        "#.#...#",
        "#...#G#",
        "#######",
    ]
    # een tweede doolhof, waarin een muur tussen S en G staat
    _closed = [
        "#####",
        "#S#G#",
        "#####",
    ]

    def _neighbors(maze, state):
        # de actions: een stap omhoog, omlaag, naar links of naar rechts, maar niet tegen een muur in
        r, c = state
        result = []
        for nxt in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if maze[nxt[0]][nxt[1]] != "#":
                result.append(nxt)
        return result

    def _bfs(maze, start, goal):
        # 1. één pad in de wachtrij, de start alleen; er is nog niets uitgebreid
        frontier = deque([[start]])
        done = set()
        while len(frontier) > 0:
            # 2. het oudste pad, van voren uit de wachtrij
            path = frontier.popleft()
            state = path[-1]
            # 3. de goal test
            if state == goal:
                return path, done
            # 4. een nieuwe state: breid hem uit, en zet elk langer pad achteraan
            if state not in done:
                done.add(state)
                print(len(path) - 1, state)
                for nxt in _neighbors(maze, state):
                    frontier.append(path + [nxt])
            else:
                # 4. een state die al is uitgebreid: sla hem over
                print(len(path) - 1, state, "al uitgebreid")
        # 5. de wachtrij is leeg: er is geen weg naar het doel
        return None, done

    _path, _done = _bfs(_maze, (1, 1), (3, 5))
    print("pad:", _path)
    print(len(_path) - 1, "stappen,", len(_done), "states uitgebreid")
    print("pad:", _bfs(_closed, (1, 1), (1, 3))[0])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De aantallen stappen vooraan de regels worden nooit kleiner: de wachtrij geeft de paden af op volgorde van hun aantal stappen. Een regel met `al uitgebreid` eindigt in een state die al eerder is uitgebreid, zoals de weg terug naar `(1, 1)`: de search slaat hem over. Het doel komt eruit na 6 stappen, een kortste pad; alle 11 andere states waren toen al uitgebreid, ook het doodlopende `(1, 5)`, even ver van de start als het doel. In het tweede doolhof is de wachtrij na de start leeg, dus geeft `_bfs` `None`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bfs():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Je eigen breadth-first search

        `bfs(maze, start, goal)` hieronder is de search van het uitgewerkte voorbeeld zonder de `print`-regels. Vervang de twee `...` door stap 2 en 4 van het recept. De check roept `bfs(MAZE, (4, 1), (5, 13))` aan met het doolhof van het speelblok, en dat moet een pad van 17 stappen geven. Hij probeert ook de twee doolhoven hieronder, `SMALL` uit het uitgewerkte voorbeeld en `CLOSED`, waarin geen weg naar G leidt, en een start die zelf het doel is. Elk kortste pad telt als goed.
        """),
        C.mazes("Twee kleine doolhoven, elk met de nummers van zijn rijen en kolommen. SMALL: 5 rijen bij 7 kolommen, het doolhof van het uitgewerkte voorbeeld, met S op rij 1, kolom 1 en G op rij 3, kolom 5. CLOSED: 6 rijen bij 7 kolommen, met S op rij 1, kolom 1 en G op rij 4, kolom 4, ingemetseld."),
        L.task("Laat `bfs` het oudste pad van voren uit de wachtrij pakken, en elk pad dat één action langer is achteraan zetten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bfs():
    box_bfs = L.code_box("bfs", lines=20)
    box_bfs
    return (box_bfs,)


@app.cell(hide_code=True)
def check_bfs(box_bfs):
    result_bfs = L.check("bfs", box_bfs.value)
    result_bfs
    return (result_bfs,)


@app.cell(hide_code=True)
def hints_bfs():
    L.hints(
        "bfs",
        [
            "Stap 2 en 4 van het recept. Welke kant van de wachtrij gebruikt elk van de twee?",
            "`popleft()` pakt een pad van voren uit een `deque`; `append` zet er een achteraan. Het nieuwe pad is het oude met `nxt` aan het eind.",
            "`path = frontier.___()` en `frontier.append(path + [___])`.",
        ],
        why="`frontier.popleft()` geeft de paden af in de volgorde waarin ze binnenkwamen, dus op volgorde van hun aantal stappen. `path + [nxt]` is een nieuwe lijst, dus elk pad in de wachtrij houdt zijn eigen states.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_formulation():
    mo.md(r"""
    ### Oefening 2. Welk deel is wat?

    Twee andere taken als search problem: een routeplanner die met de auto een weg door een stad vindt, en een schuifpuzzel met acht stukjes en één gat. Zet elk onderdeel in zijn groep.
    """)
    return


@app.cell(hide_code=True)
def exercise_formulation():
    sort_formulation = L.sort("formulation")
    sort_formulation
    return (sort_formulation,)


@app.cell(hide_code=True)
def check_formulation(sort_formulation):
    result_formulation = L.check("formulation", sort_formulation)
    result_formulation
    return (result_formulation,)


@app.cell(hide_code=True)
def hints_formulation():
    L.hints(
        "formulation",
        [
            "Vraag bij elk onderdeel: is het een situatie, een zet van de ene situatie naar de volgende, een vraag over één situatie, of iets wat je langs een pad optelt?",
            "Elke taak heeft één onderdeel in elke groep. De goal test is een vraag die je met ja of nee beantwoordt.",
            "Het kruispunt en de plek van de stukjes zijn states. Wat brengt je van de ene naar de volgende, en wat tel je onderweg op?",
        ],
        why="**State:** een kruispunt in de stad; waar elk stukje ligt. **Action:** één straat rijden; een stukje in het gat schuiven. **Goal test:** is dit het adres?; liggen de stukjes op volgorde? **Path cost:** de minuten die je reed; hoe vaak je een stukje schoof.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_newest():
    mo.md(r"""
    ### Oefening 3. Het nieuwste pad eerst
    """)
    return


@app.cell(hide_code=True)
def choice_newest_cell():
    choice_newest = L.choice(
        "newest",
        question="In `bfs` vervang je `frontier.popleft()` door `frontier.pop()`, die het nieuwste pad pakt, achter uit de wachtrij. Wat gebeurt er in MAZE?",
        options={
            "none": "Hij vindt geen weg naar het doel meer",
            "longer": f"Hij vindt nog steeds een weg naar het doel, maar een langere: {steps(len(C.DEPTH_FIRST) - 1)} in plaats van {len(C.SHORTEST) - 1}",
            "same": "Er verandert niets: hij pakt dezelfde paden, in een andere volgorde",
        },
        correct="longer",
        why={
            "longer": "Met het nieuwste pad eerst volgt de search één weg zo diep als hij gaat, en keert hij pas om bij een doodlopend stuk: dat is *depth-first search*. Hij vindt een weg, maar niets maakt die een kortste.",
            "none": "De search breidt nog steeds elke state uit die hij kan bereiken, tot een pad in het doel eindigt, dus hij vindt nog steeds een weg. Welke weg, hangt af van de volgorde. Zie *Het idee*.",
            "same": "Welk pad als eerste uit de wachtrij komt, bepaalt welk pad als eerste bij het doel komt. Zie *Het idee*: dat is wat de ene search van de andere onderscheidt.",
        },
    )
    choice_newest
    return (choice_newest,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** In [Leren van rewards](/course/open/rl-what-is-rl?lang=nl) leerde een agent door te proberen; hier kent hij zijn doolhof en plant hij de hele weg voordat hij een stap zet. De set `done` doet wat `seen` deed in [De volledige backward pass, in de goede volgorde](/course/open/nn-micrograd-backward?lang=nl), maar `build_order` ging eerst zo diep als hij kon, en deze search gaat steeds één stap verder in elke richting. De prijs van een kortste pad: hij breidt elke state uit die dichter bij de start ligt dan het doel. In de volgende les, *A\*: zoeken met richtinggevoel*, krijgt de search een schatting van hoe ver het doel nog is, en breidt hij veel minder states uit.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_newest, result_bfs, result_formulation):
    L.wrap_up(result_bfs, result_formulation, choice_newest)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
