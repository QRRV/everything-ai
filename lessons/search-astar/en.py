# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson search-astar. Route: course/route/lessen/overige-takken.yaml. Checks, the mazes and the play block: checks.py.
# termen:allow A star

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("search-astar", "en")
    C = L.checks   # the mazes, and the course's own A* and breadth-first search that draw the play block

    def steps(n):
        """A number of steps, in words."""
        return f"{n} step" if n == 1 else f"{n} steps"


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Searching a maze: breadth-first search](/course/open/search-bfs?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_order():
    quick_order = L.choice(
        "quick-order",
        question="In which order does breadth-first search take the paths out of its queue?",
        options={
            "newest": "The newest first: the one that came in last",
            "right": "The oldest first: the one that came in first",
            "closest": "The one that ends closest to the goal first",
        },
        correct="right",
        why={
            "right": "Its frontier is a queue: `popleft()` hands out the paths in the order they came in, so in order of their number of steps.",
            "newest": "Taking the newest path first is depth-first search, which does not always find a shortest way. See [the previous lesson](/course/open/search-bfs?lang=en).",
            "closest": "Breadth-first search does not know where the goal lies: it only sees whether a state is the goal. See [the previous lesson](/course/open/search-bfs?lang=en).",
        },
    )
    quick_order
    return (quick_order,)


@app.cell(hide_code=True)
def quick_check_expanded():
    quick_expanded = L.choice(
        "quick-expanded",
        question="Breadth-first search takes out a path to the goal of 10 steps. Which states has it expanded by then?",
        options={
            "path": "Only the states on that path",
            "towards": "Only the states that lie on the side of the goal",
            "right": "Every state that is less than 10 steps from the start, in every direction",
        },
        correct="right",
        why={
            "right": "It tries the paths in order of their number of steps, so every state closer to the start than the goal comes out first.",
            "path": "It does not know the path before it has found it. See the play block of [the previous lesson](/course/open/search-bfs?lang=en).",
            "towards": "It does not know on which side the goal lies. See the play block of [the previous lesson](/course/open/search-bfs?lang=en).",
        },
    )
    quick_expanded
    return (quick_expanded,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Breadth-first search looks in every direction, also away from the goal. A route planner that searched a whole country like that would look at every road closer to you than your destination. Yet you know roughly where the goal lies: can the search use that?

    **An everyday example.** You look for the station in a strange town, and you can see its tower. You do not walk every street around you in rings: you walk towards the tower, and only turn off when a street ends.
    """)
    return


@app.cell(hide_code=True)
def predict_how_many():
    guess_how_many = L.guess(
        "how-many",
        question=f"Think of a search that first takes out the path with the fewest steps so far plus an estimate of the steps still to go. In MAZE, breadth-first search expanded {len(C.BFS_DONE)} states to find a path of {steps(len(C.BFS_PATH) - 1)}. How many do you think this one expands?",
        options={
            "same": "About as many: around 50",
            "half": "About half: around 30",
            "path": "Only the states of the path: around 17",
        },
        reveal=f"**{len(C.DONE)}, and it still finds a path of {steps(len(C.SHORTEST) - 1)}.** This search is *A\\**. Walls make it expand more than the states of the path: the estimate does not see them.",
    )
    guess_how_many
    return (guess_how_many,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        Breadth-first search is **uninformed search**: it knows nothing about where the goal lies. **Informed search** uses an estimate of how far a state is from the goal: a **heuristic function**, in short the heuristic $h$. In the maze, $h$ counts the steps to the goal as if there were no walls: the rows apart plus the columns apart. Walls can only make the way longer, so $h$ never says more than the real number of steps.

        **A\*** is the search of the previous lesson with one change: it takes out of its frontier the path with the smallest $f$, the sum of two numbers.
        """),
        L.formula(r"f = [[loss:g]] + [[pred:h]]"),
        mo.md(r"""
        $g$ is the path cost so far, and $h$ the estimate of the steps still to go from where the path ends, so $f$ estimates the whole way through this path. The frontier is now a **priority queue**, which always hands out the item with the smallest number first. Python's `heapq` keeps a list as one: `heapq.heappush(frontier, (f, path))` puts a pair in, and `heapq.heappop(frontier)` takes out the pair with the smallest `f`: Python compares pairs by their first values first.

        When a path to the goal comes out, its $f$ is its real length, since $h$ is 0 at the goal. Every path still in the queue has an $f$ at least as large, and since $h$ never says too much, none reaches the goal in fewer steps: the path is a shortest one. Skipping a state in `done` loses nothing, because one step changes this $h$ by at most 1, so the first path to a state that comes out is a shortest one to it. A state far off to the side has a large $h$: paths through it get an $f$ above that length, and never come out.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **uninformed search** | | search that knows nothing about where the goal lies | |
        | **informed search** | | search that estimates how far each state is from the goal | |
        | **heuristic function** | | an estimate of the path cost from a state to the goal; in short, the heuristic | `h(state, goal)` |
        | **priority queue** | | a queue that hands out the item with the smallest number first | `frontier` |
        | `heapq` | heap q | Python's priority queue, on a list | `heapq.heappop(frontier)` |
        | **A\*** | A star | search that takes out the path with the smallest $f$ first | `astar` |
        | $g$ | g | the path cost so far | `g` |
        | $h$ | h | the estimate of the steps still to go | `h(nxt, goal)` |
        | $f$ | f | $g + h$, the estimated length of the whole way | `f` |

        Here is <span class="c-pred">h</span> for every state of MAZE: one step changes it by exactly 1.
        """),
        C.heuristic("MAZE with h, the rows plus the columns apart from G, in every open cell: 16 top left, 14 above S, 12 right of and below S, 1 above and below G."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Both searches run side by side in MAZE, one expanded state per step. A grey cell is a state a search expanded, a dashed ring a state on its frontier, and the thick ring the state it took out last. A search that has taken out the goal shows its path. The block opens at the last step.
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
            return "Step 0: both frontiers hold one path, the start alone."
        if k < _a_end:
            return (f"Step {k}: both searches expanded {k} states. The farthest: breadth-first search "
                    f"{steps(_far(_b))} from the start, A\\* {steps(_far(_a))}.")
        if k == _a_end:
            return f"Step {k}: A\\* takes out the goal. Breadth-first search has got no farther than {steps(_far(_b))} from the start."
        if k < _b_end:
            return f"Step {k}: A\\* is done. Breadth-first search expanded {k} states, the farthest {steps(_far(_b))} from the start."
        return (f"Step {k}: breadth-first search takes out the goal as well: a path of {steps(len(_b['path']) - 1)}, after expanding "
                f"{len(_b['done'])} states. A\\* needed {len(_a['done'])}.")

    def _frame(k):
        k = int(k)
        _b, _a = C.at_step(C.BFS_STEPS, k), C.at_step(C.ASTAR_STEPS, k)
        _alt = (f"MAZE twice, after step {k}: breadth-first search with {len(_b['done'])} states expanded, "
                f"A* with {len(_a['done'])}.")
        for _name, _s in (("Breadth-first search", _b), ("A*", _a)):
            if _s["path"] is not None:
                _alt += f" {_name} found a path of {steps(len(_s['path']) - 1)}, drawn."
        return C.side_by_side(k, ("Breadth-first search", "A*"), _alt).text, _say(k)

    explore_both = L.explore("both", "step", _frame, start=_b_end, low=0, high=_b_end, step=1, role="",
                             task="Slide to the step where A\\* takes out the goal. How far did breadth-first search get by then?")
    explore_both
    return


@app.cell(hide_code=True)
def text_after_play():
    _k = len(C.ASTAR_STEPS) - 1
    _b = C.at_step(C.BFS_STEPS, _k)["done"]
    mo.md(rf"""
    A\* takes out the goal at step {_k}, with a path as short as that of breadth-first search. By then, breadth-first search has got no farther than {steps(max(_b.values()))} from the start. A\* headed for the goal, and only turned aside where a wall was in the way.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A\*, from a start to a goal:

    1. Put one pair in the priority queue: the path with only the start, and its $f$, $h$ of the start. `done` starts empty.
    2. Take out the pair with the smallest $f$.
    3. Goal test: is the state where its path ends the goal? Then this path is a shortest one: stop.
    4. Is the state not in `done` yet? Then expand it: put it in `done`, and put every path one action longer in the priority queue, with its $f = g + h$: its steps plus $h$ of where it ends.
    5. Go back to step 2 while the priority queue is not empty.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    SMALL and `_neighbors` are those of the previous lesson's worked example. `_h` is the heuristic, and `_astar` follows the recipe, with its steps as comments. Inside it, a `print` shows every state it expands, after its $f$, its $g$ and its $h$. Run the cell.
    """)
    return


@app.cell
def example_astar():
    import heapq

    # SMALL: '#' is a wall; the start S and the goal G are cells you can stand on
    _maze = [
        "#######",
        "#S..#.#",
        "#.#...#",
        "#...#G#",
        "#######",
    ]

    def _neighbors(maze, state):
        # the actions: a step up, down, left or right, but not into a wall
        r, c = state
        result = []
        for nxt in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if maze[nxt[0]][nxt[1]] != "#":
                result.append(nxt)
        return result

    def _h(state, goal):
        # the steps to the goal if there were no walls: the rows apart plus the columns apart
        return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

    def _astar(maze, start, goal):
        # 1. one pair in the priority queue: f of the start, and the path with only the start
        frontier = [(_h(start, goal), [start])]
        done = set()
        while len(frontier) > 0:
            # 2. the pair with the smallest f
            f, path = heapq.heappop(frontier)
            state = path[-1]
            # 3. the goal test
            if state == goal:
                return path, done
            # 4. a new state: expand it, and put every longer path in, with its f
            if state not in done:
                done.add(state)
                print(f, len(path) - 1, _h(state, goal), state)
                for nxt in _neighbors(maze, state):
                    g = len(path)
                    f = g + _h(nxt, goal)
                    heapq.heappush(frontier, (f, path + [nxt]))
        # 5. the priority queue is empty: no way leads to the goal
        return None, done

    _path, _done = _astar(_maze, (1, 1), (3, 5))
    print("path:", _path)
    print(len(_path) - 1, "steps,", len(_done), "states expanded")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `g = len(path)` is the number of steps of the longer path: one more than that of `path`. Here $f$ stays 6: $h$ is exactly right on the whole path. Below the wall, `(2, 1)` has an $f$ of 6 as well, 1 step plus an $h$ of 5; with equal $f$, Python compares the paths, and the one along the top comes first. So A\* expands only the 6 states of the path, all but G, where breadth-first search expanded 11.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_astar():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. From breadth-first search to A\*

        `astar(maze, start, goal)` below is the search of the worked example without the `print`. Replace the two `...`: step 2 of the recipe, and $f$ for the longer path of step 4. The check calls `astar(MAZE, (4, 1), (5, 13))`, then the three mazes below, and a start that is the goal. Every shortest path counts as right, even with other expanded states where two paths have the same $f$.
        """),
        C.mazes("SMALL, 5 rows by 7 columns: S at (1, 1), G at (3, 5). CLOSED, 6 by 7: G walled in at (4, 4). TRAP, 7 by 11: S at (5, 1), G at (3, 9), behind a wall in column 7 with a gap only in row 5, so heading straight for G runs into it."),
        L.task("Make `astar` take out the pair with the smallest f, and give every longer path its f = g + h."),
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
            "Step 2 of the recipe, and the formula of *The idea*.",
            "`heapq.heappop(frontier)` gives back the pair that went in, `(f, path)`: give it two names. For the longer path, `g` is its number of steps, and `h(nxt, goal)` its estimate.",
            "`f, path = heapq.___(frontier)` and `f = g + ___`.",
        ],
        why="`heapq.heappop` hands out the pair with the smallest $f$: the path that seems to reach the goal in the fewest steps. $f = g + h$ adds the steps so far to the estimate from `nxt`, where the longer path ends.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_f():
    mo.md(r"""
    ### Exercise 2. One f
    """)
    return


@app.cell(hide_code=True)
def choice_f_cell():
    _s, _g, _h = C.FAR_STATE, C.FAR_G, C.FAR_H
    choice_f = L.choice(
        "f-value",
        question=f"In MAZE, a path of {steps(_g)} ends in `{_s}`. G is at `{C.GOAL}`. What is $f$ for this path?",
        options={
            "g": str(_g),
            "h": str(_h),
            "right": str(_g + _h),
        },
        correct="right",
        why={
            "right": f"$g$ = {_g} steps so far, and $h$ = {abs(_s[0] - C.GOAL[0])} row apart plus {abs(_s[1] - C.GOAL[1])} columns apart = {_h}. So $f$ = {_g} + {_h} = {_g + _h}: the whole way through this path takes at least {_g + _h} steps.",
            "h": f"{_h} is $h$ alone. $f$ also counts the {_g} steps the path took already.",
            "g": f"{_g} is $g$ alone. $f$ also counts $h$, the estimate of the steps still to go.",
        },
    )
    choice_f
    return (choice_f,)


@app.cell(hide_code=True)
def text_exercise_large():
    mo.md(r"""
    ### Exercise 3. An estimate that says too much
    """)
    return


@app.cell(hide_code=True)
def choice_large_cell():
    choice_large = L.choice(
        "too-large",
        question="To expand even fewer states, someone makes $h$ three times as large: `3 * h(nxt, goal)`. What can go wrong?",
        options={
            "none": "The search finds no way to the goal any more",
            "longer": "The search can hand back a way that is not a shortest one",
            "nothing": "Nothing: a larger estimate is always better",
        },
        correct="longer",
        why={
            "longer": f"Then $h$ outweighs the steps taken, so the search goes on with paths close to the goal, even after a long way round, while a shorter path waits in the queue. In TRAP, it hands back a way of {steps(len(C.TOO_LARGE) - 1)}, where {len(C.TRAP_PATH) - 1} is enough.",
            "none": "Every path still goes into the queue, so a way to the goal is still found; only which one comes out first changes.",
            "nothing": "A shortest path is only sure when $h$ never says more than the real number of steps. See *The idea*.",
        },
    )
    choice_large
    return (choice_large,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** With $h$ = 0, $f$ is just $g$, and A\* takes the paths in order of their number of steps, as [breadth-first search](/course/open/search-bfs?lang=en) does. Search comes back in a branch still to come, *Search, logic, planning and optimization*, whose project is your own search and planner. In the next lesson, *Winning a game: minimax*, an opponent moves too, and a path is no longer enough: you need a move for every answer.
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
