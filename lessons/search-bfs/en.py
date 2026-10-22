# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson search-bfs. Route: course/route/lessen/overige-takken.yaml. Checks, the mazes and the play block: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("search-bfs", "en")
    C = L.checks   # the mazes, and the course's own breadth-first search that draws the play block

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

    Two short questions about [Learning from rewards](/course/open/rl-what-is-rl?lang=en) and [The full backward pass, in the right order](/course/open/nn-micrograd-backward?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_action():
    quick_action = L.choice(
        "quick-action",
        question="A robot in a maze can step up, down, left or right, and it gets a reward when it reaches the exit. In the words of *Learning from rewards*, what is \"a step to the left\"?",
        options={
            "reward": "A reward",
            "environment": "The environment",
            "right": "An action",
        },
        correct="right",
        why={
            "right": "The robot chooses it, and it changes where the robot is: an action.",
            "reward": "A reward is the number the environment gives back after a step; the step itself is what the robot chooses. See [Learning from rewards](/course/open/rl-what-is-rl?lang=en).",
            "environment": "The environment is the maze around the robot; the step is what the robot does in it. See [Learning from rewards](/course/open/rl-what-is-rl?lang=en).",
        },
    )
    quick_action
    return (quick_action,)


@app.cell(hide_code=True)
def quick_check_seen():
    quick_seen = L.choice(
        "quick-seen",
        question="What does this program show?\n\n```python\nseen = set()\nfor v in [\"a\", \"b\", \"a\", \"c\", \"b\"]:\n    seen.add(v)\nprint(len(seen))\n```",
        options={
            "none": "`0`",
            "right": "`3`",
            "all": "`5`",
        },
        correct="right",
        why={
            "right": "`seen.add(v)` puts `v` in the set, and a value that is in it already stays there once: `a`, `b` and `c`.",
            "all": "A set keeps each value once: the second `a` and the second `b` are in it already. See the set `seen` in [The full backward pass, in the right order](/course/open/nn-micrograd-backward?lang=en).",
            "none": "`seen` starts empty, but `seen.add(v)` puts every value in it. See the set `seen` in [The full backward pass, in the right order](/course/open/nn-micrograd-backward?lang=en).",
        },
    )
    quick_seen
    return (quick_seen,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Learning from rewards](/course/open/rl-what-is-rl?lang=en), the agent does not know where an action leads, so it has to try. Often you do know: you have the map of a town, or the rules of a puzzle. Then an agent can think before it acts: try the actions in its head, and only then take the best way. That is **search**, one of the oldest ideas of AI, and what a route planner does.

    **An everyday example.** Pour water into the entrance of a maze. It flows into every corridor at once, a little further every second. The first water that reaches the exit came the shortest way.
    """)
    return


@app.cell(hide_code=True)
def predict_first_path():
    guess_first_path = L.guess(
        "first-path",
        question="A search tries every path of 1 step first, then every path of 2 steps, then of 3, and so on, until a path reaches the goal. What do you think is true of that first path to the goal?",
        options={
            "luck": "It is the way the search happened to try first; a shorter one may exist",
            "fewest": "It is found after looking at as few cells as possible",
            "shortest": "It is a shortest way to the goal",
        },
        reveal="**It is a shortest way.** Every shorter path came before it, and none of them reached the goal. This search is *breadth-first search*, the search of this lesson.",
    )
    guess_first_path
    return (guess_first_path,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        To search, you first write the task down as a **search problem**: the **states** it can be in, the state you start in, the actions in each state and where each action leads, a **goal test**, and the **path cost**. In the maze, a state is the cell you stand in, as a tuple `(row, column)`; the actions are a step up, down, left or right, but not into a wall; and the path cost of a path is its number of steps, where a route planner could count minutes. A solution is a path from the start to a goal, one action at a time.

        A search keeps the paths it found but has not followed further: the **frontier**. Again and again it takes one path out and **expands** the state where it ends: it puts every path that is one action longer in the frontier. Which path it takes out first is what makes one search differ from another. **Breadth-first search** takes the oldest one, the one that came in first: its frontier is a **queue**, as at the checkout of a shop. A set `done` remembers the states it expanded, so it never expands one twice and never walks in circles; `state not in done` asks whether a state is not in it yet, the opposite of `in`.

        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **state** | one situation of the task; in the maze, the cell you stand in | `state = (4, 1)` |
        | **search problem** | a task as states, a start, the actions in each state, a goal test and a path cost | |
        | **goal test** | the question whether a state is a goal | `state == goal` |
        | **path cost** | what a path costs; here, its number of steps | `len(path) - 1` |
        | **frontier** | the paths found but not expanded yet | `frontier` |
        | **expand** | put every path one action longer in the frontier | |
        | **queue** | a row in which what comes in first goes out first | |
        | **breadth-first search** | search that takes the oldest path from its frontier first | `bfs` |
        | `deque` | Python's queue, from `collections`: `append` puts at the back, `popleft()` takes from the front | `frontier.popleft()` |

        A path is a list of its states, the start included, so its number of steps is `len(path) - 1`.
        """),
        L.watch_out("*State* means other things elsewhere, such as the hidden state of a recurrent network, a vector of numbers. Here, as in reinforcement learning, a state is one situation of the task."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Breadth-first search on the maze of this lesson, step by step. S is the start and G the goal; the numbers along the top and the left are the columns and the rows, as the code counts them. A cell with a number is an expanded state, and the number is its <span class="c-loss">number of steps</span> from the start; a dashed ring is a state on the frontier, and the thick ring the state taken out last.
    """)
    return


@app.cell(hide_code=True)
def explore_steps_cell():
    _last = len(C.STEPS) - 1

    def _say(k):
        _s = C.STEPS[k]
        if k == 0:
            return "Step 0: the queue holds one path, with only the start in it."
        if _s["path"] is not None:
            return (f"Step {k}: the path it takes out ends in the goal: {steps(len(_s['path']) - 1)}. Every shorter path came out "
                    f"before it, so this one is a shortest. The search expanded {len(_s['done'])} states.")
        _far = sorted(set(_s["frontier"].values()))
        _ring = steps(_far[0]) if len(_far) == 1 else ", ".join(str(_d) for _d in _far[:-1]) + f" or {steps(_far[-1])}"
        return (f"Step {k}: the search expands `{_s['mark']}`, {steps(_s['done'][_s['mark']])} from the start, and puts the paths "
                f"to its neighbors at the back of the queue. The frontier holds {len(_s['frontier'])} states, {_ring} from the start.")

    def _frame(k):
        k = int(k)
        _s = C.STEPS[k]
        _alt = (f"The maze of this lesson, {len(C.MAZE)} rows by {len(C.MAZE[0])} columns, after step {k} of breadth-first search: "
                f"{len(_s['done'])} {'state' if len(_s['done']) == 1 else 'states'} expanded, {len(_s['frontier'])} on the frontier.")
        if _s["path"] is not None:
            _alt += f" The path of {steps(len(_s['path']) - 1)} from S to G is drawn."
        return C.picture(k, _alt).text, _say(k)

    explore_steps = L.explore("steps", "step", _frame, start=20, low=0, high=_last, step=1, role="",
                              task="Slide to the last step. How long is the path the search finds, and how many states did it expand to find it?")
    explore_steps
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(rf"""
    The frontier is a ring that grows outwards: its states are all *d* or *d* + 1 steps from the start. To find the path of {steps(len(C.SHORTEST) - 1)}, the search expanded {len(C.DONE)} of the {C.open_cells(C.MAZE)} states, including the ones on the side away from the goal.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Breadth-first search, from a start to a goal:

    1. Put one path in the queue: the start alone. The set `done` starts empty.
    2. Take the oldest path out, from the front of the queue. Its last state is where it ends.
    3. Goal test: is that state the goal? Then this path is a shortest one: stop.
    4. Is the state not in `done` yet? Then expand it: put it in `done`, and put every path that is one action longer at the back of the queue.
    5. Go back to step 2 while the queue is not empty. An empty queue means that no way leads to the goal.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A small maze, as a list of strings: `#` is a wall, `S` the start and `G` the goal. `_neighbors` gives the states one action away, and `_bfs` follows the recipe; its `print` shows the number of steps and the last state of every path it takes out before the goal. `path + [nxt]` makes a new list, the path with `nxt` at the end, as `+` glues two strings together; `path` itself stays as it was. At the end, `_bfs` searches a second maze, where a wall stands between S and G. Run the cell.
    """)
    return


@app.cell
def example_bfs():
    from collections import deque

    # the maze: '#' is a wall; the start S and the goal G are cells you can stand on
    _maze = [
        "#######",
        "#S..#.#",
        "#.#...#",
        "#...#G#",
        "#######",
    ]
    # a second maze, where a wall stands between S and G
    _closed = [
        "#####",
        "#S#G#",
        "#####",
    ]

    def _neighbors(maze, state):
        # the actions: a step up, down, left or right, but not into a wall
        r, c = state
        result = []
        for nxt in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if maze[nxt[0]][nxt[1]] != "#":
                result.append(nxt)
        return result

    def _bfs(maze, start, goal):
        # 1. one path in the queue, the start alone; nothing is done yet
        frontier = deque([[start]])
        done = set()
        while len(frontier) > 0:
            # 2. the oldest path, from the front of the queue
            path = frontier.popleft()
            state = path[-1]
            # 3. the goal test
            if state == goal:
                return path, done
            # 4. a new state: expand it, and put every longer path at the back
            if state not in done:
                done.add(state)
                print(len(path) - 1, state)
                for nxt in _neighbors(maze, state):
                    frontier.append(path + [nxt])
            else:
                # 4. a state that is done already: skip it
                print(len(path) - 1, state, "done already")
        # 5. the queue is empty: no way leads to the goal
        return None, done

    _path, _done = _bfs(_maze, (1, 1), (3, 5))
    print("path:", _path)
    print(len(_path) - 1, "steps,", len(_done), "states expanded")
    print("path:", _bfs(_closed, (1, 1), (1, 3))[0])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The numbers of steps at the start of the lines never go down: the queue hands out the paths in order of their number of steps. A line with `done already` ends in a state that was expanded before, such as the way back to `(1, 1)`: the search skips it. The goal comes out after 6 steps, a shortest path; by then all 11 other states were expanded, even the dead end `(1, 5)`, as far from the start as the goal. In the second maze, the queue is empty after the start, so `_bfs` gives `None`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bfs():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Your own breadth-first search

        `bfs(maze, start, goal)` below is the search of the worked example without the `print` lines. Replace the two `...` with steps 2 and 4 of the recipe. The check calls `bfs(MAZE, (4, 1), (5, 13))` with the maze of the play block, which should give a path of 17 steps. It also tries the two mazes below, `SMALL` from the worked example and `CLOSED`, where no way leads to G, and a start that is the goal itself. Every shortest path counts as right.
        """),
        C.mazes("Two small mazes, each with the numbers of its rows and columns. SMALL: 5 rows by 7 columns, the maze of the worked example, with S at row 1, column 1 and G at row 3, column 5. CLOSED: 6 rows by 7 columns, with S at row 1, column 1 and G at row 4, column 4, walled in."),
        L.task("Make `bfs` take the oldest path from the front of the queue, and put every path that is one action longer at the back."),
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
            "Steps 2 and 4 of the recipe. Which end of the queue does each one use?",
            "`popleft()` takes a path from the front of a `deque`; `append` puts one at the back. The new path is the old one with `nxt` at the end.",
            "`path = frontier.___()` and `frontier.append(path + [___])`.",
        ],
        why="`frontier.popleft()` hands out the paths in the order they came in, so in order of their number of steps. `path + [nxt]` is a new list, so every path in the queue keeps its own states.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_formulation():
    mo.md(r"""
    ### Exercise 2. Which part is which?

    Two other tasks as search problems: a route planner that finds a way by car through a town, and a sliding puzzle with eight tiles and one gap. Put each part in its group.
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
            "For each part, ask: is it a situation, a move from one situation to the next, a question about one situation, or something you add up along a path?",
            "Each task has one part in each group. The goal test is a question you can answer with yes or no.",
            "The crossroads and the places of the tiles are states. What takes you from one to the next, and what do you add up on the way?",
        ],
        why="**State:** a crossroads in the town; where each tile lies. **Action:** drive one street; slide a tile into the gap. **Goal test:** is this the address?; are the tiles in order? **Path cost:** the minutes you drove; how many times you slid a tile.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_newest():
    mo.md(r"""
    ### Exercise 3. The newest path first
    """)
    return


@app.cell(hide_code=True)
def choice_newest_cell():
    choice_newest = L.choice(
        "newest",
        question="In `bfs`, you replace `frontier.popleft()` by `frontier.pop()`, which takes the newest path, from the back of the queue. What happens in MAZE?",
        options={
            "none": "It finds no way to the goal any more",
            "longer": f"It still finds a way to the goal, but a longer one: {steps(len(C.DEPTH_FIRST) - 1)} instead of {len(C.SHORTEST) - 1}",
            "same": "Nothing changes: it takes the same paths out, in another order",
        },
        correct="longer",
        why={
            "longer": "Taking the newest path first, the search follows one way as deep as it goes, and only turns back at a dead end: that is *depth-first search*. It finds a way, but nothing makes it a shortest one.",
            "none": "The search still expands every state it can reach until a path ends in the goal, so it still finds a way. Which way depends on the order. See *The idea*.",
            "same": "Which path comes out first decides which path reaches the goal first. See *The idea*: that is what makes one search differ from another.",
        },
    )
    choice_newest
    return (choice_newest,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Learning from rewards](/course/open/rl-what-is-rl?lang=en), an agent learned by trying; here it knows its maze and plans the whole way before it takes a step. The set `done` does what `seen` did in [The full backward pass, in the right order](/course/open/nn-micrograd-backward?lang=en), but `build_order` went as deep as it could first, and this search goes one step further in every direction at a time. The price of a shortest path: it expands every state closer to the start than the goal. In the next lesson, *A\*: search with a sense of direction*, the search gets an estimate of how far the goal is, and expands far fewer states.
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
