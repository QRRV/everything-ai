# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-value-iteration. Route: course/route/lessen/overige-takken.yaml. Checks, the worlds and the play block: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rl-value-iteration", "en")
    C = L.checks   # the grid worlds, their tables T, and the course's own value iteration that draws the play block


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A*: search with a sense of direction](/course/open/search-astar?lang=en) and [Expected value and variance](/course/open/prob-expectation?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_path():
    quick_path = L.choice(
        "quick-path",
        question="What does A\\* give back when it takes the goal out of its frontier?",
        options={
            "every": "An action for every state of the maze",
            "right": "A path: the states from the start to the goal, one action at a time",
            "far": "The number of steps from every state to the goal",
        },
        correct="right",
        why={
            "right": "A\\* gives back the path it took out: from the start, one action at a time, to the goal.",
            "every": "A\\* only knows what to do on the path it found; a state off that path gets no action. See [A*: search with a sense of direction](/course/open/search-astar?lang=en).",
            "far": "A\\* estimates the steps to the goal with $h$, but it gives back only one path. See [A*: search with a sense of direction](/course/open/search-astar?lang=en).",
        },
    )
    quick_path
    return (quick_path,)


@app.cell(hide_code=True)
def quick_check_expected():
    quick_expected = L.choice(
        "quick-expected",
        question="A game pays 10 points with probability 0.8, and nothing with probability 0.2. What is its expected value?",
        options={
            "middle": "5",
            "right": "8",
            "most": "10",
        },
        correct="right",
        why={
            "right": "Each outcome times its probability, added up: 0.8 · 10 + 0.2 · 0 = 8.",
            "middle": "5 is the average of 10 and 0, as if both came equally often. Weigh each outcome by its probability. See [Expected value and variance](/course/open/prob-expectation?lang=en).",
            "most": "10 is the outcome that comes most often, but one game in five pays nothing. See [Expected value and variance](/course/open/prob-expectation?lang=en).",
        },
    )
    quick_expected
    return (quick_expected,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In the maze of A\*, every action did what it said, so a path was a plan. A robot on a wet floor cannot count on that: a step can slip. After a slip it stands off its path, and the path no longer says what to do. It needs a choice for every state it can end up in. And a slip makes some ways riskier than others.

    **An everyday example.** An icy path runs along a canal. The shortest way is right at the edge, but every step can slide sideways. You walk a little further from the water, even though that way is longer.
    """)
    return


@app.cell(hide_code=True)
def predict_way():
    guess_way = L.guess(
        "way",
        question="In the grid world of this lesson, a row of pits lies below the way from S to the goal G: straight along the pits takes 5 steps, along the top row 7. Every step slips with probability 0.2, 0.1 to each side, and the discount factor is 0.9. Which do you think is better?",
        options={
            "straight": "The straight way: fewer steps, so less discount",
            "top": "The top row: further from the pits",
            "same": "Both are about as good",
        },
        reveal=f"**The top row.** Along the pits, every step falls in with probability 0.1. From S, the straight way has an expected return of {C.number(C.STRAIGHT[C.START], 2, 'en')}; the top row {C.number(C.V_STAR[C.START], 2, 'en')}. Value iteration, below, finds that out on its own.",
    )
    guess_way
    return (guess_way,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A task whose actions can slip is written down as a **Markov decision process**, an MDP: its states, its actions, a discount factor $\gamma$, and the **transition function**, which says for every state and action what can happen, with its probability and its reward. In this grid world, an action goes where it says with probability 0.8, and slips to each side of it with 0.1; a step into the edge stays where it is. Landing on G gives +1, in a pit −1, and there the task is over; every other step gives 0. The course writes it down as a table `T`: `T[s][a]` lists the outcomes of action `a` in state `s`, each a tuple `(probability, next state, reward)`.

        What can happen next depends only on the state and the action, not on how you got there: that is the *Markov* in the name. So one table is enough, and one number per state. The **value function** $V(s)$ is the return you can expect from state $s$ if you follow a [policy](/course/open/rl-what-is-rl?lang=en); in this lesson always the best one, so: if you act as well as you can. $Q(s, a)$ is the same if you take action $a$ first. The **Bellman equation** ties them together:
        """),
        L.formula(r"[[pred:Q(s, a)]] = \sum [[data:p]] \left( [[data:r]] + [[hyper:\gamma]] \, [[pred:V(s')]] \right) \qquad [[pred:V(s)]] = \max_a [[pred:Q(s, a)]]"),
        mo.md(r"""
        The sum goes over the outcomes of $a$ in $s$: each counts with its probability $p$, its reward $r$, and the value of the state $s'$ where it lands, a step later and so times $\gamma$. That is an expected value, as in the quick check. **Value iteration** starts with $V = 0$ everywhere and uses the equation as an update, for every state, round after round, until the values hardly change. The best policy then takes in each state $s$ the action $\pi(s)$ with the largest $Q(s, a)$.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **Markov decision process**, MDP | | states, actions, a transition function and a discount factor; what happens next depends only on the state and the action | `T`, `gamma` |
        | **transition function** | | for every state and action: what can happen, with its probability and reward | `T[s][a]` |
        | **value function**, $V(s)$ | V of s | the return you can expect from state $s$ if you follow a policy; here the best one | `V[s]` |
        | **action-value function**, $Q(s, a)$ | Q of s a | the same, if you take action $a$ first | `q` |
        | **Bellman equation** | | $Q$ from the rewards and the values one step on; $V$ is the largest $Q$ | |
        | **value iteration** | | the Bellman equation as an update, round after round, from $V = 0$ | `value_iteration` |
        """),
        L.watch_out("*Value* is also a plain word. Here the value of a state is the return you can expect from it: the value function $V(s)$."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Value iteration on GRID, round by round, without slip and with a slip of 0.2. A number is the <span class="c-pred">value</span> of that state after the round; a triangle points to each action with the largest $Q$, none where all four are equally good. The pits are grey, G is +1. The line below shows $V$(S) in every round, dashed without slip, with a mark at the round you chose.
    """)
    return


@app.cell(hide_code=True)
def explore_rounds_cell():
    _titles = ("Without slip", "With slip 0.2")
    _line = {"title": "V(S) per round", "x": "round", "y": "V(S)", "still": "without slip", "slip": "with slip",
              "alt": "V(S) after every round, from 0 to {n}: without slip 0 up to round {still_to} and {still_end} from round {still_from} on; "
                     "with slip below 0.01 up to round {tiny_to}, above 0.1 from round {above}, and {slip_end} from round {near} on. "
                     "The mark is at round {k}: {a} without slip and {b} with slip."}

    def _say(k):
        _a = C.number(C.STEPS["still"][k]["V"][C.START], 2, "en")
        _b = C.number(C.STEPS["slip"][k]["V"][C.START], 2, "en")
        if k == 0:
            return "Round 0: every value is 0. The triangles already show the best action where the next step decides: next to G the step onto it, next to a pit the actions that cannot fall in."
        if k < 5:
            return f"Round {k}: only states at most {k} {'step' if k == 1 else 'steps'} from G have a value. S is 5 steps away: $V$(S) is still 0 in both."
        return f"Round {k}: $V$(S) is {_a} without slip and {_b} with slip."

    def _frame(k):
        k = int(k)
        _alt = (f"GRID twice after round {k} of value iteration, 3 rows by 6 columns, with the pits along the bottom row: "
                f"without slip V(S) is {C.number(C.STEPS['still'][k]['V'][C.START], 2, 'en')}, "
                f"with slip {C.number(C.STEPS['slip'][k]['V'][C.START], 2, 'en')}.")
        if k == C.ROUNDS:
            _alt += " Without slip the policy goes straight along the pits; with slip it goes up to the top row."
        return C.side_by_side(k, _titles, _alt, decimal=".").text + C.value_of_s(k, _line, "en"), _say(k)

    explore_rounds = L.explore("rounds", "round", _frame, start=C.ROUNDS, low=0, high=C.ROUNDS, step=1, role="",
                               task="Slide from round 0 to the right. In which round does the value of S first rise above 0.1, without slip and with slip?")
    explore_rounds
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(rf"""
    Without slip, $V$(S) passes 0.1 in round {C.ABOVE["still"]}: S is 5 steps from G, and after $k$ rounds a value is the best expected return of the next $k$ steps, here 0.9⁴ ≈ 0.66. With slip, only in round {C.ABOVE["slip"]}: the straight way is worth almost nothing, and the top row takes 7 steps. And the policies differ: without slip straight along the pits, with slip up and along the top row.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Value iteration on an MDP:

    1. Write the MDP down as a table `T`: for every state and action, its outcomes `(probability, next state, reward)`.
    2. Give every state the value 0.
    3. One round: for every state, work out the $Q$ of every action, the outcomes' $p \cdot (r + \gamma \, V(s'))$ added up, and make the largest $Q$ its new value.
    4. Repeat step 3 until the values hardly change.
    5. The policy: in every state, the action with the largest $Q$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    SMALL is a world of 2 rows by 3 columns: S bottom left, G top right, and a pit bottom right. `C.table(C.SMALL)` makes its table `T`; the first `print` shows the outcomes of up in `(1, 1)`. `T[s][a]` looks up the state, then the action: `T` is a dictionary of dictionaries. `for p, nxt, r in T[s][a]` gives the three values of each outcome their own names, as `x, y = point` does. `max(qs)` is the largest number of the list `qs`, and `np.argmax(qs)` its place, as `.argmax()` gives it for an array; `round(x, 3)` rounds `x` to 3 decimals, to print it. Run the cell.
    """)
    return


@app.cell
def example_value_iteration():
    import numpy as np

    # 1. the MDP of SMALL as a table: T[state][action] lists the outcomes,
    #    each as (probability, next state, reward)
    _T = C.table(C.SMALL)
    print(_T[(1, 1)]["up"])

    def _value_iteration(T, gamma, rounds):
        # 2. every state starts with the value 0
        V = {}
        for s in T:
            V[s] = 0
        pi = {}
        for k in range(rounds):
            new = {}
            for s in T:
                # 3. the Q of every action: p * (r + gamma * V[nxt]) of its outcomes, added up
                actions = []
                qs = []
                for a in T[s]:
                    q = 0
                    for p, nxt, r in T[s][a]:
                        q = q + p * (r + gamma * V[nxt])
                    actions.append(a)
                    qs.append(q)
                # 3. the new value is the largest Q; 5. the policy takes the action that gives it
                new[s] = max(qs)
                pi[s] = actions[np.argmax(qs)]
            V = new
            # 4. a round is done: the values of (0, 0), (0, 1), (1, 0) and (1, 1)
            print(k + 1, round(V[(0, 0)], 3), round(V[(0, 1)], 3),
                  round(V[(1, 0)], 3), round(V[(1, 1)], 3))
        return V, pi

    _V, _pi = _value_iteration(_T, 0.9, 12)
    print(_pi[(0, 0)], _pi[(0, 1)], _pi[(1, 0)], _pi[(1, 1)])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Up in `(1, 1)` lands on `(0, 1)` with probability 0.8, slips left to S with 0.1, and slips right into the pit with 0.1, reward −1. In round 1, only `(0, 1)` gets a value: right lands on G with probability 0.8, so 0.8. Each round, the values reach one state further and change less; after 12 rounds, less than 0.001. The policy is right, right, up, and in `(1, 1)` left: back to S, not up. Up gets to `(0, 1)` more often, but can slip into the pit; left cannot. With the values after 12 rounds, left has a $Q$ of 0.66 and up 0.64.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_vi():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Your own value iteration

        `value_iteration(T, gamma, rounds)` below is the function of the worked example, without the `print`. The check calls `value_iteration(T, 0.9, 30)` on the table `T` of GRID, where $V$(S) should come out at 0.42, and also on `T_SMALL` and `T_BRIDGE`, the two worlds below, with other discount factors and fewer rounds. Where two actions are equally good, either counts as right.
        """),
        C.worlds("Two small grid worlds, each with the numbers of its rows and columns. SMALL: 2 rows by 3 columns, S at row 1, column 0, G at row 0, column 2, a pit at row 1, column 2. BRIDGE: 2 rows by 4 columns, S at row 0, column 0 and G at row 0, column 3, with pits along the whole bottom row.", decimal="."),
        L.task("Make `value_iteration` add up each outcome's p · (r + γ V(s′)) into `q`, and take the action with the largest Q as the policy."),
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
            "Steps 3 and 5 of the recipe, and the Bellman equation of *The idea*.",
            "Each outcome counts with its probability, times its reward plus the discounted value of `nxt`. For the policy: `np.argmax(qs)` is the place of the largest Q, and `actions` holds the action at each place.",
            "`q = q + p * (___ + gamma * V[___])` and `pi[s] = actions[np.___(qs)]`.",
        ],
        why="`q` adds up the outcomes of one action, each weighed by its probability: an expected value. The largest `q` is the new value of the state; `actions[np.argmax(qs)]` is the action that gives it, the policy.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_neighbor():
    mo.md(r"""
    ### Exercise 2. Towards the larger value?
    """)
    return


@app.cell(hide_code=True)
def choice_neighbor_cell():
    _q = C.Q_STAR[C.START]
    choice_neighbor = L.choice(
        "neighbor",
        question=f"In GRID with slip, the state right of S has the value {C.number(C.V_STAR[(1, 1)], 2, 'en')}, the state above S {C.number(C.V_STAR[(0, 0)], 2, 'en')}. Which action does the policy take in S?",
        options={
            "right": "Right: towards the larger value",
            "up": "Up: its Q is larger, because a step to the right can slip into a pit",
            "left": "Left, into the edge: there it cannot fall",
        },
        correct="up",
        why={
            "up": f"$Q$(S, up) is {C.number(_q['up'], 2, 'en')}, $Q$(S, right) only {C.number(_q['right'], 2, 'en')}. The policy compares the $Q$ of each action, with all its outcomes, not the values of the neighbors.",
            "right": f"The value of a neighbor counts only for the outcome that lands there. Right also slips down into a pit with probability 0.1: $Q$(S, right) is {C.number(_q['right'], 2, 'en')}, against {C.number(_q['up'], 2, 'en')} for up. See the Bellman equation in *The idea*.",
            "left": f"Left slips up or down, and down is a pit: $Q$(S, left) is {C.number(_q['left'], 2, 'en')}. Work out the outcomes of each action, as in *The idea*.",
        },
    )
    choice_neighbor
    return (choice_neighbor,)


@app.cell(hide_code=True)
def text_exercise_markov():
    mo.md(r"""
    ### Exercise 3. Still an MDP?
    """)
    return


@app.cell(hide_code=True)
def choice_markov_cell():
    choice_markov = L.choice(
        "markov",
        question="Suppose a cell of GRID gets more slippery the more often you stand on it: from the third time on, a step from it slips with probability 0.5 instead of 0.2. Is the state (row, column) then still enough for an MDP?",
        options={
            "yes": "Yes: every action still has its probabilities",
            "count": "No: what happens next also depends on how often you stood on the cell before, so that belongs in the state",
            "never": "No: a task in which actions can slip is never an MDP",
        },
        correct="count",
        why={
            "count": "An MDP asks that what happens next depends only on the state and the action. Here it also depends on the way you came, so the state has to hold how often you stood on each cell.",
            "yes": "The probabilities are there, but they depend on how often you stood on the cell, which (row, column) does not say. See the MDP in *The idea*.",
            "never": "GRID slips and is an MDP: the probabilities depend only on the state and the action. See the MDP in *The idea*.",
        },
    )
    choice_markov
    return (choice_markov,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** When steps can slip, a path of [A*: search with a sense of direction](/course/open/search-astar?lang=en) is no longer a plan: the Bellman equation gives every state a value, and the policy an action for every state, even off the best way. Value iteration needs the whole table `T`: every probability and every reward. In the next lesson, *Learning from your own steps: Q-learning*, the agent does not know `T`, and learns $Q(s, a)$ from the steps it takes.
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
