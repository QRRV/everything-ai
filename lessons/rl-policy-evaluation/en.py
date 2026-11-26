# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-policy-evaluation. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the room, the course's policy evaluation and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rl-policy-evaluation", "en")
    C = L.checks   # the room and its table, the course's policy evaluation, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [When steps can slip: value iteration](/course/open/rl-value-iteration?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_q():
    quick_q = L.choice(
        "quick-q",
        question="An action has two outcomes, both with reward 0: with probability 0.8 it lands in a state with value 1.0, with 0.2 in a state with value 0.5. With $\\gamma$ = 0.9, what is $Q(s, a)$?",
        options={
            "added": "1.35, both values added and then multiplied by γ",
            "right": "0.81, which is γ times the expected value 0.9",
            "no_gamma": "0.9, each value times its probability",
            "likely": "1.0, the value of the outcome with the highest probability",
        },
        correct="right",
        why={
            "right": "Each outcome counts with its probability: 0.8 · (0 + 0.9 · 1.0) + 0.2 · (0 + 0.9 · 0.5) = 0.72 + 0.09 = 0.81.",
            "added": "Both outcomes cannot happen at once: how much does each count when it happens with probability 0.8 or 0.2? See [When steps can slip: value iteration](/course/open/rl-value-iteration?lang=en).",
            "no_gamma": "0.8 · 1.0 + 0.2 · 0.5 = 0.9 is the expected value of the next state, but that state is a step later: what does $\\gamma$ do with it? See [When steps can slip: value iteration](/course/open/rl-value-iteration?lang=en).",
            "likely": "The less likely outcome can happen too, one time in five: how much does it count in an expected value? See [When steps can slip: value iteration](/course/open/rl-value-iteration?lang=en).",
        },
    )
    quick_q
    return (quick_q,)


@app.cell(hide_code=True)
def quick_check_max():
    quick_max = L.choice(
        "quick-max",
        question="In value iteration, how does a round give a state $s$ its new value $V(s)$?",
        options={
            "mean": "It takes the mean of the four $Q(s, a)$, one for each action",
            "right": "It takes the largest $Q(s, a)$: the value of the best action in $s$",
            "last": "It keeps the $Q$ of the action the agent took there last time",
            "all": "It adds up the $Q(s, a)$ of every action, since each action could be taken",
        },
        correct="right",
        why={
            "right": "Value iteration looks for the best policy, which takes the best action in every state, so $V(s)$ is the largest $Q(s, a)$. In this lesson the policy is given, and the largest $Q$ makes way for something else.",
            "mean": "Value iteration finds the values of the best policy: does that policy take every action equally often? See [When steps can slip: value iteration](/course/open/rl-value-iteration?lang=en).",
            "last": "Value iteration works from the table `T`, and no agent takes actions while it runs: which $Q$ does the best policy use? See [When steps can slip: value iteration](/course/open/rl-value-iteration?lang=en).",
            "all": "The agent takes one action in a state, not all of them: which one does the best policy take? See [When steps can slip: value iteration](/course/open/rl-value-iteration?lang=en).",
        },
    )
    quick_max
    return (quick_max,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [When steps can slip: value iteration](/course/open/rl-value-iteration?lang=en), the values said what a state is worth if the agent acts as well as it can. Often the question is a different one: what is a state worth with the policy the agent has now? A robot vacuum comes with a fixed rule for where to go next, a game has a strategy someone wrote, an agent runs the policy it learned last week. Before you change such a policy, you want to know what it is worth from every state, and whether another policy would do better. Policy evaluation answers that, and the next lesson builds on it to make a policy better and better.

    **An everyday example.** In a board game with a die, how many turns does a piece need on average to reach the finish from a given square? You do not have to play a thousand games to find out. Work backward from the finish: the number for a square is one turn plus the average of the numbers of the six squares a roll can bring the piece to.
    """)
    return


@app.cell(hide_code=True)
def predict_steps():
    guess_steps = L.guess(
        "steps",
        question="A robot stands on a grid of 5 by 5 tiles, with its charger on the middle tile. It has no plan: at every step it goes up, down, left or right, each with probability 1/4, and a step into the edge leaves it where it is. It starts on a tile next to the charger. How many steps does it need on average to reach the charger?",
        options={
            "four": "About 4: one step in four goes onto the charger",
            "two": "About 2",
            "right": "24",
            "ten": "About 10, since on a grid this small it never gets far from the charger",
        },
        reveal="**24.** From a tile next to the charger, the robot needs 24 steps on average; from a corner, 4 steps from the charger by the shortest way, it needs 36. One step in four goes onto the charger, but the other three take the robot away from it, and from there it needs longer. *The idea* shows how to work these numbers out without letting the robot walk at all.",
    )
    guess_steps
    return (guess_steps,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The robot's rule is a **policy**, as in [Learning from rewards](/course/open/rl-what-is-rl?lang=en). A policy can give one action per state, as the best policy of value iteration did, or a probability for every action: $\pi(a \mid s)$ is the probability that the policy takes action $a$ in state $s$. The robot's rule is the **random policy**: every action with probability 1/4.

        Give every step a reward of −1, and let the task end on the charger, with the value 0. With $\gamma = 1$, the return of a walk is minus its number of steps, so the value of a tile is minus the average number of steps from there to the charger. That is the value function of this policy, $V_\pi(s)$: the return to expect from $s$ if you follow $\pi$. The **Bellman equation of a policy** ties the values of neighbors together, as the Bellman equation of value iteration did:
        """),
        L.formula(r"[[pred:V_\pi(s)]] = \sum_a \pi(a \mid s) \sum [[data:p]] \left( [[data:r]] + [[hyper:\gamma]] \, [[pred:V_\pi(s')]] \right)"),
        mo.md(r"""
        The inner part is $Q(s, a)$, as in value iteration: each outcome of $a$ with its probability $p$, its reward $r$ and the value of the state $s'$ it lands in. Where value iteration took the largest $Q(s, a)$, this policy takes each action with its probability, so its value is the average of the $Q$-values, each counted with $\pi(a \mid s)$. On a tile next to the charger, one step in four lands on the charger, worth −1 + 0, and three in four land on a tile from which the robot needs longer.

        **Policy evaluation** finds $V_\pi$ the way value iteration found the best values: start with $V = 0$ everywhere, and use the equation as an update for every state, from the old values. One such pass over all states is a **sweep**; the rounds of value iteration were sweeps too. After sweep 1 every tile is at −1: one step, counted. After sweep 2 a tile next to the charger is at ¼ · (−1 + 0) + ¾ · (−1 − 1) = −1.75, every other tile at −2. After $k$ sweeps a value counts the first $k$ steps of the walks from that tile; walks that take longer come in with later sweeps. Once the largest change of a sweep is smaller than a small number $\theta$, the values hardly change any more, and policy evaluation stops.

        This works with $\gamma = 1$ because the random robot reaches the charger sooner or later from every tile. Methods that work out values from a known table by applying the Bellman equation to every state over and over are called **dynamic programming**: value iteration is one of them, policy evaluation another.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **policy evaluation** | | working out the value of every state under a given policy, with sweeps of its Bellman equation | `evaluate` |
        | $\pi(a \mid s)$ | pi of a given s | the probability that the policy takes action $a$ in state $s$ | `pi[s][a]` |
        | **random policy** | | a policy that takes every action with the same probability | `RANDOM` |
        | $V_\pi(s)$ | V pi of s | the value of state $s$ if you follow policy $\pi$ | `V[s]` |
        | **Bellman equation of a policy** | | the value of a state as the average of its $Q$-values, each counted with the probability of its action | |
        | **sweep** | | one update of every state, from the old values | `sweep` |
        | $\theta$ | theta | policy evaluation stops once the largest change of a sweep is below it | `theta` |
        | **dynamic programming** | | methods that work out values from a known table by applying the Bellman equation to every state over and over | |
        """),
        L.watch_out("In robotics and elsewhere, *evaluating a policy* usually means measuring how often it succeeds in tests. Here policy evaluation means working out its values from the table."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Policy evaluation of the random policy on ROOM, the robot's grid, sweep by sweep. A number is the <span class="c-pred">value</span> of that tile after the sweep; the charger in the middle has the value 0. The chart beside it shows the values of two tiles after every sweep: a tile next to the charger and a corner.
    """)
    return


@app.cell(hide_code=True)
def explore_sweeps_cell():
    explore_sweeps = L.explore("sweeps", "sweep", lambda _v: C.play_frame(int(_v), L.lang), start=C.PLAY_SWEEPS,
                               values=C.LOOKS, role="",
                               task="Go to sweep 3: why is the corner at exactly −3, while the tile next to the charger is at −2.50? Then go on to sweep 100: how far is the corner still from −36?")
    explore_sweeps
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    After 3 sweeps a value counts the first 3 steps of every walk. From the corner the charger is 4 steps away, so no walk from there has ended yet: three steps of −1 make −3. From the tile next to the charger, a quarter of the walks end with the first step, so on average fewer steps are counted: −2.50. Over the sweeps the values go down to −24 next to the charger and −36 in a corner, with −30 diagonal to the charger and −32 and −34 along the edges. After 100 sweeps the corner is at −34.45, still 1.55 from −36: every sweep takes off only about 3% of what is left, so the last part takes many sweeps.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Policy evaluation of a policy $\pi$, with a table `T`:

    1. Start with $V(s) = 0$ for every state.
    2. Sweep: for every state $s$ and every action $a$, work out $Q(s, a)$ from the old values, as in value iteration.
    3. The new value of $s$ is the average of these $Q$-values, each counted with $\pi(a \mid s)$. Keep the new values apart from the old ones until the sweep is done.
    4. Note how far each value moved in this sweep, and keep the largest change.
    5. Replace the old values by the new ones. If the largest change is $\theta$ or more, go on at step 2.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A corridor of four tiles: the charger on tile 0 at the left end, and a wall after tile 3. The robot steps left or right, each with probability 1/2; against the wall it stays on tile 3. Every step costs −1, and $\gamma$ = 1. The cell keeps the four values in a list and does 200 sweeps, many more than these values need; with a fixed number of sweeps it skips step 4 of the recipe, which exercise 2 adds. It shows the values of tiles 1, 2 and 3 after the first three sweeps and after the last. Run the cell.
    """)
    return


@app.cell
def example_corridor():
    # 1. the values of the four tiles; tile 0 is the charger and keeps 0
    _V = [0, 0, 0, 0]
    for _k in range(200):
        # 2. and 3. one sweep: every new value from the old values, each step with probability 1/2
        _new = [0, 0, 0, 0]
        for _s in [1, 2, 3]:
            _left = _s - 1
            _right = min(_s + 1, 3)   # against the wall the robot stays on tile 3
            _new[_s] = 0.5 * (-1 + _V[_left]) + 0.5 * (-1 + _V[_right])
        # 5. the new values replace the old ones
        _V = _new
        if _k < 3 or _k == 199:
            print("after sweep", _k + 1, ":", round(_V[1], 2), round(_V[2], 2), round(_V[3], 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Sweep 1 gives every tile −1: one step, counted. In sweep 2, tile 1 gets ½ · (−1 + 0) + ½ · (−1 − 1) = −1.5: half of its steps land on the charger. Tile 3 is at −2 after sweep 2 and at −3 after sweep 3: the charger is three tiles away, so no walk from there has ended within three steps. After 200 sweeps the values have settled at −6, −10 and −12: from the far end of the corridor the robot needs 12 steps on average, four times the 3 steps of the shortest way.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_evaluate():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. One policy, many sweeps

        `evaluate(T, pi, gamma, sweeps)` does `sweeps` sweeps from $V = 0$ and gives back the values: a dictionary with a value for every state. `T` is a table as in value iteration, and `pi[s][a]` is the probability that the policy takes action `a` in state `s`. `k` counts the sweeps, and `new` holds the new values until a sweep is done. The loop over `p, nxt, r` in `T[s][a]` already works out `q`, the $Q$ of action `a` in state `s`, and `v` collects the new value of `s`. Replace the `...` with the share of action `a` in that value, step 3 of the recipe. The check calls `evaluate(T_ROOM, RANDOM, 1.0, 2)`, two sweeps of the random policy on ROOM, and five other cases, among them the corridor and a policy that goes up more often than the other ways.
        """),
        L.task("Make `evaluate` add the share of each action to the new value of a state."),
    ])
    return


@app.cell(hide_code=True)
def exercise_evaluate():
    box_evaluate = L.code_box("evaluate", lines=16)
    box_evaluate
    return (box_evaluate,)


@app.cell(hide_code=True)
def check_evaluate(box_evaluate):
    result_evaluate = L.check("evaluate", box_evaluate.value)
    result_evaluate
    return (result_evaluate,)


@app.cell(hide_code=True)
def hints_evaluate():
    L.hints(
        "evaluate",
        [
            "Step 3 of the recipe: `v` collects the new value of `s`, and every action `a` adds its part. How much the policy uses an action is in `pi[s][a]`.",
            "On ROOM, take a tile next to the charger after sweep 1: one action has a $Q$ of −1 + 0 = −1, the three others −1 + (−1) = −2. The random policy counts each with 1/4: 0.25 · (−1) + 0.75 · (−2) = −1.75.",
            "`                v = v + ___`",
        ],
        why="`v = v + pi[s][a] * q`: each action adds its $Q$, counted with the probability that the policy takes it, so `v` ends at the average of the $Q$-values. Value iteration took the largest `q` at this place.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_stable():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Stop when the values hardly change

        How many sweeps are enough? `until_stable(T, pi, gamma, theta)` sweeps until the largest change of a state in a sweep is smaller than `theta`, and gives back the values and the number of sweeps it did. `sweep(T, pi, gamma, V)` does one sweep, as the inside of your `evaluate` does, and gives back the new values, `new`. `sweeps` counts the sweeps, and `change` starts at 0 in every sweep; with `max` it must end up as the largest distance between the new and the old value of one state `s` in this sweep. Replace the two `...`, steps 4 and 5 of the recipe. The check calls `until_stable(T_ROOM, RANDOM, 1.0, 0.01)`, which should stop after 151 sweeps, and five other cases, among them `T_PRIZE`: the corridor of the worked example, where the step onto the charger pays +10, so that values go up as well as down.
        """),
        L.task("Make `until_stable` keep the largest change of a sweep in `change`, and stop once it is smaller than `theta`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_stable():
    box_stable = L.code_box("stable", lines=14)
    box_stable
    return (box_stable,)


@app.cell(hide_code=True)
def check_stable(box_stable):
    result_stable = L.check("stable", box_stable.value)
    result_stable
    return (result_stable,)


@app.cell(hide_code=True)
def hints_stable():
    L.hints(
        "stable",
        [
            "Step 4: `change` starts at 0 in every sweep, and `max` keeps the larger of two numbers, so after the loop over `s` it holds the largest change. A change counts how far a value moved, whichever way it moved. Step 5: the loop may stop once `change` is small.",
            "In the first sweep on ROOM every value goes from 0 to −1: it moved by 1, though `new[s] - V[s]` is −1, and `abs(-1)` is 1. So `change` ends at 1. After sweep 151 the largest change is below 0.01 for the first time.",
            "`            change = max(change, ___)`",
        ],
        why="`change = max(change, abs(new[s] - V[s]))` keeps the largest distance between a new and an old value, and `if change < theta:` stops the loop once that distance is small.",
    )
    return


@app.cell(hide_code=True)
def text_after_stable():
    mo.md(r"""
    With $\theta$ = 0.01 the loop stops after 151 sweeps, while the corner is still at −35.69, 0.31 from −36. A change of 0.01 per sweep does not mean that the values lie within 0.01 of their end: every sweep takes off about 3% of what is left, so about 30 times the last change is still to come. With $\theta$ = 0.001 the corner ends at −35.97, after 224 sweeps.

    ### The same sweeps with NumPy

    For one fixed policy, a sweep is a matrix multiplication. Number the 25 tiles of ROOM row by row. `P[i, j]` is the probability that the robot goes from tile `i` to tile `j` in one step of the policy, and `R[i]` the reward it can expect on tile `i`. Then a sweep is `V = R + gamma * P @ V`: row `i` of `P @ V` is the average value of where tile `i` leads. The cell builds `P` and `R` from the course's `T_ROOM` and `RANDOM`, does 224 sweeps and shows the values, row by row. Run the cell.
    """)
    return


@app.cell
def example_numpy():
    import numpy as np

    _P = np.zeros((25, 25))
    _R = np.zeros(25)
    _at = {}   # the number of each tile, row by row
    for _s in C.T_ROOM:
        _at[_s] = len(_at)
    for _s in C.T_ROOM:
        for _a in C.T_ROOM[_s]:
            for _p, _nxt, _r in C.T_ROOM[_s][_a]:
                _P[_at[_s], _at[_nxt]] = _P[_at[_s], _at[_nxt]] + C.RANDOM[_s][_a] * _p
                _R[_at[_s]] = _R[_at[_s]] + C.RANDOM[_s][_a] * _p * _r
    _V = np.zeros(25)
    for _k in range(224):
        _V = _R + 1.0 * _P @ _V
    for _row in range(5):
        print(np.round(_V[5 * _row:5 * _row + 5], 2))
    return


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    These are the values your `until_stable` gives with $\theta$ = 0.001: −35.97 in the corners, −23.98 next to the charger. A matrix of 25 by 25 is small; a table with a million states would make one of a million by a million, almost all zeros, and then a loop over the states does less work, as long as it takes for each state only the steps that state really has.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_forever():
    mo.md(r"""
    ### Exercise 3. A policy that never arrives

    The policy `UP` always goes up. Below the charger, two tiles reach it: the tile right below it in one step, the tile under that in two. Every other tile goes up to the top row and keeps bumping into the edge there.
    """)
    return


@app.cell(hide_code=True)
def choice_forever_cell():
    choice_forever = L.choice(
        "forever",
        question="What does your `until_stable` do with `UP` on ROOM, with $\\gamma$ = 1 and $\\theta$ = 0.01?",
        options={
            "zero": "It gives back 0 for every tile that never reaches the charger, after one sweep",
            "right": "It never stops: every sweep lowers those values by 1, so the largest change stays 1",
            "minus": "It stops after a few sweeps with −1 for every tile in the top row, since a step into the edge leaves the robot where it is and costs only once",
            "five": "It stops after 5 sweeps, once every tile has reached the top row",
        },
        correct="right",
        why={
            "right": "With `UP`, a robot in the top row bumps into the edge at every step and never reaches the charger: −1 per step, without end. After $k$ sweeps those values count $k$ steps, so every sweep lowers them by 1, the largest change never gets below $\\theta$, and the loop never gets to `return`. Only the two tiles below the charger settle, at −1 and −2. With $\\gamma$ = 0.9 the values would settle at −10, since 1 + 0.9 + 0.81 + … = 10. With $\\gamma$ = 1, policy evaluation needs a policy that reaches an end from every state, as the random robot does.",
            "zero": "A tile that never reaches the charger pays −1 at every step: does its value stay at 0 after the first sweep, which counts one step?",
            "minus": "A step into the edge costs −1 too, and in the top row the robot takes such a step every time: how many steps does sweep $k$ count?",
            "five": "Once the robot is in the top row it still takes steps, and each costs −1: does the largest change ever drop below $\\theta$?",
        },
    )
    choice_forever
    return (choice_forever,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [When steps can slip: value iteration](/course/open/rl-value-iteration?lang=en), the values were those of the best policy, found with the largest $Q$. Your `evaluate` works out the values of any policy you give it, with an average over its actions, and your `until_stable` decides when the sweeps have done enough. The values of the random robot are minus its average number of steps to the charger, from −24 next to it to −36 in a corner. Take in every tile the action with the largest $Q$ under these values, and every tile already heads for the charger: the next lesson, *Better and better: policy iteration*, turns that into a method that makes a policy better step by step. Later in this part, Monte Carlo and TD learning estimate the same $V_\pi$ without the table `T`, from walks the robot really makes.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_forever, result_evaluate, result_stable):
    L.wrap_up(result_evaluate, result_stable, choice_forever)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
