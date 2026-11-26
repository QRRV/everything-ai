# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-policy-iteration. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the room and the hall, the course's policy iteration and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rl-policy-iteration", "en")
    C = L.checks   # the room and the hall with their tables, the course's policy iteration, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [What is a policy worth? Policy evaluation](/course/open/rl-policy-evaluation?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_v():
    quick_v = L.choice(
        "quick-v",
        question="On ROOM, the robot's grid, a tile next to the charger has four neighbors: the charger (value 0), a tile in the middle of an edge (−32) and two tiles diagonal to the charger (−30 each). Every step costs −1, and $\\gamma$ = 1. What is the value of the tile under the random policy?",
        options={
            "best": "−1, since one of the four steps goes straight onto the charger",
            "right": "−24, the average of its four $Q$-values",
            "values": "−23, the average of the values of its neighbors",
            "added": "−96, its four $Q$-values added up",
        },
        correct="right",
        why={
            "right": "Each step costs −1, so the four $Q$-values are −1 + 0, −1 − 32, −1 − 30 and −1 − 30. The random policy counts each with 1/4: (−1 − 33 − 31 − 31) / 4 = −24.",
            "best": "−1 is the $Q$ of the step onto the charger, but the random policy takes that step only one time in four: how does it count the other three? See [What is a policy worth? Policy evaluation](/course/open/rl-policy-evaluation?lang=en).",
            "values": "The neighbors are a step away, and that step costs something: what does it add to each value before you take the average? See [What is a policy worth? Policy evaluation](/course/open/rl-policy-evaluation?lang=en).",
            "added": "The robot takes one of the four steps, not all four at once: how much does each count with the random policy? See [What is a policy worth? Policy evaluation](/course/open/rl-policy-evaluation?lang=en).",
        },
    )
    quick_v
    return (quick_v,)


@app.cell(hide_code=True)
def quick_check_theta():
    quick_theta = L.choice(
        "quick-theta",
        question="Policy evaluation of the random policy on ROOM stops after 151 sweeps with $\\theta$ = 0.01. What does that stop tell you?",
        options={
            "within": "Every value now lies within 0.01 of the value it would end at after many more sweeps",
            "right": "The last sweep moved no value by 0.01 or more, though the end can be farther",
            "final": "More sweeps would not change any value",
            "exact": "Every value moved by exactly 0.01 in the last sweep",
        },
        correct="right",
        why={
            "right": "The stop looks at the last change only. On ROOM every sweep takes off about 3% of what is left, so after 151 sweeps the corner is still 0.31 from −36, about 30 times the last change.",
            "within": "The stop compares two sweeps, not a sweep with the end: if every sweep takes off only about 3% of what is left, how much is still left once a sweep moves a value by 0.01? See [What is a policy worth? Policy evaluation](/course/open/rl-policy-evaluation?lang=en).",
            "final": "A change below 0.01 is still a change: does the next sweep move the values not at all, or only a little? See [What is a policy worth? Policy evaluation](/course/open/rl-policy-evaluation?lang=en).",
            "exact": "$\\theta$ is a bound on the largest change, not the size of every change: what does `change < theta` say about the largest change of the last sweep? See [What is a policy worth? Policy evaluation](/course/open/rl-policy-evaluation?lang=en).",
        },
    )
    quick_theta
    return (quick_theta,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [What is a policy worth? Policy evaluation](/course/open/rl-policy-evaluation?lang=en), you worked out what a policy is worth from every state. That is half of the job: an agent does not only want to know how good its policy is, it wants a good policy. This lesson turns the values of a policy into a better policy, and doing that again and again gives a best policy after only a few rounds. The same back and forth between values and a policy comes back in almost every method of reinforcement learning, also when the agent has no table and learns from what it does.

    **An everyday example.** You cycle to work by a fixed route, and you know how long the rest of the trip takes from every junction on it. At each junction you can ask: if I turn another way here, and then ride on as I always do, am I there sooner? Wherever the answer is yes, you change your route. Then you time the new route, and ask the same question again, until no junction offers a quicker turn.
    """)
    return


@app.cell(hide_code=True)
def predict_steps():
    guess_steps = L.guess(
        "steps",
        question="Under the random policy, a corner of ROOM has the value −36: the robot needs 36 steps to the charger on average. Now its policy changes once: in every tile it takes the action with the largest $Q$ under these values of the random policy, and it keeps that action from then on. How many steps does it need from a corner now?",
        options={
            "same": "About 36, as before, since the values belong to the random policy and the policy changed only once",
            "half": "About 18: one change makes it better, but not the best",
            "right": "4",
            "eight": "About 8, since it still makes a detour here and there",
        },
        reveal="**4.** One change is enough here: from every tile the robot now goes straight to the charger, so from a corner it takes the 4 steps of the shortest way, where the random robot needed 36 on average. The values of the random policy say little about the number of steps of the new policy, but they still point the right way: a tile closer to the charger has a higher value. *The idea* shows why the action with the largest $Q$ never makes a policy worse.",
    )
    guess_steps
    return (guess_steps,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The values $V_\pi$ of a policy $\pi$ say what each state is worth if you follow $\pi$. With them you can work out, for every action, what it is worth to take that action once and follow $\pi$ from then on: that is $Q_\pi(s, a)$, the inner part of the Bellman equation of a policy. **Policy improvement** makes a new policy $\pi'$ that takes in every state the action with the largest $Q_\pi(s, a)$:
        """),
        L.formula(r"\pi'(s) = \arg\max_a \sum [[data:p]] \left( [[data:r]] + [[hyper:\gamma]] \, [[pred:V_\pi(s')]] \right)"),
        mo.md(r"""
        $\arg\max_a$ is the action $a$ for which what follows is largest. Why is $\pi'$ at least as good as $\pi$? In state $s$, $V_\pi(s)$ is the average of the $Q_\pi(s, a)$, each counted with $\pi(a \mid s)$, and the largest $Q$ is at least as large as that average. So taking the best action in $s$ once, and following $\pi$ from then on, is worth at least $V_\pi(s)$. That holds in every state, so also in the state where that first step lands, and in the one after: take the best action at every step, and the value of $\pi'$ is at least that of $\pi$ in every state. This is the **policy improvement theorem**. And if the improvement changes nothing, every value of $\pi$ is already its own largest $Q$: $V_\pi$ then fits the Bellman equation of value iteration, and $\pi$ is a best policy.

        **Policy iteration** puts the two steps together: evaluate the policy, improve it, evaluate the improved policy, improve again, until the improvement gives back the policy it got. A **round** of policy iteration is one whole policy evaluation plus one improvement; the rounds of value iteration were single sweeps. Each round gives a policy at least as good as the one before, and better somewhere unless its values were already the best. Once they are, a fixed choice among equally good actions, such as always the first, gives back the same policy one round later; exercise 3 shows why the fixed choice matters. There are only so many policies that take one action per state, so the rounds end, at a best policy.

        [When steps can slip: value iteration](/course/open/rl-value-iteration?lang=en) is a short form of the same idea: each of its sweeps takes the largest $Q$ right away, as if every policy evaluation lasted a single sweep. In between lie many mixes, such as a few sweeps of evaluation, then an improvement. **Generalized policy iteration** is the name for every method in which values and a policy make each other better in turn: the values are made to fit the policy, and the policy is made to take the best action under the values.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **policy improvement** | | a new policy that takes in every state the action with the largest $Q$ under the values of the old policy | `improve` |
        | $\pi'$ | pi prime | the improved policy | `new` |
        | $\arg\max_a$ | arg max over a | the action $a$ for which what follows is largest | `np.argmax` |
        | **policy improvement theorem** | | the improved policy is at least as good as the old one, in every state | |
        | **policy iteration** | | evaluate and improve in turn, until the improvement gives back the policy it got | `policy_iteration` |
        | **round** | | in policy iteration: one whole policy evaluation plus one improvement | `rounds` |
        | **generalized policy iteration** | | every method in which values and a policy make each other better in turn | |
        """),
        L.watch_out("Policy improvement never makes a value worse, but a round can leave many values as they are: the theorem only says that the new policy is at least as good in every state."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Policy iteration on HALL, a hall of 5 by 6 tiles with the charger at the top left and the robot's start S at the bottom left. Eight tiles are puddles: a step that ends in a puddle costs −10 instead of −1. Round 1 evaluates the random policy; every later round shows the policy after one more improvement, its <span class="c-pred">values</span> and its way from S. The bars beside it count the tiles that get a new action at the end of each round.
    """)
    return


@app.cell(hide_code=True)
def explore_rounds_cell():
    explore_rounds = L.explore("rounds", "round", lambda _v: C.play_frame(int(_v), L.lang), start=1,
                               values=C.PLAY_ROUNDS, role="",
                               task="Go through the rounds. In which round does the robot first go around the puddles from S, and what does that save? Follow row 2 from round 5 to round 8: in which order do its tiles turn right?")
    explore_rounds
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Round 1 takes 972 sweeps: from S the random robot needs about 114 steps on average, about 28 of them into a puddle at 9 extra each, so S starts at −365.64. After one improvement the robot goes straight up from S, through the puddle at the left: 4 steps, −13. The improvements at the end of rounds 2 to 8 then change 9, 5, 3, 1, 1, 4 and 3 tiles. The dry way along row 2 spreads one tile per round, from the charger back toward S: each improvement looks only one step ahead, with the values of the policy it improves. In round 8 the robot goes around the puddles from S, 12 steps without a puddle: −12, one better than −13. Round 9 gives no tile a new action, so the policy is stable. The later evaluations took 10 to 16 sweeps each: a policy that goes to the charger without detours settles fast.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Policy iteration, with a table `T`:

    1. Start with a policy, such as the random policy.
    2. Evaluate it: sweeps of policy evaluation until the largest change is below $\theta$.
    3. Improve it: in every state, work out the $Q$ of every action from these values, and take the action with the largest $Q$; of equally good actions, always the first.
    4. If the improved policy is the same as the old one, stop: it is a best policy. Otherwise go on at step 2 with the improved policy.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    One improvement on ROOM, from the values of the random policy with $\theta$ = 0.001. The cell works out the four $Q$-values of the corner `(0, 0)` from these values, step 3 of the recipe for one state. Then it improves every tile with the course's `improve` and evaluates the improved policy: it shows the policy in the corner, the value of the corner and the number of sweeps, and the value of a tile next to the charger. Run the cell.
    """)
    return


@app.cell
def example_corner():
    # 2. the values of the random policy
    _V, _n = C.until_stable(C.T_ROOM, C.RANDOM, 1.0, 0.001)
    # 3. the Q of every action in the corner, from these values
    for _a in C.T_ROOM[(0, 0)]:
        _q = 0
        for _p, _nxt, _r in C.T_ROOM[(0, 0)][_a]:
            _q = _q + _p * (_r + 1.0 * _V[_nxt])
        print(_a, round(_q, 2))
    # 3. the improved policy in every tile, and its values
    _better = C.improve(C.T_ROOM, _V, 1.0)
    _W, _m = C.until_stable(C.T_ROOM, _better, 1.0, 0.001)
    print("corner:", _better[(0, 0)], round(_W[(0, 0)], 2), "after", _m, "sweeps")
    print("next to the charger:", round(_W[(1, 2)], 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Up and left bump into the edge: the robot stays in the corner, so their $Q$ is −1 − 35.97 = −36.97. Down and right lead to tiles worth −33.97 under the random policy, so their $Q$ is −34.97. Down and right are equally good, and `improve` takes the first of them, down. After the improvement every tile heads straight for the charger. Evaluating the new policy takes only 5 sweeps: after 4 sweeps every way to the charger is counted in full, and the fifth sweep changes nothing. The corner is at −4, the tile next to the charger at −1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_improve():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The action with the largest Q

        `improve(T, V, gamma)` makes the improved policy from the values `V`: in every state `s`, probability 1.0 for the action with the largest $Q$, and 0.0 for every other action. `actions` holds the actions of `s`, in the order of `T[s]`. For each action `a`, the loop over `p, nxt, r` works out `q`, as in value iteration, and `qs.append(q)` puts it in the list `qs`, in the same order as `actions`. Replace the `...` so that `best` is the action with the largest $Q$; of equally good actions, take the first. The loop at the end then gives `best` probability 1.0 in `pi[s]`. The check calls `improve(T_ROOM, V_ROOM, 1.0)`, with `V_ROOM` the values of the random policy, and five other cases, among them HALL and the corridor of the lesson before.
        """),
        L.task("Make `improve` give every state probability 1.0 for its action with the largest Q."),
    ])
    return


@app.cell(hide_code=True)
def exercise_improve():
    box_improve = L.code_box("improve", lines=18)
    box_improve
    return (box_improve,)


@app.cell(hide_code=True)
def check_improve(box_improve):
    result_improve = L.check("improve", box_improve.value)
    result_improve
    return (result_improve,)


@app.cell(hide_code=True)
def hints_improve():
    L.hints(
        "improve",
        [
            "Step 3 of the recipe: `qs` holds the $Q$ of every action, in the order of `actions`. `best` must be an action, such as `\"down\"`, not a number: find the place of the largest $Q$ in `qs`, then the action at that place in `actions`.",
            "In value iteration you found the place of the largest $Q$ with NumPy, the first place if several are equally large. In the corner of ROOM, `qs` is about [−36.97, −34.97, −36.97, −34.97], so the place is 1, and the action at place 1 is down.",
            "`        best = ___`",
        ],
        why="`best = actions[np.argmax(qs)]`: `np.argmax(qs)` is the place of the largest $Q$, the first of equally large ones, and `actions` holds the action at each place. The improved policy takes that action with probability 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_iteration():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Evaluate, improve, repeat

        `policy_iteration(T, gamma, theta)` runs policy iteration from the random policy and gives back the stable policy, its values and the number of rounds. The first lines make the random policy `pi`: in every state `s`, each action `a` gets probability `1 / len(T[s])`. `rounds` counts the rounds. Every round evaluates `pi` with `until_stable(T, pi, gamma, theta)`, as you wrote it in the lesson before, which gives back the values `V` and the number of `sweeps`. Replace the two `...`: `new` must be the improved policy, from `improve`, the course's version of exercise 1, and the loop must stop once `new` is the same policy as `pi`, step 4 of the recipe. The check calls `policy_iteration(T_ROOM, 1.0, 0.01)` and five other cases, among them HALL, the hall of the play block.
        """),
        L.task("Make `policy_iteration` improve the policy every round, and stop once the improvement gives back the same policy."),
    ])
    return


@app.cell(hide_code=True)
def exercise_iteration():
    box_iteration = L.code_box("iteration", lines=14)
    box_iteration
    return (box_iteration,)


@app.cell(hide_code=True)
def check_iteration(box_iteration):
    result_iteration = L.check("iteration", box_iteration.value)
    result_iteration
    return (result_iteration,)


@app.cell(hide_code=True)
def hints_iteration():
    L.hints(
        "iteration",
        [
            "Step 3: `new` comes from `improve`, which needs the table, the values of `pi` and the discount factor. Step 4: the loop may stop once `new` and `pi` are the same policy, in every state and for every action; `if` checks that.",
            "Two dictionaries are equal when they have the same keys with the same values, and Python compares them with the same sign as two numbers. On ROOM, the improvement at the end of round 3 gives back the policy that round 3 evaluated, so your function stops after 3 rounds.",
            "`        if ___:`",
        ],
        why="`new = improve(T, V, gamma)` improves the policy with the values of this round, and `if new == pi:` stops once the improvement gives back the policy it got: then `pi` is a best policy, and `V` its values.",
    )
    return


@app.cell(hide_code=True)
def text_after_iteration():
    mo.md(r"""
    On ROOM your function stops after 3 rounds, though round 2 already has the best values: −4 in a corner, −1 next to the charger. The improvement at the end of round 2 still changes four tiles, such as tile `(1, 0)`. With the values of the random policy it went right; now down and right both have a $Q$ of −3, and `improve` takes the first, down. Round 3 evaluates that policy, finds the same values, and the improvement changes nothing. On HALL your function stops after 9 rounds, with 1,068 sweeps in all, 972 of them for the random policy. Value iteration needs 16 sweeps on HALL with the same $\theta$: policy iteration needs few rounds, but every round holds a whole policy evaluation.

    ### The same improvement with NumPy

    For every action, the table can be written as a matrix, as in the lesson before. Number the 30 tiles of HALL row by row. `P[i]` holds the probability of going from one tile to another with action number `i`, and `R[i]` the reward to expect. Then `R[i] + gamma * P[i] @ v` gives the $Q$ of that action in every state at once, and the array `Q`, with a row for each of the four actions, holds them all. `np.argmax(Q, axis=0)` gives for every column, every state, the place of its largest $Q$. The cell builds `P` and `R` from the course's `T_HALL`, where `enumerate` gives each action of a tile with its number `i`, 0 to 3. Then it improves the random policy of HALL, `RANDOM_HALL`, this way, from its values, and prints an arrow for the action of every tile, row by row. Run the cell.
    """)
    return


@app.cell
def example_numpy():
    import numpy as np

    _at = {}   # the number of each tile, row by row
    for _s in C.T_HALL:
        _at[_s] = len(_at)
    _P = np.zeros((4, 30, 30))
    _R = np.zeros((4, 30))
    for _s in C.T_HALL:
        for _i, _a in enumerate(C.T_HALL[_s]):
            for _p, _nxt, _r in C.T_HALL[_s][_a]:
                _P[_i, _at[_s], _at[_nxt]] = _P[_i, _at[_s], _at[_nxt]] + _p
                _R[_i, _at[_s]] = _R[_i, _at[_s]] + _p * _r
    _V, _n = C.until_stable(C.T_HALL, C.RANDOM_HALL, 1.0, 0.001)
    _v = np.array([_V[_s] for _s in C.T_HALL])
    _Q = np.array([_R[_i] + 1.0 * _P[_i] @ _v for _i in range(4)])
    _best = np.argmax(_Q, axis=0)
    _signs = "↑↓←→"   # up, down, left, right: the order of the actions
    for _row in range(5):
        _line = ""
        for _c in range(6):
            if C.HALL[_row][_c] == "G":
                _line = _line + "G "
            else:
                _line = _line + _signs[_best[6 * _row + _c]] + " "
        print(_line)
    return


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    These are the arrows of round 2 of the play block. Under the values of the random policy, a tile closer to the charger looks better, so every tile heads for the charger by the shortest way, puddle or not: column 0, and column 1 above the bottom row, go straight up, through the top row of puddles. What a puddle costs only shows once round 2 evaluates this policy.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_ties():
    mo.md(r"""
    ### Exercise 3. Equally good actions

    On ROOM, round 2 already had the best values, yet its improvement changed four tiles, each from one best action to another just as good. `improve` takes the first of equally good actions, so after one such change the policy stays as it is.
    """)
    return


@app.cell(hide_code=True)
def choice_ties_cell():
    choice_ties = L.choice(
        "ties",
        question="Suppose `improve` took one of the equally good actions at random, with a new draw in every round. What could that do to your `policy_iteration`?",
        options={
            "worse": "It would end with a worse policy, since a random action is sometimes a bad one",
            "sweeps": "Each evaluation would take many more sweeps, since a policy with random choices gives every state values that are noisy and slow to settle down",
            "right": "It could run for a very long time: the policy can keep switching between equally good actions, so the new one keeps differing from the old",
            "same": "Nothing: it would still stop after 3 rounds on ROOM",
        },
        correct="right",
        why={
            "right": "With a new draw in every round, tile `(1, 0)` can get down in one round and right in the next. Both are best, so the values stay the same, but the policy keeps changing, and the comparison of `new` and `pi` rarely finds the two equal. Once the values are the best, 16 tiles of ROOM have two best actions, and a new draw repeats all 16 choices only one round in 65,536 (2 to the 16th), so the loop runs about 65,536 rounds on average. A fixed choice among equal actions, as `np.argmax` makes, prevents this; another way is to stop once the values no longer change.",
            "worse": "Every action the draw picks from has the largest $Q$: can a draw among best actions pick a bad one?",
            "sweeps": "Each round still evaluates one fixed policy: does the draw happen during the evaluation, or only in the improvement?",
            "same": "In round 3 on ROOM, tile `(1, 0)` has two best actions again: does a new draw always give the action it got in round 2?",
        },
    )
    choice_ties
    return (choice_ties,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [What is a policy worth? Policy evaluation](/course/open/rl-policy-evaluation?lang=en), you worked out the values of a given policy. Your `improve` turns those values into a policy that is at least as good, by taking in every state the action with the largest $Q$, and your `policy_iteration` repeats evaluation and improvement until the policy no longer changes: then it is a best policy, on ROOM after 3 rounds and on HALL after 9. Values that fit a policy, and a policy that takes the best action under the values: that back and forth, generalized policy iteration, also drives the coming lessons, where the table `T` is gone. The next lesson, *Learning from whole games: Monte Carlo prediction*, estimates the values of a policy from walks the robot really makes.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_ties, result_improve, result_iteration):
    L.wrap_up(result_improve, result_iteration, choice_ties)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
