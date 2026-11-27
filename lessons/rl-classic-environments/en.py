# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-classic-environments. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the course's fixed policies and its Q-learner with bins, and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-classic-environments", "en")
    C = L.checks   # the course's policies and runs in CartPole and MountainCar, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [One interface for every environment: Gymnasium](/course/open/rl-gymnasium?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_table():
    quick_table = L.choice(
        "quick-table",
        question="A world of Gymnasium has `Discrete(500)` as its observation space and `Discrete(6)` as its action space. What shape does a Q-table for it get?",
        options={
            "flip": "6 rows and 500 columns",
            "right": "500 rows, one per observation, and 6 columns",
            "plus": "506 rows: one for each observation and one for each action",
            "one": "500 rows with one number each",
        },
        correct="right",
        why={
            "right": "A row per observation and a column per action: 500 × 6 = 3,000 Q's, as in Taxi. In this lesson the observations are real numbers, and a table first needs them cut into bins.",
            "flip": "The agent looks up the row of its observation and then compares the actions within that row: which of the two needs the rows? See [One interface for every environment: Gymnasium](/course/open/rl-gymnasium?lang=en).",
            "plus": "A Q belongs to a pair: one observation with one action. How many pairs are there? See [One interface for every environment: Gymnasium](/course/open/rl-gymnasium?lang=en).",
            "one": "Q-learning keeps a Q for every action of an observation, to compare them: how many columns does that take? See [One interface for every environment: Gymnasium](/course/open/rl-gymnasium?lang=en).",
        },
    )
    quick_table
    return (quick_table,)


@app.cell(hide_code=True)
def quick_check_cut():
    quick_cut = L.choice(
        "quick-cut",
        question="A time limit cuts an episode off after its 500th step, while the task itself goes on. What does `env.step` give as `terminated` and `truncated` for that last step?",
        options={
            "ended": "`terminated` is `True`, `truncated` is `False`",
            "right": "`terminated` is `False`, `truncated` is `True`",
            "both": "Both are `True`, since the episode is over either way",
            "none": "Both are `False`: a time limit sets no flag",
        },
        correct="right",
        why={
            "right": "The task did not end, the clock did: `truncated`. The target of that step keeps γ times the largest Q of the next state. CartPole of this lesson has just such a limit: 500 steps.",
            "ended": "`terminated` says that the task itself ended, such as a car at its goal: did that happen here? See [One interface for every environment: Gymnasium](/course/open/rl-gymnasium?lang=en).",
            "both": "Over is over for the loop, but the two flags tell two reasons apart: which reason holds here? See [One interface for every environment: Gymnasium](/course/open/rl-gymnasium?lang=en).",
            "none": "Without a flag, how would the loop of an episode know that it has to stop? See [One interface for every environment: Gymnasium](/course/open/rl-gymnasium?lang=en).",
        },
    )
    quick_cut
    return (quick_cut,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A new method in RL is worth something only when it does better than the old ones on the same tasks. So researchers test on a few shared worlds, always in the same version, and report how their agent did there. Each classic world became one of these **benchmark environments** because it tests something else. CartPole, from 1983, asks an agent to balance a pole on a cart, with four numbers to go on. MountainCar, from 1990, gives the same −1 at every step until a car reaches a flag, so the reward says nothing about how to get there. Atari games ask an agent to learn from the pixels of a screen, and MuJoCo lets it move bodies with actions that are real numbers. In this lesson you play CartPole and MountainCar, cut their numbers into bins so that a Q-table can learn them, and see what Atari and MuJoCo ask that a table cannot give.

    **An everyday example.** A driving test checks parking, merging onto a highway and an emergency stop. Each part tests something else, and parking well says little about the emergency stop. A method of RL that only did well on CartPole has passed the parking.
    """)
    return


@app.cell(hide_code=True)
def predict_flag():
    guess_flag = L.guess(
        "flag",
        question="In MountainCar, a random agent pushes left, not at all or right, at random, at every step. In how many of 1,000 episodes does it reach the flag within the time limit of 200 steps, do you think?",
        options={
            "half": "About half of them",
            "tenth": "About 1 in 10",
            "right": "Not a single one",
            "most": "Almost all, since 200 steps leave plenty of time",
        },
        reveal="**Not a single one.** In 1,000 episodes of random pushes, the car never reaches the flag: the pushes undo each other, and the engine alone is too weak for the slope, since pushing right at every step does not get there either. A fixed policy that pushes the way the car moves reaches the flag in every one of 100 episodes, in 120 steps on average. For a learner the −1 at every step says nothing until it has reached the flag once: that is what MountainCar tests.",
    )
    mo.vstack([mo.Html(C.world_picture(L.lang)), guess_flag])
    return (guess_flag,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Real numbers.** In CartPole and MountainCar an observation is a few real numbers, and the observation space is a `Box`: `Box(low, high, (4,))` holds 4 numbers, each between its low and its high. In CartPole, `obs[0]` is the position of the cart, on a track from −2.4 to 2.4, `obs[1]` its velocity, `obs[2]` the angle of the pole in radians, positive when it leans to the right, and `obs[3]` its angular velocity: how fast the angle changes, positive when the pole falls to the right. The action space is `Discrete(2)`: 0 pushes the cart to the left, 1 to the right. Every step gives +1, and a step lasts 0.02 seconds, so 500 steps are 10 seconds of balancing. In MountainCar, `obs[0]` is the position of the car, from −1.2 to 0.6, and `obs[1]` its velocity, from −0.07 to 0.07. The engine pulls with 0.001 and the slope with up to 0.0025, so the car cannot drive up the hill: it has to swing back and forth to build up speed.

        | World | observation space | action space | reward | ends when | time limit |
        |---|---|---|---|---|---|
        | CartPole-v1 | `Box`, 4 numbers | `Discrete(2)`: 0 left, 1 right | +1 every step | pole past 12°, or cart past ±2.4 | 500 steps |
        | MountainCar-v0 | `Box`, 2 numbers | `Discrete(3)`: 0 left, 1 nothing, 2 right | −1 every step | car at the flag, 0.5 | 200 steps |

        **A sparse reward.** In CartPole every step tells the agent something: the pole is still up. In MountainCar every step gives −1, whatever the agent does, and only reaching the flag stops the −1s. Until the agent has reached the flag once, the reward says nothing about which push was better. A reward that stays the same almost everywhere and tells something only at rare moments, such as at the goal, is a **sparse reward**, and it makes exploration hard.

        **Bins.** A Q-table has a row for every observation, and a real number can take endlessly many values. **Discretization** cuts each number into bins. You choose a range and the edges between the bins; the bin of a number is how many edges lie below it. With 12 bins for the position of MountainCar, from −1.2 to 0.6, the 11 edges lie 0.15 apart, at −1.05, −0.9, and so on up to 0.45. The position −0.5 lies above 4 of them, so it lands in bin 4. A number with one bin has no edges and always lands in bin 0: the table ignores it. The bins of all the numbers of an observation together pick one row, like the digits of a number: with 10 bins for each of four numbers, bins 3, 0, 7 and 2 give row 3072. In general, start at row 0, and for each number, multiply the row by that number's count of bins and add its bin. With 4 bins for both numbers of MountainCar, position bin 2 and velocity bin 1 give row 2 × 4 + 1 = 9. The table gets as many rows as all the counts of bins multiplied: 12 × 12 = 144 for MountainCar with 12 bins each. CartPole in this lesson cuts only its pole: 1 bin for the position of the cart and 1 for its velocity, and as many as you choose for the angle and the angular velocity.
        """),
        mo.Html(C.bins_picture(L.lang)),
        mo.md(r"""
        **Pixels and bodies.** The Arcade Learning Environment (Bellemare and colleagues, 2013) turned games of the Atari 2600, a game console from 1977, into environments. An observation is the screen, 210 by 160 pixels, each one of 128 colors; an action is one of at most 18 positions of the joystick and its button; the reward comes from the score. No table can hold a row for every screen: there are more possible screens than atoms in the universe. In 2015, DQN (Mnih and colleagues) put a neural network in the place of the table. With the same network and settings for every game, it learned 49 games from the pixels and the score alone, and in 29 of them it scored at least 75% of what a professional human games tester scored. **MuJoCo** (Todorov, Erez and Tassa, 2012) is a physics simulator for bodies with joints. In its Hopper, a body on one leg sees 11 numbers and gives back 3, a turning force for each of its three joints, each a real number from −1 to 1: the action space is `Box(-1, 1, (3,))`. A Q-table needs a column for every action; cutting each force into 10 bins already gives 10 × 10 × 10 = 1,000 columns, and the Ant of MuJoCo, with 8 joints, would need 100,000,000. The DeepMind Control Suite (Tassa and colleagues, 2018) holds more tasks of this kind, also built on MuJoCo. Atari and MuJoCo need packages that the course does not install, and learning them takes a neural network instead of a table, as DQN did.
        """),
        mo.Html(C.bench_picture(L.lang)),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **benchmark environment** | | a world that many researchers test on, in a fixed version, so that their results can be compared | `gym.make("CartPole-v1")` |
        | **CartPole** | cart pole | balancing a pole on a cart from four numbers, with two actions: push left or right | `"CartPole-v1"` |
        | **sparse reward** | | a reward that tells something only at rare moments, such as at the goal | |
        | **discretization** | | cutting real numbers into bins, so that a table gets a row for each combination of bins | `row_of(obs, edges)` |
        | **MuJoCo** | mu-jo-co | a physics simulator for bodies with joints, with actions that are real numbers | |
        | `Box(low, high, shape)` | box | real numbers, each between its low and its high | `env.observation_space` |
        | edges | | the borders between the bins of a number | `edges` |
        """),
        L.watch_out("Bins are a choice, and they decide what the table can tell apart. A row covers every observation in its bins at once, and the table learns one Q per action for all of them together."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Q-learning in a world whose observations are real numbers:

    1. For each number of the observation, choose a range and a count of bins; the edges lie between the bins, equally spread over the range. A number that does not matter gets one bin, and no edges.
    2. The bin of a number is how many of its edges lie below it.
    3. The row of an observation: start at 0, and for each number, multiply the row by its count of bins and add its bin.
    4. Make a Q-table of zeros with as many rows as all the counts of bins multiplied, and a column per action.
    5. Learn with Q-learning as before, with the row of the observation wherever the table needs a row.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    MountainCar, with `env.reset(seed=0)`, and a fixed policy that pushes the way the car moves: right, 2, when its velocity is 0 or more, and left, 0, when it is below 0. The cell makes the world, prints its spaces, and plays one episode; every 20 steps, and at the end, it prints the position and the velocity. `_steps % 20` is the remainder of `_steps` divided by 20, so `_steps % 20 == 0` is true at step 20, 40, 60 and so on. Run the cell.
    """)
    return


@app.cell
def example_car():
    import gymnasium as gym

    _env = gym.make("MountainCar-v0")
    print("observation_space:", _env.observation_space)
    print("action_space:", _env.action_space)
    _obs, _info = _env.reset(seed=0)
    print("start: position", round(float(_obs[0]), 3), "velocity", round(float(_obs[1]), 4))
    _steps = 0
    _terminated, _truncated = False, False
    while not (_terminated or _truncated):
        if _obs[1] >= 0:
            _action = 2   # push right, the way the car moves
        else:
            _action = 0   # push left
        _obs, _reward, _terminated, _truncated, _info = _env.step(_action)
        _steps += 1
        if _steps % 20 == 0 or _terminated:
            print("step", _steps, "position", round(float(_obs[0]), 3), "velocity", round(float(_obs[1]), 4))
    print("terminated", _terminated, "truncated", _truncated, "after", _steps, "steps")
    return (gym,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The car starts at −0.473, standing still, so the policy pushes right. It climbs to about −0.30, rolls back, and from then on the policy pushes left: past the bottom and up the left hill, to −1.113 at step 80. There it turns, and with the slope and the engine pulling the same way, it races down and up the right hill, to the flag at step 122: `terminated` is `True`. Over 100 episodes this policy reaches the flag every time, in 120 steps on average; pushing right at every step never gets there, and random pushes did not in 1,000 episodes.

    Now the bins of one observation, by hand. At step 60 the car is at position −0.758 with velocity −0.0303. With 12 bins for each number, the edges of the position lie at −1.05, −0.9, −0.75, …: −0.758 lies above the first two, so its bin is 2. The edges of the velocity lie 0.0117 apart, at −0.0583, −0.0467, −0.035, −0.0233, …: −0.0303 lies above the first three, so its bin is 3. The row of this observation is 2 × 12 + 3 = 27, one of the 144 rows of the table.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_policy():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A fixed policy for CartPole

        `pole_policy(obs)` gets an observation of CartPole and gives back an action: 1 pushes the cart to the right, 0 to the left. Of the four numbers of `obs`, `obs[0]` is the position of the cart, `obs[1]` its velocity, `obs[2]` the angle of the pole and `obs[3]` its angular velocity. A positive angle leans to the right, and a positive angular velocity means that the pole falls to the right. Replace the `...` with a test that is true when the pole falls to the right, so that the cart is pushed under it.

        The check calls `hold(pole_policy, 0)`: it plays one episode of CartPole with your policy, the world seeded with `env.reset(seed=0)`, and gives the number of steps until the pole falls or the cart runs off the track, or until the time limit cuts the episode off, at 500. Your policy passes when the pole stays up for at least 100 steps, with seed 0 and with seeds 1 to 4.
        """),
        L.task("Make `pole_policy` push the cart toward the side the pole falls to."),
    ])
    return


@app.cell(hide_code=True)
def exercise_policy():
    box_policy = L.code_box("pole_policy", lines=4)
    box_policy
    return (box_policy,)


@app.cell(hide_code=True)
def check_policy(box_policy):
    result_policy = L.check("pole_policy", box_policy.value)
    result_policy
    return (result_policy,)


@app.cell(hide_code=True)
def hints_policy():
    L.hints(
        "pole_policy",
        [
            "The four numbers of `obs` are, in order, the position and the velocity of the cart, and the angle and the angular velocity of the pole. Which of them says which way the pole is falling right now, and what is its sign when it falls to the right?",
            "The angle says where the pole is; the angular velocity says where it is going. Index 3 of `obs` is the angular velocity, and it is above 0 when the pole falls to the right.",
            "`    if ___:`",
        ],
        why="`if obs[3] > 0:` pushes the cart to the right exactly when the pole falls to the right, so the cart keeps getting under it. Over 100 episodes this one line keeps the pole up for 198 steps on average, until the cart runs off the track.",
    )
    return


@app.cell(hide_code=True)
def text_after_policy():
    mo.md(r"""
    In the picture, the angle with your policy and with random pushes, in the episode with seed 0. Your policy reacts to where the pole is going, not to where it is. Over 100 episodes, pushing toward the side the pole leans keeps it up for 41 steps on average: that policy keeps pushing while the pole is already swinging back, so every swing is bigger than the one before, 2.7°, 4.8°, 11.4° and 13.3° with seed 0, and then the pole is down. The angular velocity alone holds it for 198 steps, and then the cart runs off the track, in every one of the 100 episodes: the pole is still up, but the pushes have carried the cart to one side. Both together, `3 * obs[2] + obs[3] > 0`, keep the pole up and the cart on the track for all 500 steps in every one of 100 episodes. One line and four numbers are enough for CartPole, if you know the line. A learner does not: it has to find a policy from the rewards alone.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_bin():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The bin of a number

        `bin_of(x, edges)` gives back the bin of the number `x`: how many of the edges in `edges` lie below it, step 2 of the recipe. `edges` is a NumPy array of edges, from small to large. Comparing an array with a number gives an array of `True` and `False`, one for each number of the array, and `np.sum` counts the `True`s, since `True` counts as 1; `int` makes a plain whole number of that count. Replace the `...` with the comparison whose `True`s are the edges below `x`.

        The check calls `bin_of(-0.5, pos_12)`, with `pos_12` the 11 edges of the position of MountainCar with 12 bins: the example of The idea, bin 4. Then four other cases: a position below every edge, a position above every edge, the angle 0.05 of CartPole with `angle_12`, its 11 edges with 12 bins, and the number 0.3 with `no_edges`, an empty array: a number with one bin.
        """),
        L.task("Make `bin_of` count the edges that lie below `x`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_bin():
    box_bin = L.code_box("bin_of", lines=2)
    box_bin
    return (box_bin,)


@app.cell(hide_code=True)
def check_bin(box_bin):
    result_bin = L.check("bin_of", box_bin.value)
    result_bin
    return (result_bin,)


@app.cell(hide_code=True)
def hints_bin():
    L.hints(
        "bin_of",
        [
            "Step 2 of the recipe: count the edges that lie below `x`. `np.sum` counts the `True`s of an array of `True` and `False`, and `int` makes a whole number of the count. Which comparison of `edges` and `x` is `True` for exactly the edges below `x`?",
            "`np.array([1, 4, 9]) > 3` gives `[False, True, True]`: one answer for each number of the array. In the example, 4 edges of `pos_12` lie below −0.5: −1.05, −0.9, −0.75 and −0.6. Which way around should the comparison go, so that those four are the `True`s?",
            "`    return int(np.sum(___))`",
        ],
        why="`edges < x` is `True` for every edge below `x`, and `np.sum` counts them: 4 for −0.5 with `pos_12`, 0 below every edge, 11 above every edge, and 0 for a number without edges.",
    )
    return


@app.cell(hide_code=True)
def text_after_bin():
    mo.md(r"""
    In the picture, your `bin_of` for seven positions of MountainCar, with 12 bins. Each dot lies between two edges, and its bin counts the edges to its left. Equally spread edges are the simplest choice; nothing stops you from putting more edges where the numbers often are, such as near an angle of 0 in CartPole.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_row():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. A row for every combination of bins

        `row_of(obs, edges)` gives back the row of the observation `obs` in a Q-table: step 3 of the recipe. `edges` is a list with one array of edges for each number of `obs`, so `edges[i]` holds the edges of number `i`, and `len(edges[i])` is how many edges it has. `bin_of` is the course's own solution of exercise 2. `for i in range(len(obs)):` goes through the numbers of `obs` one by one; `row = 0` starts the row at 0, and each turn multiplies `row` and adds the bin of number `i`. Replace the `...`: how many bins number `i` has.

        The check calls `row_of([-0.52, 0.004], car_12)`, with `car_12` the edges of MountainCar with 12 bins for each number. Then four other cases: an observation of CartPole with `cart_12`, 1 bin for each number of the cart and 12 for each number of the pole; four numbers from 0 to 1 with `digits`, 10 bins each, in which the bins are the digits of the row; MountainCar with `car_20`, 20 bins each; and CartPole with `cart_4`.
        """),
        L.task("Make `row_of` multiply the row by the number of bins of each number before adding its bin."),
    ])
    return


@app.cell(hide_code=True)
def exercise_row():
    box_row = L.code_box("row_of", lines=5)
    box_row
    return (box_row,)


@app.cell(hide_code=True)
def check_row(box_row):
    result_row = L.check("row_of", box_row.value)
    result_row
    return (result_row,)


@app.cell(hide_code=True)
def hints_row():
    L.hints(
        "row_of",
        [
            "Step 3 of the recipe: for each number `i` of `obs`, multiply `row` by that number's count of bins, and add its bin, `bin_of(obs[i], edges[i])`. How many bins does a number with `len(edges[i])` edges have?",
            "11 edges cut a range into 12 bins, and an empty array of edges leaves 1 bin: always one more bin than edges. Python multiplies before it adds, so an addition that has to be multiplied as a whole needs parentheses around it.",
            "`        row = row * ___ + bin_of(obs[i], edges[i])`",
        ],
        why="`row = row * (len(edges[i]) + 1) + bin_of(obs[i], edges[i])` turns the bins into the digits of one number: with `car_12`, position bin 4 and velocity bin 6 give 4 × 12 + 6 = 54, and with `digits` the bins 3, 0, 7 and 2 give 3072.",
    )
    return


@app.cell(hide_code=True)
def text_after_row():
    mo.md(r"""
    In the picture, your `row_of` for the middle of every cell of MountainCar with 4 bins for each number. The rows count up along the velocity first, 0 to 3 in the bottom row, then 4 to 7 one position bin up: the velocity is the last digit, the position the first. Every combination of bins gets its own row, and no two share one.

    ### The Q-table of MountainCar with NumPy

    `C.course_q("MountainCar-v0", 12)` gives the Q-table of the course's run 1 in MountainCar with 12 bins for each number, after 2,000 episodes of Q-learning with `row_of`. `.shape` gives its rows and columns. `np.abs(...).sum(axis=1) > 0` says for each row whether any of its Q's changed, and `.sum()` counts those rows. `np.argmax(..., axis=1)` gives the best action of every row, and `.reshape(12, 12)` puts the 144 rows in a grid: one line per position bin, from the left hill at the top to the flag at the bottom, and one column per velocity bin, from fast to the left to fast to the right. Run the cell.
    """)
    return


@app.cell
def example_numpy():
    import numpy as np

    _Q = C.course_q("MountainCar-v0", 12)   # run 1 of the course, 12 bins for each number, after 2,000 episodes
    print("shape:", _Q.shape)
    print("rows that changed:", (np.abs(_Q).sum(axis=1) > 0).sum())
    print("the best action of every row, a line per position bin, a column per velocity bin:")
    print(np.argmax(_Q, axis=1).reshape(12, 12))
    return (np,)


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    113 of the 144 rows changed; the agent never got to the other 31, such as a high speed on the far left, and there every Q is still 0, so `np.argmax` takes the first action, 0. In the left half of the columns, where the car moves to the left, most cells say 0, push left; in columns 6 to 9, where it moves to the right, most say 2, push right. That is the swing of the worked example, learned from −1s and a flag, with a few cells that say otherwise. The demo draws the same table as a picture.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your `row_of` in a world of your choice

    The demo trains the course's Q-learner for 2,000 episodes, with η = 0.1, γ = 0.99, ε = 0.1 and every Q at 0, using your `row_of` and the bins you choose: in MountainCar for both numbers, in CartPole for the angle and the angular velocity. Where your function does not pass its check yet, the course's takes its place. The chart shows the mean steps of the last 50 episodes: in CartPole more is better, up to 500; in MountainCar fewer is better, and 200 means that the car did not reach the flag. Beside it, the best action in every bin now. Start with MountainCar and 12 bins, then try 4 and 20.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    world_choice = mo.ui.dropdown(options=C.world_options(L.lang), value="MountainCar", label=C.DEMO_TEXT[L.lang]["world"])
    bins_choice = mo.ui.dropdown(options=C.bins_options(L.lang), value="12", label=C.DEMO_TEXT[L.lang]["bins"])
    run_choice = mo.ui.dropdown(options=C.run_options(L.lang), value=C.DEMO_TEXT[L.lang]["run"] + " 1", label=C.DEMO_TEXT[L.lang]["run"])
    train_button = mo.ui.run_button(label="Train with my row_of")
    mo.hstack([world_choice, bins_choice, run_choice, train_button], justify="start", gap=1.5, wrap=True)
    return bins_choice, run_choice, train_button, world_choice


@app.cell(hide_code=True)
def demo_train(bins_choice, box_row, result_row, run_choice, train_button, world_choice):
    if not train_button.value:
        _out = mo.md("*Pick a world, the bins and a run, then press the button to let the agent learn.*")
    else:
        _mine = result_row.state == "correct"
        _fn = run_code(box_row.value, C.R_NAMES).names["row_of"] if _mine else C.row_of
        _name, _bins, _seed = world_choice.value, bins_choice.value, run_choice.value
        _w = C.DEMO_TEXT[L.lang]
        _now = {"shown": 0}

        def _progress(_n, _run):
            if "run" not in _now:
                _now["chart"].watch(lambda: C.now_picture(_now["run"], _now["n"], L.lang))
            _now["run"], _now["n"] = _run, _n
            if _n % C.STRIDE == 0 or _n == C.EPISODES:
                _now["chart"].extend(C.chart_points(_run, _now["shown"] + 1, _n), f"**Episode {C.number(_n, 0, L.lang)} of {C.number(C.EPISODES, 0, L.lang)}**")
                _now["shown"] = _n

        _live = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"].format(w=C.WINDOW),
                        "alt": _w["alt"].format(w=C.WINDOW, world=C.SHORT[_name])},
                       C.EPISODES, role="ink", low=C.RANGE[_name][0], high=C.RANGE[_name][1], window=1, xticks=C.TICKS)
        _live.line("**Learning** …")
        _now["chart"] = _live
        _r = C.demo_run(_fn, _name, _bins, _seed, show=_progress)
        if _r == "error":
            _out = _live.stopped("*Your `row_of` gave an error during the run, or a row that does not fit the table. Does it still pass its check?*")
        elif _r == "slow":
            _out = _live.stopped("*The run took more than two minutes. Does your function do more work than it needs?*")
        else:
            _who = "your `row_of`" if _mine else "the course's `row_of`, since yours does not pass its check yet"
            _out = mo.vstack([
                L.explore("rewind", "episodes", lambda _v: C.rewind_frame(_r, int(_v), L.lang, _fn), start=C.EPISODES,
                          values=C.looks(), role="",
                          task="Go back to 0, then forward again: when does the picture start to fill, and what does the line do from then on?"),
                mo.md(f"{C.SHORT[_name]}, {_bins} bins, run {_seed + 1}, with {_who}. {C.now_text(_r, _r['Q'], L.lang, _fn)}"),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With the course's function, the bins decide whether MountainCar gets learned at all. With 12 bins for each number, all 10 runs reach the flag, the first time between episode 190 and 429. With 20 bins all 10 do too, but later, between episode 404 and 694: 400 rows instead of 144 leave more to try before the flag. With 8 bins only 3 of the 10 runs ever reach the flag, all after episode 1,900, and with 4 bins only 1. Until its first flag, every Q of a run comes from −1s alone. What pulls the agent on until then is the start: with every Q at 0 and every step −1, an action that was tried looks worse than one that was not, so the agent keeps trying other pushes, as in *One interface for every environment: Gymnasium*. Start every Q at −100 instead, and with 12 bins none of 3 runs reaches the flag in 2,000 episodes.

    CartPole, with 12 bins for the angle and the angular velocity, goes from 23 steps per episode with random pushes to between 128 and 292 at the end of the 10 runs, the mean of the last 100 episodes. With the largest Q alone, without random actions, the pole stays up for 49 to 309 steps on average: far below the 500 steps of one line with the angle and the angular velocity. In the next lesson, a library learns CartPole with a neural network instead of a table.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sparse():
    mo.md(r"""
    ### Exercise 4. Which benchmark?

    A team has written a new way to explore, meant for tasks in which the reward tells nothing until the goal is reached. They want to show on one classic benchmark that it helps.
    """)
    return


@app.cell(hide_code=True)
def choice_sparse_cell():
    choice_sparse = L.choice(
        "sparse",
        question="Which benchmark of this lesson tests just that?",
        options={
            "cart": "CartPole: +1 for every step that the pole stays up",
            "right": "MountainCar: −1 for every step until the flag",
            "atari": "An Atari game, such as Breakout: points for every brick it hits",
            "hopper": "Hopper of MuJoCo: a reward for every bit it moves forward, and for staying upright",
        },
        correct="right",
        why={
            "right": "Every step of MountainCar gives the same −1, whatever the car does, so until the flag the reward says nothing about which push was better: random pushes never get there. CartPole, the Atari games and Hopper give a reward that changes with almost every step, and test other things: balancing, pixels and actions that are real numbers.",
            "cart": "In CartPole every step that the pole stays up gives +1, and a fall ends the episode at once: does that reward stay silent until a goal?",
            "atari": "An Atari game hands out points all through the game, and what it tests is learning from pixels: does it wait with its reward until a goal?",
            "hopper": "Hopper rewards every bit of progress, and what it tests is moving a body with actions that are real numbers: does its reward stay silent until a goal?",
        },
    )
    choice_sparse
    return (choice_sparse,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [One interface for every environment: Gymnasium](/course/open/rl-gymnasium?lang=en) every observation was a whole number, a row of the table. Here the observations are real numbers, and bins turn them into rows. CartPole and MountainCar each test something else: balancing, and a reward that says nothing until the goal. The next lesson, *Using an RL library: Stable-Baselines3*, trains an agent with a neural network on CartPole, with a library of tested algorithms, and judges it over several seeds and many test episodes.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_sparse, result_bin, result_policy, result_row):
    L.wrap_up(result_policy, result_bin, result_row, choice_sparse)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
