# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-gymnasium. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the course's runs in three worlds of Gymnasium, and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-gymnasium", "en")
    C = L.checks   # the course's runs in FrozenLake, Taxi and CliffWalking, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en) and [Careful or bold: SARSA and Q-learning](/course/open/rl-sarsa?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_target():
    quick_target = L.choice(
        "quick-target",
        question="A Q-learning agent takes a step with reward −1 and lands in a state whose largest $Q$ is 4. With $\\gamma$ = 0.5, what is the target of that step?",
        options={
            "reward": "−1, the reward of the step alone",
            "right": "1, the reward plus $\\gamma$ times that largest $Q$",
            "plain": "3, the reward plus the largest $Q$, without $\\gamma$",
            "all": "1.5, $\\gamma$ times the reward and the largest $Q$ added together",
        },
        correct="right",
        why={
            "right": "$-1 + 0.5 \\cdot 4 = 1$: the reward plus the discounted largest $Q$ where the agent landed. In this lesson the target gets one more question: did the episode really end there?",
            "reward": "The reward alone is the target only when nothing follows the step: what does Q-learning add for the state where the agent landed? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
            "plain": "Rewards that come later count for less: what does $\\gamma$ do to the largest $Q$ of the next state? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
            "all": "$\\gamma$ discounts only what comes after the step, not the reward of the step itself: which number does it multiply? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
        },
    )
    quick_target
    return (quick_target,)


@app.cell(hide_code=True)
def quick_check_cliff():
    quick_cliff = L.choice(
        "quick-cliff",
        question="On the cliff of 4 by 12, Q-learning learns with ε-greedy, ε = 0.1. After 500 episodes, which way does its largest $Q$ lead from S to G?",
        options={
            "top": "Up to the top row, along it, and down again: 17 steps, far from the edge",
            "right": "Along the edge of the cliff: 13 steps",
            "middle": "Along the middle row: 15 steps, one row from the edge",
            "none": "Nowhere: it keeps falling off",
        },
        correct="right",
        why={
            "right": "Q-learning learns the values of always taking the best action, so its way runs along the edge, 13 steps; SARSA, on-policy, learns the top row. In this lesson the same cliff comes back as a world of Gymnasium.",
            "top": "That is the way of SARSA, which learns with its own random steps in mind: which policy does Q-learning learn the values of? See [Careful or bold: SARSA and Q-learning](/course/open/rl-sarsa?lang=en).",
            "middle": "The target of Q-learning takes the largest $Q$, never a random step down: how close to the edge does that let its way run? See [Careful or bold: SARSA and Q-learning](/course/open/rl-sarsa?lang=en).",
            "none": "Its random steps make it fall now and then while it learns, but its largest $Q$ does lead to G: along which row? See [Careful or bold: SARSA and Q-learning](/course/open/rl-sarsa?lang=en).",
        },
    )
    quick_cliff
    return (quick_cliff,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In every lesson about RL so far, the world came with its own code. GRID of Q-learning had `step(state, action, rng)`, the cliff of SARSA had `step(state, action)`, and each had its own states, its own names for actions and its own way to say that an episode was over. An agent written for one world did not run in another. In 2016, OpenAI released Gym: a library with many environments, from a frozen lake to Atari games, behind one interface. Since 2022 the Farama Foundation maintains it as Gymnasium. Researchers test a new method on the same environments, in the same versions, so that their results can be compared, and libraries of RL, such as Stable-Baselines3 in a later lesson, train on any environment that follows the interface. In this lesson you write one Q-learning agent and let it learn three worlds without changing a line.

    **An everyday example.** Every car has the accelerator on the right, the brake to its left and the steering wheel in front of you. Someone who can drive one car can drive almost any car, whatever engine is under the hood. Gymnasium gives every environment the same pedals.
    """)
    return


@app.cell(hide_code=True)
def predict_fail():
    guess_fail = L.guess(
        "fail",
        question="The course trains the same Q-learning agent in the three worlds of the picture: FrozenLake for 10,000 episodes, Taxi for 5,000 and CliffWalking for 500, with η = 0.1, γ = 0.99 and ε = 0.1, and every Q at 0 at the start. Each world gets 10 runs, each with random numbers of its own. In which world do some of the runs fail to learn the task, do you think?",
        options={
            "taxi": "Taxi: 500 states and 6 actions are too much to learn in 5,000 episodes",
            "cliff": "CliffWalking: in 500 episodes the agent keeps falling off the cliff",
            "right": "FrozenLake, the smallest",
            "none": "None: in all three worlds, all 10 runs learn the task",
        },
        reveal="**FrozenLake, the smallest.** In 3 of its 10 runs the largest Q never leads to G, and the agent reaches G in 0% of the episodes; the other 7 reach G in 73.0% to 74.0% of them, close to 74.4%, the best any policy can do within 100 steps. In Taxi all 10 runs deliver every passenger, and on CliffWalking all 10 take the shortest way, 13 steps. Why the smallest world is the hardest one comes after the demo.",
    )
    mo.vstack([mo.Html(C.world_picture(L.lang)), guess_fail])
    return (guess_fail,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **One interface.** Gymnasium is a library for Python, imported as `gym`. `env = gym.make("FrozenLake-v1")` makes an environment by its name; `-v1` is its version, which goes up whenever the world changes, so that a result always says which world it came from. Every environment has the same two methods. `obs, info = env.reset(seed=0)` starts an episode and gives the first observation, `obs`: what the agent gets to see of the state. `obs, reward, terminated, truncated, info = env.step(action)` takes one action and gives back five things: the observation after the step, the reward, two flags that say whether the episode is over, and `info`, a dict with extra information that the agent does not need to learn. `seed=0` fixes the random numbers of the world, so that the same seed gives the same episodes; `env.reset()` without a seed starts the next episode, and the random numbers go on from where they were.

        **Spaces.** `env.observation_space` and `env.action_space` say what observations and actions look like: the **observation space** and the **action space**. In the three worlds of this lesson both are `Discrete(n)`: the whole numbers 0 to n − 1. `env.action_space.n` is that n, and `env.action_space.sample()` gives a random action. Other worlds have other spaces: in CartPole of the next lesson an observation is four real numbers, a `Box`. An agent with a Q-table needs a `Discrete` observation space, for a row per observation, and a `Discrete` action space, for a column per action:

        | World | `observation_space` | `action_space` | the actions | time limit |
        |---|---|---|---|---|
        | FrozenLake-v1 | `Discrete(16)` | `Discrete(4)` | 0 left, 1 down, 2 right, 3 up | 100 steps |
        | Taxi-v4 | `Discrete(500)` | `Discrete(6)` | 0 south, 1 north, 2 east, 3 west, 4 pick up, 5 drop off | 200 steps |
        | CliffWalking-v1 | `Discrete(48)` | `Discrete(4)` | 0 up, 1 right, 2 down, 3 left | none |

        On FrozenLake and CliffWalking the observation is the cell, counted row by row from the top left: on FrozenLake row × 4 + column. In Taxi it holds four things at once: the row and the column of the taxi, where the passenger is, at one of the four stops or in the taxi, and the destination: 5 × 5 × 5 × 4 = 500 observations.

        **Ended or cut off.** An episode can stop for two reasons. `terminated` is `True` when the task itself ends: the agent stands on G or in a hole, or the passenger has been delivered. Nothing comes after that. `truncated` is `True` when a time limit cuts the episode off: after 100 steps on FrozenLake, after 200 in Taxi. The world has not ended there: the taxi is still somewhere in the streets, and the passenger is still waiting. Stopping an episode early like this is **truncation**. For learning, the difference lies in the target. After `terminated`, the target of the last step is just its reward. After `truncated`, the state still has a future, so the target stays the reward plus γ times the largest Q of the next state, as after any other step. Older versions of Gym had one flag, `done`, for both reasons; Gymnasium splits it in two.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **Gymnasium** | | a library for Python with many environments behind one interface: `reset`, `step` and spaces | `import gymnasium as gym` |
        | **observation space** | | what an observation can be, such as `Discrete(16)`: a whole number from 0 to 15 | `env.observation_space` |
        | **action space** | | what an action can be, such as `Discrete(4)` | `env.action_space` |
        | **truncation** | | stopping an episode early, by a time limit, while the task has not ended | `truncated` |
        | `terminated` | | the task ended, and nothing follows | |
        | `Discrete(n)` | discrete n | the whole numbers 0 to n − 1 | `.n` |
        """),
        L.watch_out("The same number is another action in another world: 0 is left on FrozenLake and up on CliffWalking. An agent that only uses what the spaces say, as yours will, runs in every world; an agent with \"0 is left\" in its code runs in one."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Q-learning in any world of Gymnasium with `Discrete` spaces:

    1. Make the world with `gym.make`, and a Q-table of zeros with `env.observation_space.n` rows and `env.action_space.n` columns.
    2. Seed the world once, with `env.reset(seed=...)`, so that a run can be repeated.
    3. Start an episode with `env.reset()`: it gives the first observation.
    4. Choose an action, ε-greedy: with probability ε a random one of the `env.action_space.n` actions, else the one with the largest $Q$ in the row of the observation.
    5. Take it with `env.step(action)`: you get the next observation, the reward, `terminated`, `truncated` and `info`.
    6. Work out the target: the reward alone if `terminated`; otherwise, also if `truncated`, the reward plus γ times the largest $Q$ of the next observation. Move $Q$ a step η toward it.
    7. Go on from the next observation at step 4, until `terminated` or `truncated` is `True`. Then start the next episode at step 3.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    FrozenLake, with `env.reset(seed=6)`: the agent starts in 0, top left, and takes the actions right, right, down and down. The cell below makes the world, prints its spaces and takes the four steps; `info` holds `prob`, the probability of what happened. `break` stops the loop as soon as the episode is over. Run the cell.
    """)
    return


@app.cell
def example_frozen():
    import gymnasium as gym

    _env = gym.make("FrozenLake-v1")
    print("observation_space:", _env.observation_space, "action_space:", _env.action_space)
    _obs, _info = _env.reset(seed=6)
    print("start:", _obs)
    for _action in [2, 2, 1, 1]:   # right, right, down, down
        _nxt, _reward, _terminated, _truncated, _info = _env.step(_action)
        print("from", _obs, "action", _action, "to", _nxt, "reward", _reward,
              "terminated", _terminated, "truncated", _truncated, "prob", round(_info["prob"], 3))
        _obs = _nxt
        if _terminated or _truncated:
            break
    return (gym,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first three steps go where the agent asked: 0 to 1, 1 to 2, and down from 2 to 6, each with probability 1/3, `prob` 0.333. The fourth, down from 6, slips to the right, into the hole at 7: `terminated` is `True`, the reward is 0, and the episode is over. Nothing comes after a hole, so the target of this step is just its reward, 0.

    Now Taxi, in run 4 of the course after 5,000 episodes. In state 249 the taxi stands in the middle of the streets, the passenger waits at Y, and the destination is G. The six Q's of 249 are −4.85, −6.56, −5.21, 5.30, −7.82 and −6.47; the largest, 5.30, belongs to west. Suppose a step lands in 249 with reward −1, and it is the 200th step of the episode: `truncated` is `True` and `terminated` is `False`. The target is −1 + 0.99 × 5.30 = 4.25, the same as for any other step to 249: from there, the largest Q delivers the passenger in 14 steps. A target of −1 alone would tell the agent that the middle of the streets leads nowhere, only because the clock ran out there.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_play():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A random agent in any world

        `play_random(env, seed)` plays one episode with random actions in a world of Gymnasium, and gives back `total`, the return, `steps`, the number of steps, and `terminated`, whether the task ended. `env.reset(seed=seed)` starts the episode, and `env.action_space.seed(seed)` fixes the random numbers of `env.action_space.sample()`, so that the same seed gives the same episode. `total, steps = 0, 0` sets both to 0, and `terminated, truncated = False, False` sets both flags to `False`. `while not (...):` repeats its block as long as the test between the parentheses is not true. `total += reward` adds `reward` to `total`, and `steps += 1` adds 1 to `steps`. Replace the two `...`: the test, so that the loop stops as soon as the episode has ended or has been cut off, and the call that takes the `action` and gives its five values to `obs`, `reward`, `terminated`, `truncated` and `info`.

        The check calls `play_random(gym.make("FrozenLake-v1"), 0)`. Then four other episodes: Taxi with seeds 0 and 27, CliffWalking with seed 0, and FrozenLake with seed 115.
        """),
        L.task("Make `play_random` stop when the episode ends or is cut off, and take each step with `env.step`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_play():
    box_play = L.code_box("play_random", lines=11)
    box_play
    return (box_play,)


@app.cell(hide_code=True)
def check_play(box_play):
    result_play = L.check("play_random", box_play.value)
    result_play
    return (result_play,)


@app.cell(hide_code=True)
def hints_play():
    L.hints(
        "play_random",
        [
            "Steps 5 and 7 of the recipe. The loop goes on as long as neither `terminated` nor `truncated` is `True`. The other `...` is the call that takes the `action` and gives back `obs, reward, terminated, truncated, info`, in that order.",
            "`while not (a or b):` stops as soon as `a` or `b` is `True`. In the worked example, the step from 6 with action 1 gave back 7, 0, `True`, `False` and `info`: which method of `env` took that step?",
            "`        obs, reward, terminated, truncated, info = ___`",
        ],
        why="`while not (terminated or truncated):` stops at an end and at a cut alike, and `obs, reward, terminated, truncated, info = env.step(action)` takes one step. The same lines run in every world of Gymnasium.",
    )
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    In the picture, your `play_random` plays seeds 0, 1 and 2 in each world. FrozenLake ends within a few steps, in a hole; a random agent reaches G there in 1 of 72 episodes. In Taxi the clock cuts every one of these episodes off at 200 steps, with a return far below 0; by chance, a random agent delivers the passenger in about 1 of 21 episodes. CliffWalking has no time limit, so a random walk goes on until it reaches G: 3,791 steps with seed 0, and 6,453 on average.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_target():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The target, ended or cut off

        `target(Q, reward, nxt, terminated, truncated, gamma)` gives back the target of Q-learning for one step: step 6 of the recipe. `Q` is a Q-table as a NumPy array, with a row per observation and a column per action. `Q[nxt]` is the row of `nxt`, the observation after the step, and the method `.max()` of an array gives its largest number. `reward`, `terminated` and `truncated` come from `env.step`, and `gamma` is γ. Replace the two `...`: the test for a step after which nothing follows, and the largest Q of `nxt`.

        The check calls `target(Q_taxi, -1, 249, False, False, 0.99)`, with `Q_taxi` the Q-table of the course's run 4 of Taxi: a step like any other, to 249, the middle of the worked example. Then four other steps: the same step as the 200th of its episode, with `truncated` `True`; two steps of FrozenLake from `Q_ones`, a table in which every Q is 1: into the hole at 7, and onto G at the 100th step, where both flags are `True`; and a step with γ = 0.9 to 257, the middle with the passenger in the taxi.
        """),
        L.task("Make `target` give the reward alone only when the task ended, and otherwise the reward plus γ times the largest Q of the next observation."),
    ])
    return


@app.cell(hide_code=True)
def exercise_target():
    box_target = L.code_box("target", lines=4)
    box_target
    return (box_target,)


@app.cell(hide_code=True)
def check_target(box_target):
    result_target = L.check("target", box_target.value)
    result_target
    return (result_target,)


@app.cell(hide_code=True)
def hints_target():
    L.hints(
        "target",
        [
            "Step 6 of the recipe: after an end, the target is the `reward` alone; after any other step, a cut included, it is the `reward` plus `gamma` times the largest Q of the next observation. Which of the two flags says that the task itself ended?",
            "In the worked example, the largest number of row 249 is 5.30, and the target of the 200th step is −1 + 0.99 × 5.30 = 4.25, although `truncated` is `True`. Which row of `Q` belongs to `nxt`, and which method gives its largest number?",
            "`    return reward + gamma * ___`",
        ],
        why="`if terminated:` gives the reward alone only when the task ended; after a cut, `return reward + gamma * Q[nxt].max()` keeps the future of `nxt`, since the world goes on there.",
    )
    return


@app.cell(hide_code=True)
def text_after_target():
    mo.md(r"""
    In the picture, your `target` for four of the steps. The 200th step of Taxi keeps its future, 4.25, like any other step to the middle. With `Q_ones` the hole at 7 has a Q of 1, so a target that does not ask `terminated` would give 0.99 for falling in, as if something good came after the hole. In a table that starts at 0 that mistake hides: the Q's of an end stay 0, since no step starts there.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_episode():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. One agent for every world

        `q_episode(env, Q, rng)` plays one episode of Q-learning in a world of Gymnasium, updates `Q` after every step and gives back `total`, the return. It knows nothing about the world but its spaces. `Q[obs, action]` is the Q of `obs` and `action`, and `np.argmax(Q[obs])` the action with the largest Q of `obs`. `rng` gives the random numbers, as in Q-learning: `rng.random()` a number from 0 to 1, and `rng.integers(n)` a whole number from 0 to n − 1, the same as `rng.integers(0, n)`. `EPSILON`, `ETA` and `GAMMA` are ε, η and γ: 0.1, 0.1 and 0.99. `target` is the course's own solution of exercise 2. `env.reset()` starts the episode, and `env.step(action)` gives `nxt`, the observation after the step, `reward`, `terminated`, `truncated` and `info`, as in exercise 1; `obs = nxt` moves the agent on. Replace the two `...`: how many actions the random action is chosen from, in whatever world, and the target of the step.

        The check calls `q_episode(gym.make("FrozenLake-v1"), Q_frozen_2000, np.random.default_rng(1))`, with `Q_frozen_2000` the Q-table of the course's run 4 of FrozenLake after 2,000 episodes, the world seeded with `env.reset(seed=1)`, and a new random generator with seed 1. It compares the return and the Q-table that your function leaves with those of the course. Then four other cases: Taxi from `Q_zeros`, a table of zeros, and from `Q_taxi_250`, after 250 episodes; CliffWalking from `Q_zeros` and from `Q_cliff_50`.
        """),
        L.task("Make `q_episode` choose a random action from all the actions of the world, and move Q toward the target of exercise 2."),
    ])
    return


@app.cell(hide_code=True)
def exercise_episode():
    box_episode = L.code_box("q_episode", lines=14)
    box_episode
    return (box_episode,)


@app.cell(hide_code=True)
def check_episode(box_episode):
    result_episode = L.check("q_episode", box_episode.value)
    result_episode
    return (result_episode,)


@app.cell(hide_code=True)
def hints_episode():
    L.hints(
        "q_episode",
        [
            "Steps 4 and 6 of the recipe. A random `action` comes from `rng.integers`, with as many choices as the world has actions: the action space knows how many. The target is what `target` gives for this step.",
            "`Discrete(6)` of Taxi has 6 as its `.n`, and `env.action_space` is the action space of the world. The second `...` calls `target` with the values of the step in the order of its `def` line: `Q`, `reward`, `nxt`, `terminated`, `truncated`, and `GAMMA` as γ.",
            "`            action = rng.integers(___)`",
        ],
        why="`rng.integers(env.action_space.n)` chooses from all the actions of a world, 6 in Taxi and 4 in the other two, and `target(Q, reward, nxt, terminated, truncated, GAMMA)` is the target of exercise 2. The function needs nothing else from the world: that is what one interface buys.",
    )
    return


@app.cell(hide_code=True)
def text_after_episode():
    mo.md(r"""
    In the picture, your `q_episode` learns CliffWalking for 500 episodes and Taxi for 2,000, each from a table of zeros. On the cliff the largest Q leads along the edge, as Q-learning did in *Careful or bold*; in Taxi it takes the passenger from Y to G. The same function, without a change, learns FrozenLake too: that world comes in the demo.

    ### The Q-tables with NumPy

    A Q-table is an array, so NumPy can look at all of it at once. `C.course_q(name)` gives the Q-table of the course's run 4 at the end, for the world `name`. `.shape` gives its rows and columns. `np.abs` makes every number positive, `.sum(axis=1)` adds up each row, and `> 0` says for each row whether any of its Q's changed; `.sum()` then counts the rows that did. `np.argmax(..., axis=1)` gives the best action of every row, `np.max(..., axis=1)` its largest Q, `np.round(..., 2)` rounds to 2 decimals, and `.reshape(4, 4)` puts the 16 numbers of FrozenLake in the 4 rows of the lake. Run the cell.
    """)
    return


@app.cell
def example_numpy():
    import numpy as np

    _frozen = C.course_q("FrozenLake-v1")    # run 4 of the course, after 10,000 episodes
    _taxi = C.course_q("Taxi-v4")            # after 5,000 episodes
    _cliff = C.course_q("CliffWalking-v1")   # after 500 episodes
    print("shapes:", _frozen.shape, _taxi.shape, _cliff.shape)
    _frozen_rows = (np.abs(_frozen).sum(axis=1) > 0).sum()   # the rows with a Q that is not 0
    _taxi_rows = (np.abs(_taxi).sum(axis=1) > 0).sum()
    _cliff_rows = (np.abs(_cliff).sum(axis=1) > 0).sum()
    print("rows that changed:", _frozen_rows, _taxi_rows, _cliff_rows)
    print("FrozenLake, the best action of every cell:")
    print(np.argmax(_frozen, axis=1).reshape(4, 4))
    print("FrozenLake, the largest Q of every cell:")
    print(np.round(np.max(_frozen, axis=1).reshape(4, 4), 2))
    return (np,)


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    On FrozenLake 11 of the 16 rows changed: the 4 holes and G stay 0, since no step starts there. In Taxi 400 of the 500 rows changed: in the other 100 the passenger is already at the destination, where an episode ends. On CliffWalking 37 of the 48 rows changed: the 10 cells of the cliff and G stay 0.

    The best actions of FrozenLake look odd at first: in S, the cell top left, it is 0, left, into the edge of the lake, and in the three cells to its right it is 3, up, into the edge again. On slippery ice that is clever. Left from S slips up or down, or stays put, and never goes into a hole; up from the top row slips left or right, but never down, toward the holes at 5 and 7. The holes and G show 0: every Q there is 0, and `np.argmax` takes the first. The largest Q of S is 0.57, below the chance of 74.0% to reach G, since γ = 0.99 discounts a G that comes after many steps.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your `q_episode` in a world of your choice

    The demo trains your `q_episode` in the world and the run you choose, from a table of zeros, as the course did: run 1 with seed 0, run 2 with seed 1, and so on. Where your function does not pass its check yet, the course's takes its place. The chart shows the mean return of the last episodes, 100 on FrozenLake, 50 in Taxi and 10 on CliffWalking; on FrozenLake that is the share of episodes that reach G. Beside it, what the largest Q does in the world now. Start with FrozenLake, run 1, and then run 4.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    world_choice = mo.ui.dropdown(options=C.world_options(L.lang), value="FrozenLake", label=C.DEMO_TEXT[L.lang]["world"])
    run_choice = mo.ui.dropdown(options=C.run_options(L.lang), value=C.DEMO_TEXT[L.lang]["run"] + " 1", label=C.DEMO_TEXT[L.lang]["run"])
    train_button = mo.ui.run_button(label="Train my q_episode")
    mo.hstack([world_choice, run_choice, train_button], justify="start", gap=1.5, wrap=True)
    return run_choice, train_button, world_choice


@app.cell(hide_code=True)
def demo_train(box_episode, result_episode, run_choice, train_button, world_choice):
    if not train_button.value:
        _out = mo.md("*Pick a world and a run, then press the button to let your agent learn.*")
    else:
        _mine = result_episode.state == "correct"
        _fn = run_code(box_episode.value, C.E_NAMES).names["q_episode"] if _mine else C.q_episode
        _name, _seed = world_choice.value, run_choice.value
        _w = C.DEMO_TEXT[L.lang]
        _now = {"shown": 0}

        def _progress(_n, _run):
            if "run" not in _now:
                _now["chart"].watch(lambda: C.now_picture(_now["run"], _now["n"], L.lang))
            _now["run"], _now["n"] = _run, _n
            if _n % C.STRIDE[_name] == 0 or _n == C.EPISODES[_name]:
                _now["chart"].extend(C.chart_points(_run, _now["shown"] + 1, _n), f"**Episode {C.number(_n, 0, L.lang)} of {C.number(C.EPISODES[_name], 0, L.lang)}**")
                _now["shown"] = _n

        _live = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"].format(w=C.WINDOW[_name]),
                        "alt": _w["alt"].format(w=C.WINDOW[_name], world=C.SHORT[_name])},
                       C.EPISODES[_name], role="ink", low=C.RANGE[_name][0], high=C.RANGE[_name][1], window=1,
                       xticks=C.TICKS[_name])
        _live.line("**Learning** …")
        _now["chart"] = _live
        _r = C.demo_run(_fn, _name, _seed, show=_progress)
        if _r == "error":
            _out = _live.stopped("*Your `q_episode` gave an error during the run, or gave something that is not a number. Does it still pass its check?*")
        elif _r == "slow":
            _out = _live.stopped("*The run took more than two minutes. Does your function do more work than it needs?*")
        else:
            _who = "your `q_episode`" if _mine else "the course's `q_episode`, since yours does not pass its check yet"
            _out = mo.vstack([
                L.explore("rewind", "episodes", lambda _v: C.rewind_frame(_r, int(_v), L.lang), start=C.EPISODES[_name],
                          values=C.looks(_name), role="",
                          task="Go back to 0, then forward again: from when on does the largest Q lead to the goal, and what does the line do from then on?"),
                mo.md(f"{C.SHORT[_name]}, run {_seed + 1}, with {_who}. {C.now_text(_name, _r['Q'], L.lang)}"),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With the course's function, 3 of the 10 runs of FrozenLake never learn: runs 1, 2 and 10. Their largest Q never leads to G; run 1 reaches G once, in episode 9,313, and one reward is not enough. The other seven reach G in 73.0% to 74.0% of the episodes. The cause is the start. With every Q at 0, `np.argmax` takes the first action, left, so the agent mostly walks into the left edge of the lake: with ε = 0.1 such an agent reaches G in about 1 of 10,740 episodes, while a random agent does so in 1 of 72. Until the agent first reaches G, no reward changes a Q, and nothing pulls it away from left. Start every Q at 1 instead of 0, an optimistic start as in *Averaging as you go: drifting machines and optimistic starts*, and every action looks good until it has been tried: then all 10 runs reach G in 74.0% of the episodes.

    In Taxi and on CliffWalking every step costs −1, so an action that was tried soon looks worse than one at 0 that was not: there the zeros work as an optimistic start of their own. All 10 runs of Taxi deliver all 300 passengers, with a mean return of 7.93, the best possible, and all 10 runs of CliffWalking take the 13 steps along the edge.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cut():
    mo.md(r"""
    ### Exercise 4. A robot arm against the clock

    A team trains a robot arm in a simulator to put a cup on a table. Each step gives −1, putting the cup down gives +10 and ends the episode, and a time limit cuts every episode off after 50 steps. In one episode, at the 50th step, the arm holds the cup just above the table when the clock runs out: `terminated` is `False` and `truncated` is `True`.
    """)
    return


@app.cell(hide_code=True)
def choice_cut_cell():
    choice_cut = L.choice(
        "cut_off",
        question="Which target should the Q of this last step move toward?",
        options={
            "right": "−1 plus γ times the largest Q where the arm is now",
            "reward": "−1 alone, since the episode is over",
            "zero": "0, since a step that the clock cuts off should not count",
            "ten": "+10, the reward of putting the cup down, since the arm was about to do that anyway",
        },
        correct="right",
        why={
            "right": "The clock is not part of the task: the world goes on from where the arm holds the cup, one step from +10. So the target keeps γ times the largest Q there, as after any other step. With −1 alone, the arm would learn that holding the cup above the table is worth little, only because the clock often runs out there.",
            "reward": "The episode is over for the agent, but is the task? What would have followed if the clock had not run out?",
            "zero": "The step happened, and its reward counts like any other. And the state the arm is in after the step?",
            "ten": "A target cannot count a reward that did not come: which number of the Q-table says what the state of the arm is worth?",
        },
    )
    choice_cut
    return (choice_cut,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en) and [Careful or bold: SARSA and Q-learning](/course/open/rl-sarsa?lang=en), every world came with its own code. With Gymnasium, the same `q_episode` learns FrozenLake, Taxi and CliffWalking, and every method of this domain runs in every world that follows the interface. The next lesson, *CartPole, MountainCar and the benchmarks of RL*, goes to worlds in which an observation is a list of real numbers instead of a cell, so that a Q-table first needs those numbers cut into bins.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_cut, result_episode, result_play, result_target):
    L.wrap_up(result_play, result_target, result_episode, choice_cut)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
