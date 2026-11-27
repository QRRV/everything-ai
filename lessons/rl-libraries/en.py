# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-libraries. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the course's runs of PPO, the stand-in models and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import torch
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-libraries", "en")
    C = L.checks   # the course's runs of PPO in CartPole, the stand-in models of the check, and the pictures
    torch.set_num_threads(1)   # PPO's small networks are as fast on one thread, and a run then does not depend on the number of cores


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [CartPole, MountainCar and the benchmarks of RL](/course/open/rl-classic-environments?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rows():
    quick_rows = L.choice(
        "quick-rows",
        question="The course cut the angle and the angular velocity of CartPole into 12 bins each, and gave the position and the velocity of the cart 1 bin each. How many rows did its Q-table get?",
        options={
            "plus": "26, one for each bin of each number",
            "right": "144, one for each pair of bins of the pole",
            "four": "48, 12 bins for each of the four numbers",
            "all": "20,736: 12 × 12 × 12 × 12, since every number counts",
        },
        correct="right",
        why={
            "right": "12 × 12 × 1 × 1 = 144 rows of 2 Q's each, with bins chosen by hand. The network of this lesson takes the four numbers as they are, without bins.",
            "plus": "A row stands for one combination of bins, one bin of every number at once: how many combinations do 12, 12, 1 and 1 bins make? See [CartPole, MountainCar and the benchmarks of RL](/course/open/rl-classic-environments?lang=en).",
            "four": "Four times 12 counts bins, not combinations of them, and the cart's numbers did not get 12 bins: how many did they get? See [CartPole, MountainCar and the benchmarks of RL](/course/open/rl-classic-environments?lang=en).",
            "all": "The position and the velocity of the cart got 1 bin each, so the table ignores them: how many rows does that leave? See [CartPole, MountainCar and the benchmarks of RL](/course/open/rl-classic-environments?lang=en).",
        },
    )
    quick_rows
    return (quick_rows,)


@app.cell(hide_code=True)
def quick_check_greedy():
    quick_greedy = L.choice(
        "quick-greedy",
        question="The course tested its Q-learner with the largest Q alone, without random actions. Why leave out the random actions in a test?",
        options={
            "right": "To see what the agent learned, without the exploring of training",
            "speed": "Random actions make a test slower to compute",
            "rule": "Gymnasium allows random actions only during training",
            "more": "So that the agent can still learn from its mistakes during the test",
        },
        correct="right",
        why={
            "right": "Random actions are for trying things out while learning; a test asks what the agent does with what it knows. With the largest Q alone, the pole stayed up for 49 to 309 steps on average. PPO of this lesson explores in its own way, and a test switches that off too.",
            "speed": "A random action is as quick to pick as the largest Q: is time what the test is about? See [CartPole, MountainCar and the benchmarks of RL](/course/open/rl-classic-environments?lang=en).",
            "rule": "Gymnasium takes whatever action the agent gives it, random or not: who decides which actions a test takes? See [CartPole, MountainCar and the benchmarks of RL](/course/open/rl-classic-environments?lang=en).",
            "more": "A test changes nothing in the Q-table: what is it meant to measure, then? See [CartPole, MountainCar and the benchmarks of RL](/course/open/rl-classic-environments?lang=en).",
        },
    )
    quick_greedy
    return (quick_greedy,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The course's own Q-learner with 12 bins kept the pole of CartPole up for 49 to 309 steps in its tests, far from the 500 of the time limit, and it needed bins chosen by hand. People who use RL rarely write the algorithm themselves. A small mistake in the code of an RL algorithm seldom stops it with an error: the agent just learns worse, and nothing says why. When Henderson and colleagues (2018) ran the same algorithm from different code, with the same settings, the results differed widely. An **RL library** holds versions of the algorithms that have been tested against known results on benchmark environments, with one interface for all of them. In this lesson, **Stable-Baselines3** trains an agent on CartPole in a few lines. A library does not make a single run trustworthy, though: another seed gives another run. So you also learn to judge an agent the way research in RL does, over several seeds and many test episodes.

    **An everyday example.** You buy a bathroom scale instead of building one: the factory has tested it, so you trust its numbers more than those of a scale you built yourself. Still, one weighing right after dinner says little about what you really weigh. You weigh yourself on several mornings and look at all of them.
    """)
    return


@app.cell(hide_code=True)
def predict_seeds():
    guess_seeds = L.guess(
        "seeds",
        question="PPO of Stable-Baselines3 trains on CartPole for 25,000 steps, with the library's default settings: about 20 seconds on the course's computer. Five seeds each train an agent, and each agent is tested on 100 episodes. In how many of those 500 test episodes does the pole stay up for all 500 steps, do you think?",
        options={
            "few": "About 1 in 10",
            "half": "About half of them",
            "right": "Every single one",
            "none": "None: 25,000 steps are far too few for a neural network",
        },
        reveal="**Every single one.** After 25,000 steps, between 288 and 341 episodes of training, each of the 5 seeds keeps the pole up for all 500 steps in every one of its 100 test episodes. The course's Q-table with 12 bins got to 49 to 309 steps on average, after 2,000 episodes. With 10,000 steps, the seeds still differ a lot: the mean of their 100 test episodes lies between 293 and 416, depending on the seed. And the demo shows that even after 25,000 steps not every seed is perfect.",
    )
    guess_seeds
    return (guess_seeds,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A library of algorithms.** Stable-Baselines3, SB3 for short (Raffin and colleagues, 2021), is a Python library of tested RL algorithms, built on PyTorch: PPO, DQN, A2C, SAC, TD3 and DDPG. Every algorithm is a class with the same few methods. `PPO("MlpPolicy", env, seed=0, device="cpu")` makes a model: with the policy `"MlpPolicy"`, the environment `env`, a seed for all its random numbers, and the processor, `"cpu"`, since its small networks run faster there than on a graphics card. `model.learn(total_timesteps=25000)` trains it for 25,000 steps of the environment, over as many episodes as those steps make. `model.predict(obs)` gives the action for an observation. `model.save("ppo_cartpole")` writes the trained model to a file, and `PPO.load("ppo_cartpole")` reads it back. DQN and A2C go the same way: only the name of the class changes.

        **PPO.** Proximal policy optimization (Schulman and colleagues, 2017) learns the policy itself, without a Q-table. The policy is a neural network: with `"MlpPolicy"` an MLP with two hidden layers of 64 neurons, which gets the four numbers of CartPole as they are, without bins, and gives a chance for each action. Beside it PPO trains a second network of the same shape as its value function: like V in earlier lessons, it estimates the return from a state. PPO plays 2,048 steps with the policy as it is, then judges every action: did things go better than V expected, or worse? Then it nudges the policy, so that the actions that turned out better get more probable and the others less, in 10 passes over those 2,048 steps, but never far from the policy that played them. That last part is the *proximal*, and it keeps the training steady. Then it plays the next 2,048 steps. A later level of the course builds PPO itself; here you use it.
        """),
        mo.Html(C.ppo_picture(L.lang)),
        mo.md(r"""
        **Exploring and testing.** During training, PPO draws each action from the chances of its network: an action with a chance of 0.8 is taken 8 times in 10. That is how it explores, instead of ε-greedy, and as it learns, the chances move toward 0 and 1, so it explores less. A test should show what the agent has learned, so it takes the most probable action every time. `predict` does that when you give its argument `deterministic` the value `True`, by name; without it, `predict` draws as in training. `predict` gives back two things: the action, and a state that only networks with a memory use, so `action, _ = model.predict(...)` keeps the action and drops the state.

        **Judging an agent.** One run of RL is one sample. The seed decides the first numbers of the networks, the draws and the starts of the episodes, and two seeds can learn at quite different speeds. Henderson and colleagues (2018) trained the same algorithm with the same settings on two groups of 5 seeds, and the two groups looked like two different algorithms. So judge an agent this way: train it with several seeds, 5 or more; test every trained agent on many episodes, such as 100, with seeds that training did not use; and report every seed, or the median and the worst seed, never the best seed alone. Agarwal and colleagues (2021) show more careful ways to summarize results over a few seeds.

        **Three libraries.** Stable-Baselines3 is one of several libraries, each made for something else:

        | Library | Made for | How |
        |---|---|---|
        | Stable-Baselines3 | using a tested algorithm in a few lines | one interface for every algorithm, with documentation and tests against known results |
        | CleanRL | reading an algorithm and changing it | each algorithm in one file of a few hundred lines, from making the environment to the last update |
        | RLlib | many environments and learners at once | part of the package Ray, which spreads the work over many processor cores or computers |

        The course installs only Stable-Baselines3. CleanRL (Huang and colleagues, 2022) is meant to be read and copied, one file at a time, and RLlib (Liang and colleagues, 2018) comes with Ray.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **Stable-Baselines3** | stable baselines three | a Python library of tested RL algorithms with one interface, SB3 for short | `from stable_baselines3 import PPO` |
        | **PPO** | p-p-o | proximal policy optimization: learns a policy as a neural network, a little at a time | `PPO("MlpPolicy", env)` |
        | **CleanRL** | clean r-l | RL algorithms, each in one file to read and change | |
        | **RLlib** | r-l-lib | a library that spreads environments and learners over many processor cores or computers | |
        | `total_timesteps` | total time steps | how many steps of the environment `learn` trains for, over all episodes | `model.learn(total_timesteps=25000)` |
        | `deterministic` | | an argument of `predict`: with `True`, the most probable action instead of a draw | `model.predict(obs, ...)` |
        """),
        L.watch_out("One run says little. A result of RL holds for the seeds and the test episodes it was measured on: the more of each, the more it says."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Training an agent with an RL library, and judging it:

    1. Make the environment, and the model with a seed of its own: `PPO("MlpPolicy", env, seed=..., device="cpu")`.
    2. Let the model learn: `model.learn(total_timesteps=...)`.
    3. Test it in an environment of its own, on many episodes with seeds that training did not use, with the most probable action at every step.
    4. Do steps 1 to 3 for several seeds, 5 or more.
    5. Report every seed, or the median and the worst seed of their test means; never the best seed alone.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell imports `PPO`, makes CartPole, and trains a model with seed 0 for 10,000 steps: about 8 seconds on the course's computer. `model.num_timesteps` is how many steps it really took. Then it plays one test episode in an environment of its own, `_test`, seeded with 1000, with the most probable action at every step; every 100 steps, and at the end, it prints the position of the cart and the angle of the pole. Run the cell.
    """)
    return


@app.cell
def example_ppo():
    import gymnasium as gym
    from stable_baselines3 import PPO

    _env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", _env, seed=0, device="cpu")
    model.learn(total_timesteps=10000)
    print("trained for", model.num_timesteps, "steps")

    _test = gym.make("CartPole-v1")
    _obs, _info = _test.reset(seed=1000)
    _steps = 0
    _terminated, _truncated = False, False
    while not (_terminated or _truncated):
        _action, _ = model.predict(_obs, deterministic=True)   # the most probable action
        _obs, _reward, _terminated, _truncated, _info = _test.step(_action)
        _steps += 1
        if _steps % 100 == 0 or _terminated:
            print("step", _steps, "position", round(float(_obs[0]), 2), "angle", round(float(_obs[2]), 3))
    print("terminated", _terminated, "truncated", _truncated, "after", _steps, "steps")
    return PPO, gym, model


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `learn` stops at 10,240 steps, not at 10,000: PPO learns from whole batches of 2,048 steps, and 5 batches are the first to reach 10,000. In the test episode the pole stays up for all 500 steps, until the time limit cuts the episode off: `truncated` is `True`. The cart drifts slowly to the right, to 0.22, and at every print the angle is within 0.003 radians of 0. One episode says little, though: over 100 test episodes, seeds 1000 to 1099, this agent keeps the pole up for 416 steps on average, and for all 500 in 59 of them. On your computer, the run can come out a little differently: the tiny differences in how processors round numbers grow over the thousands of updates of a run.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_play():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A test episode of your own

        `play_test(model, seed)` plays one test episode of CartPole with a trained `model` and gives back its return. It makes the world with `gym.make`, starts it with `env.reset(seed=seed)`, which gives `obs, info`, and plays until `terminated` or `truncated`; `env.step(action)` gives `obs, reward, terminated, truncated, info`, and `total` adds up every `reward`. Replace the `...` in the line `action, _ = ...` with a call of `model.predict` that gives the most probable action for `obs`.

        The check calls `play_test(steady, 0)`. `steady` is a stand-in for a trained model: it has a `predict` like a model of Stable-Baselines3, with a fixed policy inside instead of a network, the line `3 * obs[2] + obs[3] > 0` of *CartPole, MountainCar and the benchmarks of RL*. With the most probable action it gives that policy's action; when it draws, as in training, it pushes the other way one time in 4. Then four other cases: `steady` with seed 1, and two other stand-ins, `falling`, which pushes where the pole falls, and `leaning`, which pushes where the pole leans.
        """),
        L.task("Make `play_test` ask `model` for the most probable action at every step."),
    ])
    return


@app.cell(hide_code=True)
def exercise_play():
    box_play = L.code_box("play_test", lines=10)
    box_play
    return (box_play,)


@app.cell(hide_code=True)
def check_play(box_play):
    result_play = L.check("play_test", box_play.value)
    result_play
    return (result_play,)


@app.cell(hide_code=True)
def hints_play():
    L.hints(
        "play_test",
        [
            "Step 3 of the recipe: at every step, `action, _` should get what `model.predict` gives for `obs`, with the most probable action instead of a draw. Which argument of `model.predict` switches the draws off, and which value does it need?",
            "`predict` has the places `observation`, `state`, `episode_start` and `deterministic`, in that order, so a bare `True` in the second place lands in `state`. Give the argument by its name, with `=` between the name and the value.",
            "`        action, _ = model.predict(obs, ___)`",
        ],
        why="`action, _ = model.predict(obs, deterministic=True)` takes the most probable action at every step: `steady` then keeps the pole up for all 500 steps, `falling` until the cart runs off the track, and `leaning` until the pole falls.",
    )
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    In the picture, the return of your `play_test` with each stand-in and the seeds 0 to 4. A test with the most probable action shows what a policy does when it means it. With draws instead, as in training, `steady` keeps the pole up for 451 steps on average over the same five seeds, and with seed 4 for 334: a test that draws measures the policy and the dice together. In the demo below, your `play_test` tests real agents of Stable-Baselines3: their `predict` works the same way.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_judge():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Judging over seeds

        `judge(returns)` gives back two numbers about a set of test results: the median of the seeds, and the worst seed. `returns` is a NumPy array with a row per seed and a column per test episode, such as `returns_10k`: the test returns of the course's seeds 0 to 4 after 10,000 steps, 5 rows of 100. `per_seed` should hold the mean of each row, one number per seed; `np.median(per_seed)` is the median of those means, and `per_seed.min()` the smallest. Replace the `...`: the mean of each row of `returns`.

        The check calls `judge(returns_10k)`. Then three other cases: `returns_25k`, the same seeds after 25,000 steps; `returns_mixed`, 7 seeds of 20 test episodes each; and `returns_two`, 2 seeds of 3.
        """),
        L.task("Make `per_seed` hold one mean per seed."),
    ])
    return


@app.cell(hide_code=True)
def exercise_judge():
    box_judge = L.code_box("judge", lines=3)
    box_judge
    return (box_judge,)


@app.cell(hide_code=True)
def check_judge(box_judge):
    result_judge = L.check("judge", box_judge.value)
    result_judge
    return (result_judge,)


@app.cell(hide_code=True)
def hints_judge():
    L.hints(
        "judge",
        [
            "Step 5 of the recipe: one mean per seed. `per_seed` should be `returns.mean(...)` with the axis that runs along the test episodes of one seed, so that each row of `returns` gives one number. Which axis is that, in an array with a row per seed?",
            "Axis 0 runs down the columns, across the rows: a mean over the seeds, one for each test episode. The other axis runs along each row. Give it to `mean` by its name, `axis`.",
            "`    per_seed = returns.mean(___)`",
        ],
        why="`per_seed = returns.mean(axis=1)` gives one mean per seed: for `returns_10k` 416.39, 398.43, 292.57, 293.41 and 353.09, so a median of 353.09 and a worst seed of 292.57.",
    )
    return


@app.cell(hide_code=True)
def text_after_judge():
    mo.md(r"""
    In the picture, the five seeds of `returns_10k`, with your median and your worst seed. After 10,000 steps the best seed keeps the pole up for 416 steps on average and the worst for 293: picking one seed could make the same algorithm look a lot better or a lot worse. The median is the middle seed, which neither the luckiest nor the unluckiest seed can move far, and the worst seed tells how badly a run can go. The best seed alone tells neither.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Train and test five seeds yourself

    The demo trains PPO with the defaults of Stable-Baselines3, as in the worked example, for the steps you choose, with five seeds: 0 to 4, or 5 to 9. The chart shows the return of every episode of training, thin, and the mean of the last 10, thick; a seed that is done stays behind as a dashed line. Beside it, what the policy network does now: for every angle and angular velocity, with the cart in the middle and standing still, how often it pushes right. The line is where `3 * obs[2] + obs[3]` is 0, the policy that kept the pole up in *CartPole, MountainCar and the benchmarks of RL*. After training, your `play_test` tests every agent on 100 episodes, and your `judge` gives the median and the worst seed; where your functions do not pass their checks yet, the course's take their place. Start with 10,000 steps and seeds 0 to 4: about a minute on the course's computer.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    steps_choice = mo.ui.dropdown(options=C.steps_options(L.lang), value=C.number(10000, 0, L.lang), label=C.DEMO_TEXT[L.lang]["steps"])
    seeds_choice = mo.ui.dropdown(options=C.set_options(L.lang), value=C.DEMO_TEXT[L.lang]["set"].format(a=0, b=4), label=C.DEMO_TEXT[L.lang]["seeds"])
    train_button = mo.ui.run_button(label="Train and test five seeds")
    mo.hstack([steps_choice, seeds_choice, train_button], justify="start", gap=1.5, wrap=True)
    return seeds_choice, steps_choice, train_button


@app.cell(hide_code=True)
def demo_train(box_judge, box_play, result_judge, result_play, seeds_choice, steps_choice, train_button):
    if not train_button.value:
        _out = mo.md("*Pick the steps and the seeds, then press the button to train five agents and test them.*")
    else:
        _mine = result_play.state == "correct", result_judge.state == "correct"
        _play = run_code(box_play.value, C.P_NAMES).names["play_test"] if _mine[0] else C.play_test
        _judge = run_code(box_judge.value, C.J_NAMES).names["judge"] if _mine[1] else C.judge
        _steps, _seeds = steps_choice.value, C.seed_list(seeds_choice.value)
        _w = C.DEMO_TEXT[L.lang]
        _runs, _now = [], {}

        def _progress(_n, _run):
            if _now.get("run") is not _run:
                _now["run"] = _run
                _now["chart"].watch(lambda: C.now_picture(_now["run"], L.lang))
            _now["chart"].extend([(_n, _run["episodes"][-1][1])])

        for _k, _seed in enumerate(_seeds):
            _live = L.live(C.chart_labels(_seed, L.lang), C.actual(_steps), role="ink", low=0, high=500,
                           window=C.WINDOW, xticks=C.ticks(C.actual(_steps)))
            for _done in _runs:
                _live.reference(C.curve(_done), "", role="ink")
            _now["chart"] = _live
            _live.line(_w["learning"].format(s=_seed, k=_k + 1, n=len(_seeds)))
            _r = C.train(_seed, _steps, show=_progress, limit=150.0)
            if _r == "slow":
                break
            _runs.append(_r)
        if len(_runs) < len(_seeds):
            _out = _live.stopped("*A seed took more than two and a half minutes to train, so the demo stopped. Is your computer busy with something else?*")
        else:
            _live.line(_w["testing"].format(n=len(_runs)))
            _t = C.demo_test(_runs, _play, _judge)
            if _t == "error":
                _out = _live.stopped("*Your `play_test` or your `judge` gave an error, or something other than numbers. Do they still pass their checks?*")
            elif _t == "slow":
                _out = _live.stopped("*Testing took more than two minutes. Does your `play_test` do more work than it needs?*")
            else:
                _who = {(True, True): "your `play_test`", (True, False): "your `play_test` and the course's `judge`, since yours does not pass its check yet",
                        (False, True): "the course's `play_test`, since yours does not pass its check yet", (False, False): "the course's `play_test` and `judge`, since yours do not pass their checks yet"}[_mine]
                _out = mo.vstack([
                    L.explore("rewind", "steps", lambda _v: C.rewind_frame(_runs[0], int(_v), L.lang), start=_runs[0]["steps"],
                              values=C.looks(_runs[0]), role="",
                              task=f"This is seed {_runs[0]['seed']}. Go back to 0, then forward again: when does the picture of the policy start to look like the line, and what does the chart do then?"),
                    mo.md(C.table(_runs, _t["returns"], L.lang)),
                    mo.md(_w["verdict"].format(who=_who, median=C.number(_t["median"], 2, L.lang), worst=C.number(_t["worst"], 2, L.lang),
                                               n=len(_runs), steps=C.number(_steps, 0, L.lang))),
                ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With 10,000 steps, seeds 0 to 4 reach a mean of 293 to 416 steps in their tests, and none of them keeps the pole up in every test episode. With 25,000 steps, all five do, in all 100 test episodes each. Seeds 5 to 9 tell a less tidy story. After 25,000 steps, seed 5 still drops the pole in 2 of its 100 test episodes, and seed 8 in 1. And seed 8 gets worse before it gets better: after 5,000 steps, 6,144 really, its tests average 325 steps, after 10,000 only 199. More steps do not always help at once, and five seeds that all look perfect do not promise that a sixth will be.

    The thick line of the chart stays below the tests. In training PPO draws its actions, and every draw that goes the wrong way can end an episode early: seed 0 averages 111 steps over its last 10 training episodes at 10,000 steps, against 416 in its tests. Judge an agent by its tests, not by its training curve.

    The picture of the policy shows what the network learned. It pushes right where the pole falls to the right, and the angle tilts that border, a bit less than the line does. The network found a policy of its own, close to the one line that kept the pole up in *CartPole, MountainCar and the benchmarks of RL*, from rewards alone.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_fair():
    mo.md(r"""
    ### Exercise 3. A fair report

    A team compares its new agent with PPO on CartPole. It trains each one with 10 seeds and tests every trained agent on 100 episodes. With the new agent, 3 seeds keep the pole up in every test episode, and the other 7 reach 150, 175, 190, 205, 220, 240 and 260 steps on average. With PPO, 9 seeds keep the pole up in every test episode, and one reaches 410.
    """)
    return


@app.cell(hide_code=True)
def choice_fair_cell():
    choice_fair = L.choice(
        "fair",
        question="Which sentence reports the new agent fairly?",
        options={
            "best": "It keeps the pole up for all 500 steps in every test episode, as PPO does",
            "right": "Median 230, worst seed 150, and 3 of 10 seeds at 500: below PPO",
            "train": "Its best episode in training reached 500 steps, so it learns CartPole",
            "mean": "It reaches 500 steps on average, in its 3 best seeds",
        },
        correct="right",
        why={
            "right": "The median, the worst seed and the count of seeds at 500 come from all 10 seeds, so a reader sees the 3 good seeds and the 7 others. PPO, with a median of 500 and a worst seed of 410, does better.",
            "best": "That holds for 3 of the 10 seeds: what happened in the other 7?",
            "train": "An episode of training is one draw from the chances of the policy, not a test: how did the trained agents do in their tests?",
            "mean": "Picking the 3 best seeds after the fact leaves out 7: would that pick not make any algorithm look good?",
        },
    )
    choice_fair
    return (choice_fair,)


@app.cell(hide_code=True)
def text_exercise_library():
    mo.md(r"""
    ### Exercise 4. Which library?

    A student wants to see every line of PPO, from the batch of 2,048 steps to the nudge that keeps each change of the policy small, and then try another way to keep it small.
    """)
    return


@app.cell(hide_code=True)
def choice_library_cell():
    choice_library = L.choice(
        "library",
        question="Which library fits best?",
        options={
            "sb3": "Stable-Baselines3: the most tested PPO, used in a few lines",
            "right": "CleanRL: PPO in one file of a few hundred lines",
            "rllib": "RLlib: PPO spread over many cores and computers at once",
            "none": "None of them: a library cannot be read, so first write PPO from scratch",
        },
        correct="right",
        why={
            "right": "CleanRL puts all of PPO in one file, so the student can read it from top to bottom and change one line. Stable-Baselines3 spreads its PPO over classes in several files, made for using rather than reading, and RLlib is made for scale.",
            "sb3": "Stable-Baselines3 is made for using: its PPO is spread over several classes in several files. Which library keeps one algorithm in one file?",
            "rllib": "RLlib is made for running at scale, and its code is spread over a large package: which library is made to be read?",
            "none": "All three are open source, so all their code can be read: which one keeps PPO in a single file?",
        },
    )
    choice_library
    return (choice_library,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [CartPole, MountainCar and the benchmarks of RL](/course/open/rl-classic-environments?lang=en) a table with bins learned CartPole only in part. Here PPO from a library, with a neural network instead of a table, learned it in 25,000 steps, judged over five seeds of 100 test episodes each. A later level of the course opens PPO up and builds it. The next lesson, *Pavlov's dogs and Thorndike's cats: conditioning*, goes back to where learning from reward began: in psychology, long before computers.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fair, choice_library, result_judge, result_play):
    L.wrap_up(result_play, result_judge, choice_fair, choice_library)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
