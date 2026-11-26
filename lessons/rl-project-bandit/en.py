# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-project-bandit. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the course's agent, the demo and the pictures: checks.py; the testbed the course puts in your project: project/agent/testbed.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("rl-project-bandit", "en")
    C = L.checks   # the course's agent, loop and average, the demo and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Averaging as you go: drifting machines and optimistic starts](/course/open/rl-bandit-updates?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_drift():
    quick_drift = L.choice(
        "quick-drift",
        question="The means of a bandit drift a little after every pull. Which learning rate keeps an estimate close to what its machine pays now?",
        options={
            "average": "1/n, the sample average, since it uses every reward",
            "right": "A constant one, such as η = 0.1",
            "zero": "η = 0: the estimate stays put",
        },
        correct="right",
        why={
            "right": "With a constant η, every update multiplies the share of each older reward by 1 − η, so recent rewards count more, and the estimate follows a mean that drifts.",
            "average": "After a machine has given 1,000 rewards, 1/n moves its estimate only a thousandth of the way toward a new one: how fast does that follow a mean that drifts? See [Averaging as you go: drifting machines and optimistic starts](/course/open/rl-bandit-updates?lang=en).",
            "zero": "With η = 0, every update moves the estimate a part 0 of the way toward the reward: can it then learn what the machine pays now? See [Averaging as you go: drifting machines and optimistic starts](/course/open/rl-bandit-updates?lang=en).",
        },
    )
    quick_drift
    return (quick_drift,)


@app.cell(hide_code=True)
def quick_check_optimistic():
    quick_optimistic = L.choice(
        "quick-optimistic",
        question="A greedy agent with η = 0.1 starts the estimates of its ten machines at 5, far above every reward. What does it do in its first ten pulls?",
        options={
            "same": "It pulls machine 0 ten times, since all estimates are equal",
            "right": "It tries each of the ten machines once, in turn",
            "random": "Ten random pulls, as ε-greedy would",
        },
        correct="right",
        why={
            "right": "Every reward disappoints: a first reward $r$ moves the estimate of its machine to 5 + 0.1 · ($r$ − 5) = 4.5 + 0.1$r$, below the machines still at 5. So the greedy agent goes on to the next machine, until it has tried them all.",
            "same": "Machine 0 goes first, since the estimates are equal. After its first reward, where does its estimate stand against the nine machines still at 5? See [Averaging as you go: drifting machines and optimistic starts](/course/open/rl-bandit-updates?lang=en).",
            "random": "A greedy agent never pulls a random machine: which machine has the largest estimate after the first reward? See [Averaging as you go: drifting machines and optimistic starts](/course/open/rl-bandit-updates?lang=en).",
        },
    )
    quick_optimistic
    return (quick_optimistic,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        In [Which machine pays most? The multi-armed bandit](/course/open/rl-bandits?lang=en) and [Averaging as you go: drifting machines and optimistic starts](/course/open/rl-bandit-updates?lang=en), the course wrote the agents and let them play for you. In this project you build an agent of your own, in your own project folder, in the shape agents have in practice: an object that chooses and learns, an environment that answers, and a loop between the two. The same three parts come back in every lesson of this course where an agent learns from experience, even when the environment is a grid world or a game.

        Building an agent is half the work. The other half is finding out whether it is any good: better than another agent, on which problems, and by which measure. A comparison on different problems, or on too few runs, can make the worse agent look better. So your project also gets code to measure with.

        **An everyday example.** Three cyclists want to know which of them rides fastest. Each rides once, on a different day, on a different road, in a different wind. Does the fastest time say who rides fastest? On home trainers, with the same route and the same resistance for all three, they learn more, and more still when each of them rides it many times.
        """),
        draw.project_steps(["Your bandit agent", "Your grid agent"], 0,
                           here_label="← today", title="Where your project stands: agent"),
    ])
    return


@app.cell(hide_code=True)
def predict_reward():
    guess_reward = L.guess(
        "reward",
        question="In this lesson, three versions of your agent play the same 200 drifting bandits, 10,000 pulls each, all with the learning rate 0.1: greedy with every estimate starting at 0, ε-greedy with $\\varepsilon$ = 0.1 from 0, and greedy with every estimate starting at 5. Which one gets the most reward over all 10,000 pulls, do you think?",
        options={
            "greedy": "Greedy, from 0",
            "epsilon": "ε-greedy, from 0",
            "right": "Optimistic, from 5",
            "same": "All three about the same",
        },
        reveal="**Optimistic, from 5.** Over all 10,000 pulls the optimistic agent gets 1.77 per pull, ε-greedy 1.66 and greedy 1.55. Over the first 100 pulls it is last by far, with 0.35 against 0.92 and 0.97. The demo lets your own agents play the same run.",
    )
    guess_reward
    return (guess_reward,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Two objects.** In this project the bandit and the agent are each an object of a class, in a file of your project, as in [Your own module: keeping project code in a file](/course/open/py-own-modules?lang=en).

        - The environment is a `DriftingBandit`, from `work/projects/agent/testbed.py`, which the course put in your project. `DriftingBandit(k, drift, rng)` makes a bandit of `k` machines whose means start around 0, with a standard deviation of 1. `pull(action)` gives the reward of one pull of machine `action`, its mean plus noise, and then lets every mean take a random step with the standard deviation `drift`, as in [Averaging as you go: drifting machines and optimistic starts](/course/open/rl-bandit-updates?lang=en). `best()` gives the machine with the highest mean of that moment. Only the measuring uses `best()`: the agent never sees the means.
        - The agent is your `BanditAgent`, in `work/projects/agent/bandit.py`, which you write in exercise 1. `act()` chooses a machine, ε-greedy; `learn(action, reward)` counts the reward and moves the estimate of that machine toward it, with the update of [Averaging as you go: drifting machines and optimistic starts](/course/open/rl-bandit-updates?lang=en). What it knows, a list `q` of estimates and a list `counts`, it keeps in itself, from one pull to the next.

        **One loop.** Between the two runs a loop, your `run` of exercise 2: the agent acts, the bandit answers with a reward, the agent learns from it, and so on, pull after pull. It is the loop of agent and environment of [Learning from rewards](/course/open/rl-what-is-rl?lang=en). A bandit has a single state, so the reward is all the agent observes. The picture shows one turn of that loop.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        The agent has made 53 pulls on this bandit and has tried 9 of its 10 machines. On pull 54 it chooses machine 7, which has the largest estimate: 0.97, after 40 pulls. The best machine at that moment is machine 6, with a mean of 0.87 against 0.80 for machine 7, but its two pulls so far paid little, so the agent estimates it at −0.02. Machine 7 pays −1.97 this time, and `learn` moves its estimate a tenth of the way down, to 0.68.

        **A fair comparison.** One pull says little, and one run on one bandit not much more: in the worked example below, a machine with a mean of 0.13 pays 1.49 on one pull. So your project lets every agent play 200 bandits and takes the mean over the runs at every pull, as in the learning curves of [Which machine pays most? The multi-armed bandit](/course/open/rl-bandits?lang=en). And it gives every agent the same 200 bandits. Bandit `b` is made with the seed `b` for every agent, so it starts with the same means. A bandit draws its noise and its drift from its own random number generator, in the same order whatever machine the agent pulls, so the means also drift in the same way for every agent, and the same machine pays the same at the same pull. A difference between the agents then comes from the agents, and not from bandits that happened to be easier.

        **Two measures.** Your project measures each agent in two ways, at every pull: the **mean reward**, over the bandits, and the **share of best choices**: in what share of the bandits the agent chose the best machine of that moment. Often the two agree. Exercise 4 is about a case where they do not.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **testbed** | | a fixed set of problems to compare methods on, such as the course's drifting bandits | `testbed.py` |
        | **run** | | one agent playing one bandit, from the first pull to the last | `run` |
        | **mean reward** | | the reward at one pull, averaged over the bandits | `rewards`, averaged over the runs |
        | **share of best choices** | | in what share of the bandits the agent chose the best machine of that moment | `best`, averaged over the runs |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    An agent of your own on a testbed:

    1. Write the agent as a class: in `__init__` what it knows, an estimate and a count for every machine; in `act` its choice; in `learn` what it does with a reward.
    2. Write the loop: at every pull the agent acts, you note whether it chose the best machine, the bandit pays, the agent learns, and you keep the reward.
    3. Make many bandits, the same ones for every agent you compare, and give every agent its own copy of each.
    4. Let every agent play every bandit with your loop, and keep its rewards and its best choices.
    5. At every pull, average over the runs: the mean reward and the share of best choices.
    6. Compare the agents by both measures, over the whole run and over parts of it.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell imports `DriftingBandit` from your project, makes a bandit with seed 0, and shows its means and its best machine. Then it pulls the best machine and machine 0 alternately, three times each, and pulls 10,000 times more before it shows the means again. Run the cell.
    """)
    return


@app.cell
def example_testbed():
    from agent.testbed import DriftingBandit

    # a bandit of 10 machines whose means drift with standard deviation 0.01 at every pull, made with seed 0
    _bandit = DriftingBandit(10, 0.01, np.random.default_rng(0))
    print("means:", np.round(_bandit.means, 2))
    _b = _bandit.best()
    print("best:", _b)
    # three turns: a pull of the best machine, then one of machine 0
    for _t in range(3):
        print("machine", _b, "pays", round(_bandit.pull(_b), 2), "and machine 0 pays", round(_bandit.pull(0), 2))
    # 10,000 pulls later, every mean has drifted
    for _t in range(10_000):
        _bandit.pull(0)
    print("means:", np.round(_bandit.means, 2))
    print("best:", _bandit.best())
    return (DriftingBandit,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    At the start, machine 6 has the highest mean, 1.30, and machine 7 comes next with 0.95. One pull says little: machine 0, with a mean of 0.13, pays 1.49 on the first turn, more than twice what machine 6 pays then, 0.68. After 10,000 more pulls every mean has moved: machine 7 went from 0.95 to 0.21, machine 4 from −0.54 to −1.17. Here machine 6 is still the best, now at 1.57, but on 115 of the course's 200 bandits the best machine at pull 10,000 is a different one from the one at the start.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_agent():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Your BanditAgent

        `BanditAgent(k, epsilon, eta, start, rng)` is an agent for a bandit of `k` machines; `import numpy as np` is at the top. `__init__` is given: it keeps `epsilon`, the learning rate `eta`, the random number generator `rng`, a list `q` with the estimate `start` for every machine, and a list `counts` of zeros. When `eta` is `None`, the agent takes the learning rate 1/n, the sample average. In `act`, the random choice with probability `epsilon` is given, with `self.rng.random()` and `self.rng.integers`.

        Replace the two `...`. In `act`: the machine with the largest estimate, the first one when several are equal, as a whole number. In `learn(action, reward)`: add 1 to the count of `action`, take the learning rate, and move the estimate of `action` toward `reward`.

        The check makes `BanditAgent(3, 0.0, 0.5, 0.0, np.random.default_rng(0))`, gives it the rewards 2.0 and 4.0 for machine 1 and 1.0 for machine 2, and looks at `agent.q`, `agent.counts` and `agent.act()`: they should be `[0.0, 2.5, 0.5]`, `[0, 2, 1]` and `1`. Then it tries three cases of its own: `eta` None, an optimistic start, and an agent that explores half of the time. When it is right, the check saves your class in your project, as `work/projects/agent/bandit.py`.
        """),
        L.task("Make `act` choose the machine with the largest estimate, and `learn` count the reward and move the estimate toward it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_agent():
    box_agent = L.code_box("agent", lines=26)
    box_agent
    return (box_agent,)


@app.cell(hide_code=True)
def check_agent(box_agent):
    result_agent = L.check("agent", box_agent.value)
    result_agent
    return (result_agent,)


@app.cell(hide_code=True)
def hints_agent():
    L.hints(
        "agent",
        [
            "`act` gives back the place of the largest value in `self.q`, not the value itself, as a plain whole number. `learn` adds 1 to `self.counts[action]`, gives a name `eta` the learning rate, `1 / self.counts[action]` or `self.eta`, and moves `self.q[action]` a part `eta` of the way toward `reward`.",
            "`np.argmax` finds the place, the first one when several are equal, and `int` makes it a plain whole number. In the example, `learn(1, 2.0)` with the learning rate 0.5 moves the estimate of machine 1 halfway from 0.0 to 2.0, to 1.0; `learn(1, 4.0)` moves it halfway from 1.0 to 4.0, to 2.5. Count before you take 1/n: the first reward of a machine has n = 1.",
            "`        self.q[action] = self.q[action] + eta * ___`",
        ],
        why="In `act`: `return int(np.argmax(self.q))`. In `learn`: `self.counts[action] = self.counts[action] + 1`; then `eta = 1 / self.counts[action]` when `self.eta is None`, else `eta = self.eta`; and `self.q[action] = self.q[action] + eta * (reward - self.q[action])`, the update of [Averaging as you go: drifting machines and optimistic starts](/course/open/rl-bandit-updates?lang=en), now kept in the object.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_run(result_agent):
    result_agent  # made again after the check of exercise 1, which saves your agent
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The loop

        `run(agent, bandit, pulls)` lets `agent` and `bandit` take turns `pulls` times; `for t in range(pulls)` counts the turns. It gives back two lists with a number for every pull: `rewards`, the reward of that pull, and `best`, 1 when the agent chose the best machine of that moment and 0 otherwise. Replace the `...` with one turn: the agent chooses a machine, `best` gets 1 or 0, the bandit pays, the agent learns from the reward, and `rewards` keeps it. Ask the bandit for its best machine before the pull: `pull` lets every mean drift.

        The check imports your `BanditAgent` and the course's `DriftingBandit` from your project, and lets a greedy agent with the learning rate 0.5 play 4 pulls on a bandit of 3 machines that does not drift. The rewards, to three decimals, `best` and the estimates should be `[-2.252, 0.251, -0.833, -0.512]`, `[1, 0, 0, 0]` and `[-1.126, -0.354, -0.256]`: machine 0 was the best, but its first reward, −2.252, made the greedy agent leave it. Then it tries three cases of its own: a second call that goes on where the first stopped, means that drift a lot, and 0 pulls. When it is right, the check saves your `run` as `work/projects/agent/run.py`.
        """),
        L.course_versions(),
        L.task("Make `run` let the agent and the bandit take turns, and keep the reward and whether the agent chose the best machine."),
    ])
    return


@app.cell(hide_code=True)
def exercise_run():
    box_run = L.code_box("run", lines=16)
    box_run
    return (box_run,)


@app.cell(hide_code=True)
def check_run(box_run, result_agent):
    result_agent  # check again when exercise 1 saves your agent
    result_run = L.check("run", box_run.value)
    result_run
    return (result_run,)


@app.cell(hide_code=True)
def hints_run():
    L.hints(
        "run",
        [
            "One turn, in the order of step 2 of the recipe: `agent.act()` gives the machine, `bandit.best()` says whether it was the best one, `bandit.pull` gives the `reward`, `agent.learn` takes it, and `append` adds a number to `best` and to `rewards`.",
            "In the example, the first turn chooses machine 0, which is the best, so `best` gets 1. The bandit pays −2.252, and the agent moves its estimate of machine 0 halfway, to −1.126. In the second turn machines 1 and 2 share the largest estimate, 0, and `act` takes the first of the two.",
            "`        if ___:`",
        ],
        why="`action = agent.act()`, then `if action == bandit.best():` with `best.append(1)` and `else:` with `best.append(0)`, then `reward = bandit.pull(action)`, `agent.learn(action, reward)` and `rewards.append(reward)`. `best.append(int(action == bandit.best()))` does the same in one line.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Let three of your agents play

    The button lets three versions of your `BanditAgent` play the same 200 drifting bandits with your `run`, 10,000 pulls each, all with the learning rate 0.1: greedy from 0, ε-greedy with $\varepsilon$ = 0.1 from 0, and greedy with every estimate starting at 5, the optimistic agent. While your `bandit.py` or `run.py` is not there yet, or your `bandit.py` does not give what the course expects, the course's version stands in. The demo goes like this:

    ```python
    for name in ["greedy", "epsilon", "optimistic"]:
        for b in range(200):   # bandit b gets the seed b, for every agent
            agents[name][b] = BanditAgent(10, epsilon[name], 0.1, start[name], np.random.default_rng(1000 + b))
            bandits[name][b] = DriftingBandit(10, 0.01, np.random.default_rng(b))
    for t in range(0, 10000, 100):   # 100 pulls at a time, so that the chart can grow
        for name in ["greedy", "epsilon", "optimistic"]:
            for b in range(200):
                rewards, best = run(agents[name][b], bandits[name][b], 100)
    ```

    The chart shows the mean reward of the ε-greedy agent over the 200 bandits: thin over every 10 pulls, thick its mean over the last 100 pulls; dashed, the same mean for the optimistic agent and, lighter, for the greedy one. Beside it you see what the three agents know of one of the 200 bandits: the means of that moment as short lines, the estimates in <span class="c-param">orange</span>, open for a machine that was never pulled, and how often each machine was pulled. It takes about half a minute; then you can go back through the run.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    run_button = mo.ui.run_button(label="Let my three agents play")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def demo_run(result_agent, result_run, run_button):
    result_agent, result_run   # play again with your files once a check saves them
    trained = None
    if not run_button.value:
        _out = mo.md("*Press the button to let the three agents play.*")
    else:
        _w = C.live_words(L.lang)
        _now = {}   # the live chart, what the agents know of one bandit after the last 100 pulls, and its pull
        _raw = {_a: [] for _a in C.AGENTS}   # the mean reward over the bandits of each agent at every pull

        def _progress(_t, _means, _seen):
            if "seen" not in _now:   # from the first 100 pulls on, what the agents know beside the chart
                _now["chart"].watch(lambda: C.now_picture(_now["seen"], _now["t"], L.lang))
            _now["seen"], _now["t"] = _seen, _t
            for _a in C.AGENTS:
                _raw[_a].extend(_means[_a])
            if _t % 500 == 0:
                _now["chart"].reference(C.smoothed(_raw["optimistic"]), "", role="ink", replace=True)
                _now["chart"].reference(C.smoothed(_raw["greedy"]), "", role="ink-3", replace=True)
            _now["chart"].extend([(_u / 1000, C.mean_of(_raw["epsilon"], _u - 9, _u)) for _u in range(_t - 90, _t + 1, 10)],
                                 C.progress(_t, L.lang))

        _now["chart"] = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"], "alt": _w["alt"]},
                               C.PULLS / 1000, role="ink", low=-0.5, high=2.5, window=10, key=C.live_key(L.lang))
        _now["chart"].line("**Playing** …")
        _run = C.demo(show=_progress)
        if _run == "run":
            _out = _now["chart"].stopped("*Your `work/projects/agent/run.py` stops with an error, or has no `run` in it. Does your `run` still pass its check?*")
        elif _run == "stopped":
            _out = _now["chart"].stopped("*Your `BanditAgent` or your `run` stopped with an error during the run. Do both still pass their checks?*")
        elif _run == "wrong":
            _out = _now["chart"].stopped("*Your `run` gave back something other than two lists with a number for every pull. Does it still pass its check?*")
        elif _run == "slow":
            _out = _now["chart"].stopped("*The run took more than two and a half minutes. Do your `act`, `learn` and `run` do more work than they need?*")
        else:
            trained = _run
            _who = {("bandit", "run"): "your `BanditAgent` and your `run`",
                    ("bandit",): "your `BanditAgent` and the course's `run`, since your `run.py` is not there yet",
                    ("run",): "your `run` and the course's `BanditAgent`, since your `bandit.py` is not there yet or does not give what the course expects",
                    (): "the course's `BanditAgent` and `run`, since yours are not in your project yet"}[tuple(_run["yours"])]
            _all = {_a: float(_run["rewards"][_a].mean()) for _a in C.AGENTS}
            _out = mo.vstack([
                L.explore("rewind", "pull", lambda _v: (C.rewind_frame(_run, int(_v), L.lang), C.rewind_text(_run, int(_v), L.lang)),
                          start=C.PULLS, values=C.LOOKS, role="",
                          task="Go back to pull 100, then forward again: from which pull on is machine 7 the best machine of this bandit, and which agent pulls it from then on?"),
                mo.md(f"Played with {_who}. Over all 10,000 pulls, the mean reward per pull was {C.number(_all['greedy'], 2, L.lang)} for greedy, {C.number(_all['epsilon'], 2, L.lang)} for ε-greedy and {C.number(_all['optimistic'], 2, L.lang)} for the optimistic agent."),
            ])
    _out
    return (trained,)


@app.cell(hide_code=True)
def text_after_demo():
    mo.vstack([
        mo.md(r"""
        With the course's agents, the order changes during the run. Over the first 100 pulls the optimistic agent is last by far, with 0.35 per pull against 0.92 for ε-greedy and 0.97 for greedy: it spends those pulls trying every machine, the bad ones too. From about pull 240 on it is ahead of both, and it stays ahead to the end. Over all 10,000 pulls it gets 1.77 per pull, ε-greedy 1.66 and greedy 1.55.

        The bandit beside the chart, one of the 200, shows what the optimistic agent misses. At pull 100, machine 7 has a mean of −0.30. The optimistic agent tries it 15 times in its first 223 pulls, while its mean stays below 0, and after that only 6 times more, the last time at pull 2,792. Then machine 7 drifts up: by pull 8,000 it is the best machine of the bandit, and at the end its mean is 2.40. The ε-greedy agent, which keeps pulling a random machine now and then, finds out: by the end it has pulled machine 7 a total of 2,263 times, and estimates it at 2.46. The optimistic agent still estimates it at 0.53, and the greedy agent never pulled it; both end on machine 6, with a mean of 0.96. A high start makes an agent explore at the start only: once every estimate has come down, the optimistic agent is as greedy as the greedy one, and a machine that gets better later goes unnoticed.
        """),
        L.watch_out("On a few bandits, luck decides the order. On the first of the 200 bandits alone, greedy gets more than ε-greedy over the 10,000 pulls, 1.82 per pull against 1.65, and on the first three together still 1.80 against 1.66; over all 200 it is the other way around. Per bandit, the optimistic agent gets the most on 114 of the 200, greedy on 64 and ε-greedy on only 22: greedy is ahead on many bandits, but far behind on others."),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_average():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. The mean over many runs

        `average(runs)` takes a list with one list for every run, all of the same length, with a number for every pull. It gives back a list `means` with the mean over the runs at every pull: a learning curve. Your project uses it for both measures: on the rewards it gives the mean reward, on the 1s and 0s of `best` the share of best choices. Replace the `...` with the mean at pull `t`, appended to `means`.

        `small_runs`, the course's `SMALL_RUNS`, holds three runs of four pulls, with a 1 where the agent chose the best machine: `[[0, 1, 1, 1], [0, 0, 1, 1], [1, 0, 0, 1]]`. `average(small_runs)` gives `[0.333, 0.333, 0.667, 1.0]`: at the first pull, one run of the three chose the best machine. The check calls that, then four cases of its own, with rewards, a single run and four runs. When it is right, it saves your function as `work/projects/agent/measure.py`.
        """),
        L.task("Make `average` give back the mean over the runs at every pull."),
    ])
    return


@app.cell(hide_code=True)
def exercise_average():
    box_average = L.code_box("average", lines=10)
    box_average
    return (box_average,)


@app.cell(hide_code=True)
def check_average(box_average):
    result_average = L.check("average", box_average.value)
    result_average
    return (result_average,)


@app.cell(hide_code=True)
def hints_average():
    L.hints(
        "average",
        [
            "For each pull `t`: a `total` that starts at 0, a `for` loop over the runs that adds pull `t` of each run to it, and then the mean over the runs, added to the list with `means.append`.",
            "In the example, at the first pull the three runs have 0, 0 and 1: a total of 1, divided by the 3 runs, gives 0.333. At the last pull all three have 1, so the mean is 3 / 3 = 1.0.",
            "`        means.append(___)`",
        ],
        why="`total = 0`, then `for one in runs:` with `total = total + one[t]` under it, and `means.append(total / len(runs))`. NumPy does it in one line: `np.mean(runs, axis=0)`, the mean along axis 0, over the runs.",
    )
    return


@app.cell(hide_code=True)
def text_project_demo():
    mo.md(r"""
    ### What your project can do now

    Your project now has all its parts: an agent, a loop and a measure. The slider moves a window of 1,000 pulls over the run of the demo above, from the first 1,000 pulls to the last 1,000. The two charts show both measures for all three agents, made with your `average` once it passes its check, each as its mean over the last 100 pulls. The band marks the window, and the sentence gives both measures over it.
    """)
    return


@app.cell(hide_code=True)
def project_demo(box_average, result_average, trained):
    if trained is None:
        _out = mo.md("*Let the three agents play first, with the button above: the slider measures their run.*")
    else:
        _yours = C.mine(box_average.value, result_average.state == "correct")
        _curves = C.curves(trained, _yours or C.average)
        if _curves == "stopped":
            _out = mo.md("*Your `average` stopped with an error on the runs of the demo. Does it still pass its check?*")
        elif _curves == "wrong":
            _out = mo.md("*Your `average` gave back something other than a list with a number for every pull. Does it still pass its check?*")
        elif _curves == "slow":
            _out = mo.md("*Your `average` took more than 30 seconds for the six curves. Does it do more work than it needs?*")
        else:
            _who = "your `average`" if _yours else "the course's `average`, since yours does not pass its check yet"
            _out = mo.vstack([
                L.explore("window", "last pull", lambda _v: C.window_frame(trained, _curves, int(_v), L.lang),
                          start=1000, values=C.WINDOWS, role="",
                          task="Move the window from the first 1,000 pulls to the last: from which window on does ε-greedy choose the best machine more often than the optimistic agent, and which agent gets the most reward in every window?"),
                mo.md(f"Measured with {_who}."),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_project_demo():
    mo.md(r"""
    Over pulls 1 to 1,000, both measures put the optimistic agent first: 1.34 per pull and the best machine in 68.1% of the bandits, against 1.27 and 61.0% for ε-greedy and 1.19 and 43.6% for greedy. From the window of pulls 4,001 to 5,000 on, the measures disagree. The optimistic agent still gets the most reward in every window, but ε-greedy chooses the best machine more often: over the last 1,000 pulls in 80.4% of the bandits, against 69.7%. Greedy is last by both measures in every window.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_measures():
    mo.md(r"""
    ### Exercise 4. Two measures, two orders

    Over the last 1,000 pulls of the demo, the ε-greedy agent chooses the best machine in 80.4% of the bandits, the optimistic agent in 69.7%. Yet the optimistic agent gets more per pull: 2.03 against 1.95. The course let the same three agents play 2,000 other drifting bandits and found the same: over the last 1,000 pulls, ε-greedy chooses the best machine in 80.4% of the bandits against 73.6%, and gets 1.91 per pull against 2.02.
    """)
    return


@app.cell(hide_code=True)
def choice_measures_cell():
    choice_measures = L.choice(
        "measures",
        question="How can the agent that chooses the best machine less often get more reward?",
        options={
            "easier": "The optimistic agent happened to play easier bandits",
            "right": "When ε-greedy misses, it is often a random pull that costs a lot; when the optimistic agent misses, it mostly pulls a machine that pays almost as much",
            "start": "The start value of 5 still counts at the end: it keeps the estimates of the optimistic agent high, and high estimates add to the rewards it gets from its machines at every pull",
            "luck": "Chance: over 1,000 pulls, the noise of the rewards decides",
        },
        correct="right",
        why={
            "right": "The share of best choices counts every miss the same, the reward does not. On the 2,000 other bandits, over the last 1,000 pulls, a miss of ε-greedy cost 1.19 on average against the best machine of that moment, a miss of the optimistic agent 0.46. Nine tenths of what ε-greedy loses comes from its random pulls: one pull in ten it takes any machine, even the worst. The optimistic agent misses more often, but mostly with a machine that pays almost as much as the best one.",
            "easier": "The demo gives every agent the same 200 bandits, with the same means, drift and noise: could the bandits explain the difference? What does a miss of each agent cost?",
            "start": "An estimate is what the agent expects, not what it gets: the reward comes from the mean of the machine, whatever the agent estimates. And after thousands of pulls with η = 0.1, how much of the start value is left?",
            "luck": "Every share and every mean reward here is over 200 bandits and 1,000 pulls, and 2,000 other bandits give the same order. What does a miss of each agent cost?",
        },
    )
    choice_measures
    return (choice_measures,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Which machine pays most? The multi-armed bandit](/course/open/rl-bandits?lang=en) and [Averaging as you go: drifting machines and optimistic starts](/course/open/rl-bandit-updates?lang=en), the course's code played the bandits. Now your project `agent` has a `BanditAgent`, a loop that lets it play and a measure, next to the course's testbed. Your comparison showed that the optimistic start pays off on these bandits, that ε-greedy keeps finding machines that get better, and that a fair comparison needs the same bandits for every agent, many runs and more than one measure. Your project comes back in *Project: an agent for your own grid world*, where the environment also has states. The next lesson, *What is a policy worth? Policy evaluation*, starts the part on RL with a table: problems with many states, where the agent knows how the environment works.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_measures, result_agent, result_average, result_run):
    L.wrap_up(result_agent, result_run, result_average, choice_measures)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
