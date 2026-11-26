# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-bandits. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the bandits, the runs and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rl-bandits", "en")
    C = L.checks   # the bandits, the course's agents and runs, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en) and [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_greedy():
    quick_greedy = L.choice(
        "quick-greedy",
        question="An agent chooses ε-greedy among four actions, with $\\varepsilon$ = 0.1. About how often does it take the action with the largest $Q$?",
        options={
            "ten": "About 10 in 100",
            "right": "About 92.5 in 100: a random action can be the best one too",
            "ninety": "Exactly 90 in 100, since the other 10 are random actions and never the best one",
        },
        correct="right",
        why={
            "right": "In 90 of 100 choices it takes the best action. In the other 10 it takes a random one of the four, the best one in a quarter of them: 90 + 2.5 = 92.5.",
            "ten": "$\\varepsilon$ = 0.1 is the probability of a random action: how often does the agent take the best one in the other choices? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
            "ninety": "A random action is one of all four actions, so can it be the best one too? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
        },
    )
    quick_greedy
    return (quick_greedy,)


@app.cell(hide_code=True)
def quick_check_spread():
    quick_spread = L.choice(
        "quick-spread",
        question="The rewards of a machine have a mean of 1 and a standard deviation of 1. What does that say about them?",
        options={
            "all": "Every reward is exactly 1",
            "right": "Around 1, typically about 1 away from it, some closer, some further",
            "exact": "All rewards lie between 0 and 2, since the standard deviation says how far they go at most",
        },
        correct="right",
        why={
            "right": "The standard deviation says how far the values typically lie from the mean: here about 1, some closer and some further.",
            "all": "A standard deviation of 1 means that the values spread around the mean: would they spread if all were 1? See [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).",
            "exact": "The standard deviation is how far values typically lie from the mean, not a limit: can a value lie further away? See [Describing data: mean, median and spread](/course/open/stats-describe?lang=en).",
        },
    )
    quick_spread
    return (quick_spread,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A news site has four headlines for one story and wants the one that gets the most clicks. A web shop tries two buttons. Each try tells a little, and each visitor who sees a poor headline is a click lost. Showing only the headline that did best so far is not enough: one that started badly might be better. These are the simplest problems of reinforcement learning: a choice made again and again, with a reward after each one, and no states to move between.

    **An everyday example.** In a new canteen, you try the dishes. After a few lunches, do you keep taking the one that tasted best so far, or try another now and then, in case it is better?

    A slot machine with one lever is called a *one-armed bandit*: it pays a random amount, and on average less than you put in. A row of them, each paying around a mean of its own, gives the problem its name: the **multi-armed bandit**. Which machine pays most?
    """)
    return


@app.cell(hide_code=True)
def predict_greedy():
    guess_greedy = L.guess(
        "greedy",
        question="A **greedy** agent always pulls the machine with the largest estimate so far. The course lets it play 2,000 random bandits of ten machines, for 1,000 pulls each. In how many bandits does it pull the best machine at the end?",
        options={
            "most": "In about 9 of 10 bandits",
            "right": "In about 1 of 3",
            "few": "In only about 1 of 10",
        },
        reveal="**In about 1 of 3.** Over the last 100 of the 1,000 pulls, greedy pulls the best machine in 36.7% of the 2,000 bandits. It keeps pulling a machine that paid well so far, and tries a new one only when every machine it tried has paid less than 0 on average: of the ten machines, it tries 3.3 on average. Exercise 2 counts this share.",
    )
    guess_greedy
    return (guess_greedy,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A bandit in this lesson has ten machines, numbered 0 to 9; each machine is an *arm*, an action of the agent. A pull gives a random reward around the mean of that machine, with a standard deviation of 1. The means differ from machine to machine, and the agent does not know them. Unlike GRID in [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en), a bandit has one state: a pull changes nothing about the next one. The picture shows one bandit, with 40 pulls of each machine.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        Machine 9 pays most, 3.32 on average, and machine 0 comes second with 2.04; the mean of a machine is the expected value of its reward, as in [Expected value and variance](/course/open/prob-expectation?lang=en). With 40 pulls each, you see it in the picture. The agent has to find out by pulling, and every pull it spends on a poor machine is reward lost.

        Its estimate of a machine is the **sample average**: the total of the rewards it got from that machine, divided by its number of pulls $N(a)$. Before the first pull of a machine, its estimate is 0.
        """),
        L.formula(r"[[param:Q(a)]] = \frac{[[data:r_1]] + [[data:r_2]] + \dots + [[data:r_{N(a)}]]}{N(a)}"),
        mo.md(r"""
        With these estimates, the agent of [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en) has its two choices again. Exploitation pulls the machine with the largest estimate; exploration pulls another, to find out what it pays. On the bandit of the picture, with 200 pulls, the greedy agent pulls machine 0 all 200 times: its first reward put machine 0 above the 0 of the others, so it never tried them. An agent with ε-greedy and $\varepsilon$ = 0.1 pulls machine 0 94 times and machine 9 90 times, and gets 2.43 per pull against 2.09. Each pull spent exploring costs reward now, and each pull spent exploiting can miss a better machine: the **exploration-exploitation trade-off**.

        One bandit says little: an agent can be lucky. The course draws 2,000 bandits, each with ten means drawn around 0 with a standard deviation of 1, lets every agent play each bandit for 1,000 pulls, and takes the mean reward at every pull over the 2,000. Sutton and Barto made this test in their book on reinforcement learning. Drawn against the pull, the mean reward shows how fast an agent learns, and how much it gets in the end.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **multi-armed bandit** | | machines, each an action, that pay a random reward around a mean the agent does not know; there is one state | `means` |
        | **sample average** | | the estimate of a machine: the total of its rewards divided by its number of pulls | `estimates` |
        | **exploration-exploitation trade-off** | | exploring costs reward now, exploiting can miss a better action | `epsilon` |
        | **greedy** | | always taking the action with the largest estimate: ε-greedy with $\varepsilon$ = 0 | |
        | **pull** | | one step of the agent: choosing a machine and getting its reward | |
        | $Q(a)$ | Q of a | the estimate of machine $a$ | `q[a]` |
        | $N(a)$ | N of a | the number of pulls of machine $a$ so far | `counts[a]` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    An agent with sample averages on a bandit:

    1. Give every machine a total of 0 and a count of 0; every estimate is 0.
    2. Choose a machine, ε-greedy: with probability $\varepsilon$ a random one, else the one with the largest estimate.
    3. Pull it: the bandit gives a reward.
    4. Add the reward to the total of that machine, and 1 to its count.
    5. Work out its estimate again: its total divided by its count.
    6. Go on at step 2, until the last pull. To judge the agent, let it play many bandits and take the mean reward at every pull.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A small bandit of three machines, with means 0.5, 1.5 and −0.5, and $\varepsilon$ = 0.2: ten pulls through steps 1 to 5. `rng.normal()` draws a random number around 0 with a standard deviation of 1, so `_means[_a] + _rng.normal()` is a reward around the mean of machine `_a`. Each line shows the pull, the machine, the reward, the three estimates and the three counts. Run the cell.
    """)
    return


@app.cell
def example_bandit():
    import numpy as np

    # a small bandit of three machines: their means, which the agent does not know
    _means = [0.5, 1.5, -0.5]
    _rng = np.random.default_rng(9)

    # 1. a total and a count of 0 for every machine; every estimate is 0
    _totals = [0.0, 0.0, 0.0]
    _counts = [0, 0, 0]
    _q = [0.0, 0.0, 0.0]
    for _t in range(1, 11):
        # 2. epsilon-greedy with epsilon = 0.2: a random machine, else the one with the largest estimate
        if _rng.random() < 0.2:
            _a = int(_rng.integers(0, 3))
        else:
            _a = int(np.argmax(_q))
        # 3. pull it: a reward around the mean of the machine
        _r = _means[_a] + _rng.normal()
        # 4. add the reward to the total of the machine, and 1 to its count
        _totals[_a] = _totals[_a] + _r
        _counts[_a] = _counts[_a] + 1
        # 5. the estimate of the machine again: its total divided by its count
        _q[_a] = _totals[_a] / _counts[_a]
        print(_t, _a, round(_r, 2), [round(_x, 2) for _x in _q], _counts)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    On pull 1 every estimate is 0, and `np.argmax` takes the first: machine 0. Its reward of 0.74 puts machine 0 above the others, and the agent keeps taking it; its estimate moves with every reward, 0.74, 0.95, 0.65 and 0.67, each time the total so far divided by the count. Pulls 5 and 6 are random choices, since greedy would have taken machine 0: machine 2 gives −2.53, machine 1 gives 1.45. From then on machine 1 has the largest estimate, and the agent takes it; after ten pulls its estimate is 1.34, close to its mean of 1.5. Without the random pull 6, the agent might never have tried machine 1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_estimates():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The estimate of every machine

        `estimates(totals, counts)` gives back a list `q` with the estimate of every machine: step 5 of the recipe, for all machines at once. `totals[i]` is the total of the rewards of machine `i` so far, `counts[i]` its number of pulls. Replace the two `...`: what a machine that was never pulled gets, and the sample average of the others. The check calls `estimates([3.0, 0.0, -1.5], [2, 0, 3])`, which should give `[1.5, 0.0, -0.5]`, and five other cases, among them the totals and counts of an agent after 200 pulls.
        """),
        L.task("Make `estimates` give back the sample average of every machine, and `0.0` for a machine that was never pulled."),
    ])
    return


@app.cell(hide_code=True)
def exercise_estimates():
    box_estimates = L.code_box("estimates", lines=8)
    box_estimates
    return (box_estimates,)


@app.cell(hide_code=True)
def check_estimates(box_estimates):
    result_estimates = L.check("estimates", box_estimates.value)
    result_estimates
    return (result_estimates,)


@app.cell(hide_code=True)
def hints_estimates():
    L.hints(
        "estimates",
        [
            "Steps 1 and 5 of the recipe: a machine with a count of 0 gets `0.0`, any other machine its total divided by its count. `q.append` puts a number at the end of the list `q`.",
            "For machine 0 of the example, `totals[0]` is `3.0` and `counts[0]` is `2`, so its estimate is 3.0 / 2 = 1.5. In the loop, `i` is the number of the machine.",
            "`            q.append(___)`",
        ],
        why="`q.append(totals[i] / counts[i])` under `else`, and `q.append(0.0)` under `if`: a machine that was never pulled has no rewards to average, so its estimate starts at 0, as every $Q$ did in Q-learning.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Let three agents play

    The button lets three agents play the same 300 bandits, 1,000 pulls each: greedy, $\varepsilon$ = 0.01 and $\varepsilon$ = 0.1. All three use your `estimates` once it passes its check, and the course's until then. The run of one agent goes like this; `choose` is the ε-greedy of Q-learning, without a state, since a bandit has only one:

    ```python
    for t in range(1000):
        for b in range(300):   # every bandit
            a = choose(q[b], epsilon, rng)
            r = means[b][a] + rng.normal()
            totals[b][a] = totals[b][a] + r
            counts[b][a] = counts[b][a] + 1
            q[b] = estimates(totals[b], counts[b])
    ```

    The chart shows the mean reward of $\varepsilon$ = 0.1 over the 300 bandits at every pull, thin, and its mean over the last 20 pulls, thick; dashed, the mean over the last 20 pulls of greedy and of $\varepsilon$ = 0.01. Beside it you see what greedy and $\varepsilon$ = 0.1 know of the first of the 300 bandits at that moment: its means as short lines, their estimates in <span class="c-param">orange</span>, and how often they pulled each machine. It takes a few seconds; then you can go back through the run.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    run_button = mo.ui.run_button(label="Let the three agents play with my estimates")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def demo_run(box_estimates, result_estimates, run_button):
    if not run_button.value:
        _out = mo.md("*Press the button to let the three agents play.*")
    else:
        _yours = C.mine(box_estimates.value, result_estimates.state == "correct")
        _w = C.live_words(L.lang)
        _now = {}   # the live chart, what the agents know of the first bandit after the last pull, and its pull
        _raw = {0.0: [], 0.01: []}   # the mean reward of greedy and of 0.01 at every pull, for the dashed lines

        def _progress(_t, _r, _seen):
            if "seen" not in _now:   # from the first pull on, what the agents know beside the chart
                _now["chart"].watch(lambda: C.now_picture(C.watched_means(), _now["seen"], _now["t"], L.lang))
            _now["seen"], _now["t"] = _seen, _t
            _raw[0.0].append(_r[0.0])
            _raw[0.01].append(_r[0.01])
            if _t % 25 == 0 or _t == C.STEPS:
                _now["chart"].reference(C.smoothed(_raw[0.0]), "greedy", role="ink")
                _now["chart"].reference(C.smoothed(_raw[0.01]), "ε = 0.01", role="ink")
            _now["chart"].add(_t, _r[0.1], f"**Pull {_t} of {C.STEPS}** · the three agents on 300 bandits")

        _now["chart"] = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"], "alt": _w["alt"]},
                       C.STEPS, role="ink", low=-0.4, high=2.0, window=20,
                       key=[("ink", False, "ε = 0.1: thin at every pull, thick the mean over the last 20 pulls")])
        _now["chart"].line("**Playing** …")
        _run = C.demo(_yours or C.estimates, show=_progress)
        if _run == "estimates":
            _out = _now["chart"].stopped("*Your `estimates` gave an error, or no list of ten numbers, during the run. Does it still pass its check?*")
        elif _run == "slow":
            _out = _now["chart"].stopped("*The run took more than two minutes. Does your `estimates` do more work than it needs?*")
        else:
            _last = {_e: sum(_run["curve"][_e][-100:]) / 100 for _e in C.EPSILONS}
            _who = "your `estimates`" if _yours else "the course's `estimates`, since yours does not pass its check yet"
            _out = mo.vstack([
                L.explore("rewind", "pull", lambda _v: (C.rewind_frame(_run, int(_v), L.lang), C.rewind_text(_run, int(_v), L.lang)),
                          start=C.STEPS, values=C.LOOKS, role="",
                          task="Go back to pull 1, then forward again: when does greedy stop trying machines of the first bandit, and when does ε = 0.1 find its best one?"),
                mo.md(f"Played with {_who}. Over the last 100 pulls, greedy got {C.number(_last[0.0], 2, L.lang)} per pull, ε = 0.01 {C.number(_last[0.01], 2, L.lang)} and ε = 0.1 {C.number(_last[0.1], 2, L.lang)}."),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    The course's 2,000 bandits tell the same story. Over the first 20 pulls, greedy gets 0.74 per pull and $\varepsilon$ = 0.1 gets 0.69: exploring costs at first. Around pull 100, $\varepsilon$ = 0.1 is ahead, 1.18 against 1.03. Over the last 100 pulls, $\varepsilon$ = 0.1 gets 1.36, $\varepsilon$ = 0.01 gets 1.31 and greedy 1.04. An agent that knew the means and always pulled the best machine would get 1.53: the mean of the best machine over the 2,000 bandits. Greedy pulls machine 0 at the end in 27.5% of the bandits, the first machine it tried, while machine 0 is the best in only 9.1%.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_best_share():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. How often the best machine

        The mean reward is one way to judge an agent; another is how often it pulls the best machine. `best_share(actions, best)` gives back, for every pull, the share of the runs in which the agent pulled the best machine of that run. `actions` has a row for every run and a column for every pull, with the machine pulled; `best` is a column with the best machine of every run, so `actions == best` compares every pull of a run with the best machine of that run, as in [Broadcasting](/course/open/np-broadcasting?lang=en). Replace the two `...`: `hits`, `True` where a pull took the best machine, and the share of `True` in every column.

        The check calls `best_share(small_actions, small_best)`, with three runs of four pulls, which should give the shares 0, 1/3, 2/3 and 1, then the course's runs of 2,000 bandits:

        ```python
        small_actions = np.array([[0, 2, 2, 2],
                                  [1, 1, 0, 0],
                                  [2, 2, 2, 1]])
        small_best = np.array([[2], [0], [1]])
        ```
        """),
        L.task("Make `best_share` give back, for every pull, the share of runs whose machine was the best one."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best_share():
    box_best_share = L.code_box("best_share", lines=4)
    box_best_share
    return (box_best_share,)


@app.cell(hide_code=True)
def check_best_share(box_best_share):
    result_best_share = L.check("best_share", box_best_share.value)
    result_best_share
    return (result_best_share,)


@app.cell(hide_code=True)
def hints_best_share():
    L.hints(
        "best_share",
        [
            "`hits = actions == best` holds `True` for every pull that took the best machine of its run. A share of `True` is a mean, since `True` counts as 1 and `False` as 0: `hits` has a method for it, with an axis.",
            "In [Sums and means along an axis](/course/open/np-axis?lang=en), `axis=0` went down the rows: one result per column. Here a column is a pull, and going down the rows goes over the runs.",
            "`    return hits.___`",
        ],
        why="`return hits.mean(axis=0)`: the mean down each column, over the runs, gives one share per pull. `np.mean(actions == best, axis=0)` gives the same.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_long():
    mo.md(r"""
    ### Exercise 3. A longer run

    After 1,000 pulls, $\varepsilon$ = 0.1 is ahead of $\varepsilon$ = 0.01: over the last 100 pulls it pulls the best machine in 79.5% of the bandits against 59.9%, and it gets 1.36 per pull against 1.31. The course let both play the same 2,000 bandits for 10,000 pulls.
    """)
    return


@app.cell(hide_code=True)
def choice_long_cell():
    choice_long = L.choice(
        "long-run",
        question="After 10,000 pulls, which agent gets more reward per pull over the last 1,000 pulls?",
        options={
            "same01": "ε = 0.1, as over the first 1,000 pulls",
            "right": "ε = 0.01: it pulls a random machine less often",
            "equal": "Both the same: by then they both know the best machine of every bandit, so they get the same per pull",
        },
        correct="right",
        why={
            "right": "Over the last 1,000 of 10,000 pulls, ε = 0.01 gets 1.50 per pull and ε = 0.1 gets 1.38. At the end ε = 0.1 knows the best machine more often, in 95.3% of the bandits against 87.8%, but in 1 of 10 pulls it takes a random machine, which pays about 0 on average: it can never pull the best machine in more than 91% of its pulls. ε = 0.01 takes a random machine in only 1 of 100 pulls, and where its largest estimate is not the best machine, that machine pays only 0.15 less on average.",
            "same01": "Over the first 1,000 pulls, ε = 0.1 finds the best machine sooner. But once it has, what does it still do in 1 of every 10 pulls?",
            "equal": "Even where both know the best machine, ε = 0.1 still takes a random machine in 1 of 10 pulls: what does that cost per pull?",
        },
    )
    choice_long
    return (choice_long,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it: another ε

    The slider runs the agent on the course's 2,000 bandits with another $\varepsilon$, from 0 to 0.5. The chart shows its mean reward at every pull, averaged over 20 pulls, with greedy dashed and the mean of the best machine as a thin line.
    """)
    return


@app.cell(hide_code=True)
def try_cell():
    _out = L.explore("try-epsilon", "ε", lambda _v: C.try_frame(float(_v), L.lang), start=0.1, values=C.TRY_EPSILONS,
                     role="hyper", task="Go from 0 to 0.5: which ε gets the most reward over the last 100 pulls, and why does an ε larger than that get less?")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en), the agent explored with ε-greedy in a grid of states, where a step also changes where it is. A bandit has one state, so the exploration-exploitation trade-off is all there is: your `estimates` gives the sample average of every machine, and your `best_share` shows how often each agent finds the best one. Q-learning moved $Q$ a step $\eta$ toward every new target instead of keeping totals and counts. In the next lesson, *Averaging as you go: drifting machines and optimistic starts*, you see that a sample average can be kept up to date in the same way, one reward at a time.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_long, result_best_share, result_estimates):
    L.wrap_up(result_estimates, result_best_share, choice_long)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
