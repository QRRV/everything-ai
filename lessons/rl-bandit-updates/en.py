# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-bandit-updates. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the bandits, the runs and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rl-bandit-updates", "en")
    C = L.checks   # the drifting bandits, the course's agents and runs, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Which machine pays most? The multi-armed bandit](/course/open/rl-bandits?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_average():
    quick_average = L.choice(
        "quick-average",
        question="A machine gave the rewards 2, 0 and 4. What is its sample average?",
        options={
            "last": "4, the last reward",
            "right": "2: the total 6 divided by the 3 pulls",
            "total": "6, the total of the three rewards, since the estimate keeps every reward",
        },
        correct="right",
        why={
            "right": "The sample average is the total of the rewards divided by the number of pulls: (2 + 0 + 4) / 3 = 2.",
            "last": "The sample average uses every reward of the machine, not only the last one: what do you get when you divide the total by the number of pulls? See [Which machine pays most? The multi-armed bandit](/course/open/rl-bandits?lang=en).",
            "total": "The total grows with every pull, and the estimate divides it by something: by what? See [Which machine pays most? The multi-armed bandit](/course/open/rl-bandits?lang=en).",
        },
    )
    quick_average
    return (quick_average,)


@app.cell(hide_code=True)
def quick_check_greedy():
    quick_greedy = L.choice(
        "quick-greedy",
        question="A greedy agent pulled machine 0 of three once and got 0.8, so its estimates are 0.8, 0 and 0. Which machine does it pull next?",
        options={
            "right": "Machine 0 again: it has the largest estimate",
            "other": "Machine 1 or 2, to find out what they pay",
            "random": "A random one of the three, since it has pulled each machine at most once and knows too little yet",
        },
        correct="right",
        why={
            "right": "Greedy always takes the largest estimate: 0.8 against 0 and 0, so machine 0 again. It tries machine 1 or 2 only once machine 0 drops below 0.",
            "other": "A greedy agent never pulls a machine just to find out what it pays: which machine has the largest estimate? See [Which machine pays most? The multi-armed bandit](/course/open/rl-bandits?lang=en).",
            "random": "Only ε-greedy takes a random machine now and then, and greedy is ε-greedy with $\\varepsilon$ = 0: which machine does it take? See [Which machine pays most? The multi-armed bandit](/course/open/rl-bandits?lang=en).",
        },
    )
    quick_greedy
    return (quick_greedy,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The news site with four headlines has a problem the bandits of [Which machine pays most? The multi-armed bandit](/course/open/rl-bandits?lang=en) did not have: a headline that draws clicks in the morning draws fewer by the evening, once the news has moved on. A sample average counts a click of last week as much as one of a minute ago, so it keeps showing what used to work. Prices change, tastes change, and the people an agent serves change; an agent that learns in such a world has to let old rewards count less.

    **An everyday example.** Your favorite café gets a new chef, and the last three meals were poor. Does the average of all your visits over the years tell you whether to go back, or do the last few meals tell you more?

    There is a second, smaller thing to learn here: a sample average does not need totals at all. It can be kept one reward at a time, with the same update that Q-learning used.
    """)
    return


@app.cell(hide_code=True)
def predict_drift():
    guess_drift = L.guess(
        "drift",
        question="In the bandits of this lesson, every mean takes a small random step after every pull: the means drift. The course lets two agents with ε-greedy and $\\varepsilon$ = 0.1 play 2,000 such bandits for 10,000 pulls each. One keeps the sample average; the other moves its estimate a tenth of the way toward each new reward. Over the last 1,000 pulls, how often does each one pull the best machine of that moment?",
        options={
            "same": "Both in about 80 of 100 pulls: they get the same rewards",
            "right": "The one that moves a tenth in about 80 of 100, the sample average in about 60",
            "average": "The sample average in about 80 of 100, since it uses every reward and so has the least noise in its estimates",
        },
        reveal="**The one that moves a tenth in about 80 of 100, the sample average in about 60.** Over the last 1,000 of the 10,000 pulls, the agent that moves a tenth of the way toward each reward pulls the best machine in 80.5% of its pulls, the sample average in 59.9%. *The idea* shows why, and the demo plays the run again on 300 bandits.",
    )
    guess_drift
    return (guess_drift,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        In [Which machine pays most? The multi-armed bandit](/course/open/rl-bandits?lang=en), the agent kept a total and a count for every machine and divided one by the other. The same sample average can be kept with one number per machine. Say a machine has given $n - 1$ rewards so far, with the estimate $Q$, and now gives a reward $r$, its $n$-th. The new sample average is the old one moved a part $1/n$ of the way toward $r$:
        """),
        L.formula(r"[[param:Q]] \leftarrow [[param:Q]] + [[hyper:\eta]] \left( [[data:r]] - [[param:Q]] \right)"),
        mo.md(r"""
        Here $r - Q$ is how far the reward lies from the estimate, and $\eta$ is the **learning rate**: how big a part of that distance the estimate moves. It is the update of [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en), with the reward as target. With $\eta = 1/n$ it gives exactly the sample average. For the rewards 2, 0 and 4 of the quick check, the estimate goes from 0 to 0 + 1 · (2 − 0) = 2, then to 2 + ½ · (0 − 2) = 1, then to 1 + ⅓ · (4 − 1) = 2: the total 6 divided by 3, without a total. This is an **incremental update**: no totals and no list of rewards, one number per machine that moves with every reward.

        With a constant $\eta$, such as 0.1, every step is the same part of the distance. Each update keeps $1 - \eta$ of the old estimate and adds $\eta$ of the new reward, so at every update the part that every older reward has in the estimate is multiplied by $1 - \eta$: ten updates later, a reward counts only $0.9^{10} \approx 0.35$ of what it counted when it came in. Recent rewards count more. Exercise 2 works out how much each reward counts.

        The picture shows a machine whose mean jumps from 0 to 2 after 100 pulls, with both estimates.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        After 200 pulls, the sample average is at 1.05: it still averages the 100 rewards around 0 with the 100 around 2, and every new reward moves it only 1/200 of the way. The estimate with $\eta$ = 0.1 is at 2.11. About 20 pulls after the jump it is near 2, and from then on it moves around 2 with every reward: it never settles as close as a sample average would on a machine that stays the same.

        A problem whose rewards change over time is a **nonstationary problem**. In the bandits of this lesson, every mean of every machine takes a random step with a standard deviation of 0.01 after every pull. After 10,000 pulls a mean typically lies about 1 away from where it started, as far as the means of a bandit lie apart at the start.

        Every estimate so far started at 0. With $\eta = 1/n$ the start value is gone after the first reward: the first update has $\eta$ = 1 and puts the estimate on that reward. With $\eta$ = 0.1 the start value never quite goes away: after $n$ updates it still counts $(1 - \eta)^n$. That can be put to use. Start every estimate at 5, far above what any machine pays. The first reward of a machine disappoints and its estimate drops, so even a greedy agent goes on to a machine it has not tried yet, still at 5. Before it settles on one, it has tried them all. These are **optimistic initial values**: exploration that comes from the start values, not from random pulls.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **incremental update** | | moving an estimate a part $\eta$ of the way toward each new reward: one number per machine, no totals | `update` |
        | **nonstationary problem** | | a problem whose rewards change over time, such as bandits whose means drift | |
        | **optimistic initial values** | | estimates that start above what any action pays, so that even a greedy agent tries every action | `start` |
        | $\eta$ | eta | the learning rate: $1/n$ gives the sample average, a constant one lets recent rewards count more | `eta` |
        | $Q$ | Q | the estimate of one machine | `q` |
        | $r$ | r | the reward of the newest pull | `reward` |
        | $n$ | n | the number of rewards of the machine so far, the newest one included | `n` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    An agent with incremental updates on a bandit:

    1. Give every machine an estimate, 0 or a start value above every reward, and a count of 0.
    2. Choose a machine, ε-greedy: with probability $\varepsilon$ a random one, else the one with the largest estimate.
    3. Pull it: the bandit gives a reward $r$.
    4. Add 1 to the count $n$ of that machine, and take its learning rate: $\eta = 1/n$ for the sample average, a constant one such as 0.1 for a problem that changes.
    5. Move its estimate: $Q \leftarrow Q + \eta \, (r - Q)$.
    6. Go on at step 2, until the last pull.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    One machine gives the rewards 2.0, 0.0, 4.0, 1.0 and 3.0. The cell takes them through steps 4 and 5 twice: with $\eta = 1/n$ and with the constant $\eta$ = 0.5. Each line shows the count, the reward and both estimates. Run the cell.
    """)
    return


@app.cell
def example_updates():
    # the rewards of one machine, in the order they came
    _rewards = [2.0, 0.0, 4.0, 1.0, 3.0]

    # 1. both estimates start at 0, and the count at 0
    _q_average = 0.0
    _q_constant = 0.0
    _n = 0
    for _r in _rewards:
        # 4. one reward more; its learning rate is 1/n, or the constant 0.5
        _n = _n + 1
        # 5. each estimate moves a part of the way toward the reward
        _q_average = _q_average + (1 / _n) * (_r - _q_average)
        _q_constant = _q_constant + 0.5 * (_r - _q_constant)
        print(_n, _r, _q_average, _q_constant)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With $\eta = 1/n$, the estimate is the sample average after every reward: 2.0, then (2 + 0) / 2 = 1.0, then 6 / 3 = 2.0, then 7 / 4 = 1.75, and after five rewards 10 / 5 = 2.0, without a total. With $\eta$ = 0.5, every reward moves the estimate halfway: 1.0, 0.5, 2.25, 1.625 and 2.3125. It ends above the sample average: the last reward, 3.0, counts for half of it, and the 4.0 counts twice as much as the 0.0 before it.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_update():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. One reward at a time

        `update(q, reward, eta)` gives back the new estimate: the estimate `q` moved a part `eta` of the way toward `reward`, step 5 of the recipe. Replace the `...`. The check calls `update(1.0, 3.0, 0.5)`, which should give `2.0`, and six other cases, with learning rates of $1/n$ and constant ones.
        """),
        L.task("Make `update` give back the estimate `q` moved a part `eta` of the way toward `reward`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_update():
    box_update = L.code_box("update", lines=4)
    box_update
    return (box_update,)


@app.cell(hide_code=True)
def check_update(box_update):
    result_update = L.check("update", box_update.value)
    result_update
    return (result_update,)


@app.cell(hide_code=True)
def hints_update():
    L.hints(
        "update",
        [
            "Step 5 of the recipe: the new estimate is the old one, `q`, plus `eta` times how far `reward` lies from `q`.",
            "In `update(1.0, 3.0, 0.5)`, the reward lies 3.0 − 1.0 = 2.0 above the estimate. Half of that is 1.0, so the new estimate is 1.0 + 1.0 = 2.0.",
            "`    return q + eta * ___`",
        ],
        why="`return q + eta * (reward - q)`: the distance from the estimate to the reward, a part `eta` of it, added to the estimate. `(1 - eta) * q + eta * reward` gives the same.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Let two agents play on drifting bandits

    The button lets two agents with $\varepsilon$ = 0.1 play the same 300 drifting bandits, 10,000 pulls each: one with $\eta = 1/n$, the sample average, and one with $\eta$ = 0.1. Both use your `update` once it passes its check, and the course's until then. The run of one agent goes like this; `choose` is the ε-greedy of the multi-armed bandit:

    ```python
    for t in range(10000):
        for b in range(300):   # every bandit
            a = choose(q[b], epsilon, rng)
            r = means[b][a] + rng.normal()
            counts[b][a] = counts[b][a] + 1
            q[b][a] = update(q[b][a], r, 1 / counts[b][a])   # the other agent: 0.1
        means = means + rng.normal(0, 0.01, (300, 10))   # every mean drifts a little
    ```

    The chart shows how often the agent with $\eta$ = 0.1 pulls the best machine of that moment: the share of the 300 bandits over every 10 pulls, thin, and its mean over the last 100 pulls, thick; dashed, the same mean for $\eta = 1/n$. Beside it you see what both agents know of one of the 300 bandits: the means of that moment as short lines, the estimates in <span class="c-param">orange</span>, and how often each machine was pulled. It takes about ten seconds; then you can go back through the run.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    run_button = mo.ui.run_button(label="Let the two agents play with my update")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def demo_run(box_update, result_update, run_button):
    if not run_button.value:
        _out = mo.md("*Press the button to let the two agents play.*")
    else:
        _yours = C.mine(box_update.value, result_update.state == "correct")
        _w = C.live_words(L.lang)
        _now = {}   # the live chart, what the agents know of one bandit after the last pull, and its pull
        _raw = {"average": [], "constant": []}   # the share of best machines of each agent at every pull

        def _progress(_t, _shares, _seen):
            if "seen" not in _now:   # from the first pull on, what the agents know beside the chart
                _now["chart"].watch(lambda: C.now_picture(_now["seen"], _now["t"], L.lang))
            _now["seen"], _now["t"] = _seen, _t
            _raw["average"].append(_shares["average"])
            _raw["constant"].append(_shares["constant"])
            if _t % 500 == 0 or _t == C.DEMO_STEPS:
                _now["chart"].reference(C.smoothed(_raw["average"]), "1/n", role="ink")
            if _t % 10 == 0:
                _now["chart"].add(_t / 1000, C.mean_of(_raw["constant"], _t - 9, _t),
                                  f"**Pull {C.number(_t, 0, L.lang)} of {C.number(C.DEMO_STEPS, 0, L.lang)}** · two agents on 300 drifting bandits")

        _now["chart"] = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"], "alt": _w["alt"]},
                       C.DEMO_STEPS / 1000, role="ink", low=0.0, high=1.0, window=10,
                       key=[("ink", False, "η = 0.1: thin over every 10 pulls, thick the mean over the last 100 pulls")])
        _now["chart"].line("**Playing** …")
        _run = C.demo(_yours or C.update, show=_progress)
        if _run == "update":
            _out = _now["chart"].stopped("*Your `update` gave an error, or no number, during the run. Does it still pass its check?*")
        elif _run == "slow":
            _out = _now["chart"].stopped("*The run took more than two minutes. Does your `update` do more work than it needs?*")
        else:
            _last = {_a: C.mean_of(_run["curve"][_a], C.DEMO_STEPS - 999, C.DEMO_STEPS) for _a in C.AGENTS}
            _who = "your `update`" if _yours else "the course's `update`, since yours does not pass its check yet"
            _out = mo.vstack([
                L.explore("rewind", "pull", lambda _v: (C.rewind_frame(_run, int(_v), L.lang), C.rewind_text(_run, int(_v), L.lang)),
                          start=C.DEMO_STEPS, values=C.LOOKS, role="",
                          task="Go back to pull 1, then forward again: from which pull on does η = 0.1 pull the best machine more often, and how far do the estimates of 1/n lie from the means at the end?"),
                mo.md(f"Played with {_who}. Over the last 1,000 pulls, the agent with η = 0.1 pulled the best machine in {C.percent(_last['constant'], L.lang)} of its pulls, the one with 1/n in {C.percent(_last['average'], L.lang)}."),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    The course's 2,000 bandits tell the same story. Over the first 1,000 pulls the sample average is ahead: it pulls the best machine in 67.7% of its pulls against 61.7%, and gets 1.32 per pull against 1.27. While the means have hardly moved, more rewards make a better estimate. From about pull 1,000 on, $\eta$ = 0.1 is ahead, and over the last 1,000 of the 10,000 pulls the sample average pulls the best machine in only 59.9% of its pulls: a machine it has pulled thousands of times moves less than a thousandth of the way toward a new reward. Both agents get more per pull at the end than over the first 1,000 pulls, 1.93 for $\eta$ = 0.1 and 1.77 for the sample average: as the means drift, they spread out, and the best machine of a bandit pays more.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_reward_shares():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. How much each reward counts

        With a constant $\eta$, the estimate after $n$ rewards is made of the rewards themselves, each with a share, and what is left of the start value. The newest reward counts $\eta$. Every update after a reward multiplies its share by $1 - \eta$.

        `reward_shares(eta, n)` gives back an array with the share of each of `n` rewards in the estimate, the first reward first. `i = np.arange(1, n + 1)` holds the numbers of the rewards, 1 to `n`. Replace the `...`. The check calls `reward_shares(0.5, 3)`, which should give `[0.125, 0.25, 0.5]`, and five other cases.
        """),
        L.task("Make `reward_shares` give back the share of every reward in the estimate after `n` updates with the learning rate `eta`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_reward_shares():
    box_reward_shares = L.code_box("reward_shares", lines=5)
    box_reward_shares
    return (box_reward_shares,)


@app.cell(hide_code=True)
def check_reward_shares(box_reward_shares):
    result_reward_shares = L.check("reward_shares", box_reward_shares.value)
    result_reward_shares
    return (result_reward_shares,)


@app.cell(hide_code=True)
def hints_reward_shares():
    L.hints(
        "reward_shares",
        [
            "Reward `i` gets `eta` when it comes in, and every update after it multiplies its share by 1 − η. `**` raises a number to a power, and works on the whole array `i` at once.",
            "How many updates follow reward `i` when there are `n` rewards? For `reward_shares(0.5, 3)`: the last reward, number 3, has none after it and counts 0.5. The first has two after it: 0.5 × 0.5 × 0.5 = 0.125.",
            "`    return eta * ___`",
        ],
        why="`return eta * (1 - eta) ** (n - i)`: `eta` for the reward itself, and a factor 1 − η for each of the `n - i` updates after it, for all rewards at once.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_spike():
    mo.md(r"""
    ### Exercise 3. A greedy agent that explores

    The course let a greedy agent with $\eta$ = 0.1 play 2,000 bandits whose means stay the same, for 1,000 pulls, with every estimate starting at 5. Beside it played an agent that starts at 0 and explores with ε-greedy, $\varepsilon$ = 0.1, also with $\eta$ = 0.1. The optimistic agent gets less at first: 0.34 per pull over the first 100 pulls, against 0.96. Over the last 100 pulls it pulls the best machine in 86.4% of the bandits against 76.6%, and over all 1,000 pulls it gets 1.29 per pull against 1.24. Around pull 11 something odd happens: the share of bandits in which it pulls the best machine is 10.0% on pull 10, jumps to 41.5% on pull 11, and falls back to 26.1% on pull 12.
    """)
    return


@app.cell(hide_code=True)
def choice_spike_cell():
    choice_spike = L.choice(
        "spike",
        question="Why does the optimistic agent pull the best machine on pull 11 in 41.5% of the bandits, and on pull 12 in only 26.1%?",
        options={
            "lucky": "By chance: the rewards of pull 11 happen to be large",
            "right": "After ten pulls it has tried every machine once, the best one has the largest estimate more often than any other, and one more pull drops it below another",
            "learned": "On pull 11 it has found out which machine is best, and on pull 12 it starts to explore again with random pulls, the way an agent with ε-greedy would do after it has found the best machine",
        },
        correct="right",
        why={
            "right": "Every estimate starts at 5. A first reward $r$ moves a machine to 5 + 0.1 · ($r$ − 5) = 4.5 + 0.1$r$, below the machines not yet tried, so in the first ten pulls the agent tries each machine once. After pull 10 the largest estimate is that of the largest first reward, which comes from the best machine more often than from any other: in 41.5% of the bandits. Pulling that machine again moves its estimate down to about 4.35 on average, below the largest of the nine others, so on pull 12 the agent mostly goes on to another machine. Only after many pulls of every machine does the start value stop counting.",
            "lucky": "The share is over 2,000 bandits: could chance alone lift it from 10.0% to 41.5% on one pull, and drop it again on the next?",
            "learned": "The agent is greedy, so it never pulls a random machine: what happens to the estimate of a machine each time it is pulled, while every estimate is still far above every reward?",
        },
    )
    choice_spike
    return (choice_spike,)


@app.cell(hide_code=True)
def text_try():
    mo.md(r"""
    ### Try it: another start value

    The slider runs the greedy agent with $\eta$ = 0.1 on the course's 2,000 bandits with another start value, from 0 to 10. The chart shows how often it pulls the best machine, averaged over 10 pulls, with the agent that starts at 0 and explores with $\varepsilon$ = 0.1 dashed. A start does not have to lie above every reward: above most first rewards is enough to make the agent try each machine. The higher the start, the longer the agent keeps going back to machines that have already paid little.
    """)
    return


@app.cell(hide_code=True)
def try_cell():
    _out = L.explore("try-start", "start", lambda _v: C.try_frame(float(_v), L.lang), start=5.0, values=C.STARTS,
                     role="hyper", task="Go from 0 to 10: which start value gets the most reward over all 1,000 pulls, and from which start on does the agent find the best machine in more than 85% of the bandits at the end?")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Which machine pays most? The multi-armed bandit](/course/open/rl-bandits?lang=en), the agent kept totals and counts. Your `update` keeps the same sample average with one number per machine, and with a constant learning rate it keeps up with bandits whose means drift: as your `reward_shares` shows, old rewards fade. It is the update of [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en), with the reward as target, and its start values can make even a greedy agent explore. In the next lesson, *Project: your own bandit agent*, you build an agent in your own project folder and compare a greedy, an ε-greedy and an optimistic version on drifting bandits.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_spike, result_reward_shares, result_update):
    L.wrap_up(result_update, result_reward_shares, choice_spike)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
