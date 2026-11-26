# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-monte-carlo-prediction. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the room, the course's walks, returns and estimates, and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rl-monte-carlo-prediction", "en")
    C = L.checks   # the room, the course's walks, returns and estimates, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [What is a policy worth? Policy evaluation](/course/open/rl-policy-evaluation?lang=en) and [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_value():
    quick_value = L.choice(
        "quick-value",
        question="On ROOM, the robot's grid, a tile next to the charger has the value −24 under the random policy, with a reward of −1 per step and $\\gamma$ = 1. What does that −24 say?",
        options={
            "most": "Its walks from that tile take at most 24 steps",
            "right": "Its walks from that tile take 24 steps on average",
            "every": "Every walk from that tile takes exactly 24 steps, since the policy stays the same",
            "count": "24 of every 100 walks from that tile end on the charger",
        },
        correct="right",
        why={
            "right": "With −1 per step and $\\gamma$ = 1, the return of a walk is minus its number of steps, and the value is the return to expect: the average. One walk can be much shorter or longer. This lesson finds the same −24 by averaging walks.",
            "most": "The random robot can wander for a long time before it reaches the charger: is the value a limit on the steps, or the return to expect? See [What is a policy worth? Policy evaluation](/course/open/rl-policy-evaluation?lang=en).",
            "every": "The random policy picks every step at random, so no two walks need be alike: is −24 the steps of one walk, or of all walks together? See [What is a policy worth? Policy evaluation](/course/open/rl-policy-evaluation?lang=en).",
            "count": "Every walk of the random robot reaches the charger sooner or later: what does a reward of −1 per step count? See [What is a policy worth? Policy evaluation](/course/open/rl-policy-evaluation?lang=en).",
        },
    )
    quick_value
    return (quick_value,)


@app.cell(hide_code=True)
def quick_check_step():
    quick_step = L.choice(
        "quick-step",
        question="An agent that does Q-learning in GRID never sees the table `T`. What does it know after one step?",
        options={
            "table": "Every outcome of its action, each with its probability from `T`",
            "reward": "Only its reward, not where it landed",
            "right": "Its state, its action, its reward and where it landed",
            "values": "The value of every state, worked out with value iteration before it started",
        },
        correct="right",
        why={
            "right": "That is all the agent sees of the environment: one outcome, the one that happened. Q-learning learns from such steps without `T`, and the method of this lesson learns from whole episodes of them.",
            "table": "The agent never sees `T`: does it get the probabilities of all outcomes, or only the outcome that happened? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
            "reward": "The target of Q-learning uses the largest $Q$ where the agent landed: how could it find that without knowing where it landed? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
            "values": "Value iteration needs `T`, and the agent does not have it: where do its $Q$-values come from? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
        },
    )
    quick_step
    return (quick_step,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Policy evaluation worked out what a policy is worth from the table `T`: every probability and every reward. Often nobody has that table. A robot vacuum in a house it does not know, a card game where you cannot see the cards, a program that plays against people: none of them can sweep over a table. What they can do is play, again and again, and count what each game brought. Monte Carlo prediction turns those whole games into values, with nothing but averages, and the next lessons build on it to make the policy better and to learn faster.

    **An everyday example.** How long does your commute take on average? Nobody works that out from the timings of every traffic light. You time the trip for a few weeks and take the average. One trip says little, since a traffic jam can double it; after many trips the average hardly moves any more.
    """)
    return


@app.cell(hide_code=True)
def predict_walks():
    guess_walks = L.guess(
        "walks",
        question="From its corner, the random robot of ROOM needs 36 steps on average to reach the charger, as policy evaluation worked out. Without the table, you could also let it walk from the corner many times and average the numbers of steps. How many walks does it take before that average is typically off by only about 1 step?",
        options={
            "ten": "About 10: the walks of one robot in one room are much alike",
            "thirty": "About 36: one walk for every step that the robot needs on average",
            "right": "About 1,000",
            "hundred": "About 100",
        },
        reveal="**About 1,000 walks.** The walks differ a lot: about 2 in 100 take the shortest way of 4 steps, about 1 in 100 takes more than 150, and their standard deviation is about 32 steps. The average of $n$ walks spreads less, about 32/√n: about 10 steps for 10 walks, about 3 for 100, and about 1 for 1,000. *The idea* shows where that √n comes from.",
    )
    guess_walks
    return (guess_walks,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Monte Carlo prediction** estimates the values $V_\pi$ of a policy without the table `T`, from episodes the agent plays with $\pi$, such as the robot's walks. The value of a state is the return to expect from it. Every time an episode is in that state, the rewards after it give one return that really happened; the average of many such returns comes close to the return to expect. Estimating a number by averaging over many random draws is the **Monte Carlo method**, named after the casino of Monte Carlo. First the returns of an episode: with $r_{t+1}$ the reward right after step $t$, as in [Learning from rewards](/course/open/rl-what-is-rl?lang=en),
        """),
        L.formula(r"[[loss:G_t]] = [[data:r_{t+1}]] + [[hyper:\gamma]] \, [[loss:G_{t+1}]]"),
        mo.md(r"""
        The return from step $t$ is the reward right after it plus $\gamma$ times the return from the next step, and after the last step the return is 0. So the returns of an episode are quickest to work out backward, from the end: each step needs one addition and one multiplication. In the robot's walks, with −1 per step and $\gamma$ = 1, the return from a step is minus the number of steps still to come. Then the estimate of a state $s$, from the returns $G^{(1)}, \ldots, G^{(n)}$ that followed its visits:
        """),
        L.formula(r"[[pred:V_\pi(s)]] \approx \frac{[[loss:G^{(1)}]] + [[loss:G^{(2)}]] + \dots + [[loss:G^{(n)}]]}{n}"),
        mo.md(r"""
        An episode can come back to a state: the robot in a corner bumps into the edge and stays where it is. **First-visit Monte Carlo** takes from each episode only the return after the first visit of a state; **every-visit Monte Carlo** takes the return after every visit. Over many episodes both come close to $V_\pi$. With first visit, every return comes from another episode, so the returns do not depend on each other, and their average is a plain average of random draws.

        How many episodes does that take? The returns spread a lot: from the corner the robot needs 36 steps on average, with a standard deviation of about 32. An average of $n$ returns spreads $\sqrt{n}$ times less than one return: four times as many episodes halve the spread. Why $\sqrt{n}$? The returns of different episodes are independent, and the variances of independent numbers add up: $n$ returns together have $n$ times the variance of one return. The average divides them by $n$, which divides the variance by $n^2$, so the average has $1/n$ of the variance of one return, and a standard deviation $\sqrt{n}$ times smaller.

        Each step gets the whole return after it, whatever it did itself. If the robot comes close to the charger and then wanders off, the tiles before the detour pay for the detour too. Which earlier steps deserve the credit or the blame for a reward that comes later is the problem of **credit assignment**, what [Learning from rewards](/course/open/rl-what-is-rl?lang=en) called the hard part of reinforcement learning. Monte Carlo solves it only on average: a tile from which detours are likely gets a lower average than a tile from which they are not, but that shows only after many episodes.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **Monte Carlo prediction** | | estimating $V_\pi$ from episodes played with $\pi$, by averaging the returns that followed each state | `first_visit` |
        | **Monte Carlo method** | | estimating a number by averaging over many random draws | |
        | $G_t$ | G t | the return from step $t$ to the end of the episode | `G[t]` |
        | **first-visit Monte Carlo** | | per episode, only the return after the first visit of a state counts | `first_visit` |
        | **every-visit Monte Carlo** | | the return after every visit of a state counts | |
        | **credit assignment** | | working out which earlier steps deserve the credit or the blame for a later reward | |
        """),
        L.watch_out("In other fields, *Monte Carlo* names any method that averages over random draws, such as playing out a thousand possible years of an investment. In reinforcement learning, Monte Carlo methods are those that learn from whole episodes."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Monte Carlo prediction on ROOM, walk by walk. The robot starts every walk in the corner `(0, 0)` and wanders with the random policy until it reaches the charger. A number is the <span class="c-pred">estimate</span> of that tile after the walks so far: the average of the returns after its first visits. The chart beside it follows the corner: first visit and every visit, after every walk.
    """)
    return


@app.cell(hide_code=True)
def explore_walks_cell():
    explore_walks = L.explore("play", "walks", lambda _v: C.play_frame(int(_v), L.lang), start=C.N_WALKS,
                              values=C.LOOKS, role="",
                              task="Go to walk 1: why does every visit give the corner −24.9, while first visit gives −59.0? Then go to 10, 100 and 1,000 walks: how far is the corner from −36 each time?")
    explore_walks
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    In walk 1, of 59 steps, the robot is in the corner 9 times: at the start, after bumping into the edge, and when it comes back later. First visit takes the return of the first visit, −59, the whole walk. Every visit also takes the 8 returns after the later visits, from −58 down to −6, and the nine returns together average −24.89. Over many walks the two come together: after 10 walks the corner is at −26.90 with first visit, after 100 at −35.83, and after 1,000 at −35.89, with every visit at −36.17. One walk gives a return to every tile it visits: after 20 walks every tile has a number, and after 1,000 walks every estimate lies within 1.85 of its value from policy evaluation.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Monte Carlo prediction of a policy $\pi$, first visit:

    1. Play an episode with $\pi$, from a start state until it ends. Note for every step $t$ the state and the reward right after it.
    2. Go through the episode backward, from the last step, with a return $G$ of 0: at each step $G$ becomes the reward of that step plus $\gamma$ times $G$, the return from step $t$.
    3. For each state, keep the return of its first visit in the episode: the first step at which the episode is in that state.
    4. The estimate of $V(s)$ is the average of all returns kept for $s$, over all episodes so far.
    5. Play the next episode and go back to step 1.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The corridor of policy evaluation: four tiles, the charger on tile 0 at the left end and a wall after tile 3. The robot steps left or right, each with probability 1/2; against the wall it stays on tile 3. Every step costs −1, and $\gamma$ = 1, so the return of a walk from tile 3 is minus its number of steps. The cell lets the robot walk from tile 3 a thousand times, with `rng.random()` as the coin, and shows the average of the returns after 1, 10, 100 and 1,000 walks. It uses only the return from the start, where tile 3's first visit is, so it needs no step 3. Policy evaluation gave tile 3 the value −12. Run the cell.
    """)
    return


@app.cell
def example_corridor():
    import numpy as np

    _rng = np.random.default_rng(0)
    _returns = []
    for _i in range(1000):
        # 1. one walk from tile 3 with the random policy, until the charger on tile 0
        _s = 3
        _steps = 0
        while _s != 0:
            if _rng.random() < 0.5:
                _s = _s - 1
            else:
                _s = min(_s + 1, 3)   # against the wall the robot stays on tile 3
            _steps = _steps + 1
        # 2. every step costs -1 and gamma = 1, so the return of the walk is minus its number of steps
        _returns.append(-_steps)
        # 4. the estimate: the average of the returns so far
        if _i + 1 in [1, 10, 100, 1000]:
            print("walks:", _i + 1, "estimate:", round(sum(_returns) / len(_returns), 2))
    print("the shortest walk took", -max(_returns), "steps, the longest", -min(_returns))
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first walk took 4 steps, so after 1 walk the estimate is −4. After 10 walks it is −21.2, further from −12 than after 1: a few long walks pull it down. After 100 walks it is −13.82, and after 1,000 walks −12.22, close to the −12 of policy evaluation. The walks took from 3 to 86 steps: one walk says little about the value; the average of many says much.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_returns():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The returns of an episode

        `returns(episode, gamma)` gives back the return from every step of an episode, as a list: place `t` holds $G_t$. An episode is a list of steps, and each step a pair `(state, reward)`: the state the robot was on and the reward of the step it took from there. `out` starts with a 0 for every step, and the loop over `range(len(episode) - 1, -1, -1)` goes over `t` from the last step back to the first: from `len(episode) - 1` down to 0, one step at a time. `s, r = episode[t]` takes step `t` apart into its state and its reward, and `G` holds the return from the step after `t`, 0 at the start. Replace the `...` with the return from step `t`, step 2 of the recipe. The check calls `returns(WALK, 0.9)` and five other cases, among them a walk whose last step pays +10. `WALK` is a walk in the corridor of the worked example, with the tiles as numbers: from tile 3, the robot bumps into the wall, goes left twice, right once and left twice, onto the charger: 6 steps of −1.
        """),
        L.task("Make `returns` work out the return from each step, going backward from the last step."),
    ])
    return


@app.cell(hide_code=True)
def exercise_returns():
    box_returns = L.code_box("returns", lines=10)
    box_returns
    return (box_returns,)


@app.cell(hide_code=True)
def check_returns(box_returns):
    result_returns = L.check("returns", box_returns.value)
    result_returns
    return (result_returns,)


@app.cell(hide_code=True)
def hints_returns():
    L.hints(
        "returns",
        [
            "Step 2 of the recipe, and the first formula of *The idea*: the loop goes from the last step back to the first, and when the line runs for step `t`, `G` still holds the return from the step after it. `r` is the reward of step `t`, and `gamma` the discount factor.",
            "In `WALK`, with γ = 0.9, the last step gives −1; the step before it −1 + 0.9 · (−1) = −1.9, and the one before that −1 + 0.9 · (−1.9) = −2.71.",
            "`        G = ___`",
        ],
        why="`G = r + gamma * G`: the return from step `t` is its own reward plus $\\gamma$ times the return from the next step. Going backward, each step needs one addition and one multiplication, however long the episode.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_first():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. First-visit Monte Carlo

        `first_visit(episodes, gamma)` gives back the estimate of every state the episodes visit: a dictionary from state to average return. `returns` is the course's version of your function of exercise 1. For each `episode`, `G` holds its returns, `s = episode[t][0]` is the state of step `t`, and `seen` starts as an empty set, `set()`, of the states the episode has been in so far; `seen.add(s)` puts `s` in it. `total[s]` adds up the returns kept for `s`, and `count[s]` counts them. Replace the two `...`: when a step counts, step 3 of the recipe, and the estimate of `s`, step 4. The check calls `first_visit(TEN_WALKS, 1.0)`, with `TEN_WALKS` the first ten walks of the play block, and five other cases.
        """),
        L.task("Make `first_visit` keep only the return of each state's first visit in an episode, and give back the average of the returns kept."),
    ])
    return


@app.cell(hide_code=True)
def exercise_first():
    box_first = L.code_box("first", lines=20)
    box_first
    return (box_first,)


@app.cell(hide_code=True)
def check_first(box_first):
    result_first = L.check("first", box_first.value)
    result_first
    return (result_first,)


@app.cell(hide_code=True)
def hints_first():
    L.hints(
        "first",
        [
            "Step 3 of the recipe: a step counts only at the first visit of its state in the episode. `s` is the state of step `t`, and `seen` holds the states this episode has already been in. Step 4: the estimate is an average; `total` adds up the returns of each state and `count` counts them.",
            "In walk 1 of the play block the robot is in the corner (0, 0) at step 0 and at step 1: only step 0 counts, with a return of −59. When step 1 comes, (0, 0) is already in `seen`, and so it is at the 7 visits after that.",
            "`            if ___:`",
        ],
        why="`if s not in seen:` lets only the first visit of `s` in an episode through, and `V[s] = total[s] / count[s]` makes the estimate the average of the returns kept. With `if True:` you would have every-visit Monte Carlo.",
    )
    return


@app.cell(hide_code=True)
def text_after_first():
    mo.md(r"""
    After the ten walks, your function gives the corner −26.90, and after all 1,000 walks of the play block −35.89, with −23.25 on the tile above the charger, (1, 2). Write `True` in place of the condition and you have every-visit Monte Carlo: after 1,000 walks it gives the corner −36.17.

    ### Many walks at once with NumPy

    How fast does the spread shrink? The cell plays 100,000 walks from the corner at the same time, as arrays. `row` and `col` hold where each walk is, `steps` counts its steps, and `going` is `True` for every walk that has not reached the charger yet; `dtype` says what kind of numbers an array holds. In every round each walk takes a random step: `rng.integers(0, 4, size=n)` draws a direction for each, and a comparison such as `move == 0` gives `True`, which counts as 1, for the walks that go up. `np.clip(x, 0, 4)` keeps each number of `x` between 0 and 4, so a step into the edge stays put. `steps + going` adds 1 for every walk still going, and `&` (and) and `|` (or) combine comparisons element by element: a walk keeps going while it is not on both row 2 and column 2. At the end, `reshape(-1, k)` puts the walks in rows of `k`, where −1 lets NumPy work out the number of rows itself, and `mean(axis=1)` gives the average of each row. Run the cell.
    """)
    return


@app.cell
def example_numpy(np):
    _rng = np.random.default_rng(0)
    _n = 100000
    _row = np.zeros(_n, dtype=int)
    _col = np.zeros(_n, dtype=int)
    _steps = np.zeros(_n, dtype=int)
    _going = np.ones(_n, dtype=bool)   # True for every walk that has not reached the charger yet
    while _going.sum() > 0:
        _move = _rng.integers(0, 4, size=_n)   # 0 up, 1 down, 2 left, 3 right
        _row = np.clip(_row - (_move == 0) + (_move == 1), 0, 4)
        _col = np.clip(_col - (_move == 2) + (_move == 3), 0, 4)
        _steps = _steps + _going
        _going = _going & ((_row != 2) | (_col != 2))
    print("mean:", round(_steps.mean(), 2), "standard deviation:", round(_steps.std(), 2))
    for _k in [10, 100, 1000]:
        _averages = _steps.reshape(-1, _k).mean(axis=1)
        print(_k, "walks per average: spread", round(_averages.std(), 2), "expected:", round(31.7 / np.sqrt(_k), 2))
    return


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    The 100,000 walks take 35.95 steps on average, with a standard deviation of 31.68: close to the 36 and 31.7 that follow from the table. Averages of 10 walks spread 9.93 steps, of 100 walks 3.18, of 1,000 walks 0.95: each time close to the expected 31.7/√k. There are only 100 averages of 1,000 walks, so their spread is itself measured with some noise. Ten times as many walks make the spread about 3.2 times smaller, since √10 ≈ 3.2. That is the price of Monte Carlo: it needs no table, but it needs many episodes.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_credit():
    mo.md(r"""
    ### Exercise 3. Who pays for the detour?

    The robot starts a walk in the corner. In 3 steps it reaches the tile next to the charger, then it wanders off and needs 57 more steps to reach the charger: 60 steps in all, each costing −1, with $\gamma$ = 1.
    """)
    return


@app.cell(hide_code=True)
def choice_credit_cell():
    choice_credit = L.choice(
        "credit",
        question="What does first-visit Monte Carlo keep for the corner from this walk?",
        options={
            "one": "−1: only the reward of the step taken from the corner itself",
            "three": "−3: the corner's part of the walk went well up to the tile next to the charger, and the detour after it is not the corner's doing",
            "right": "−60: the return of the whole walk after the corner, the 57 steps of the detour too",
            "none": "Nothing: a walk this long is an exception, and Monte Carlo leaves it out",
        },
        correct="right",
        why={
            "right": "Monte Carlo gives a state the whole return after its first visit: −60 for the corner, though its own 3 steps went well and the detour came later. Which steps earned a return is the credit assignment problem, and Monte Carlo solves it only on average: over many walks the returns from the corner, detours and all, average out to −36. Such long, varied returns are also why the returns from the corner spread about 32 steps, and why 1,000 walks still leave about 1 step of doubt.",
            "one": "Monte Carlo prediction estimates $V_\\pi$, the return to expect from a state to the end of the episode: does the return of the corner stop after one step?",
            "three": "Monte Carlo does not know which steps went well and which did not; it only sees the rewards of the walk: which of them come after the corner's first visit?",
            "none": "Long walks are part of what the random robot does: what would the average of the returns be if Monte Carlo left the long ones out?",
        },
    )
    choice_credit
    return (choice_credit,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [What is a policy worth? Policy evaluation](/course/open/rl-policy-evaluation?lang=en), you worked out the values of the random robot from the table `T`: −24 next to the charger, −36 in a corner. Your `returns` and `first_visit` estimate the same values from walks alone, as the agent in [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en) learned from its own steps, without `T`. The price is many episodes: an average of $n$ returns spreads $\sqrt{n}$ times less than one return, and every step gets the whole return after it, the detours too. The next lesson, *Playing better from experience: Monte Carlo control*, estimates $Q(s, a)$ the same way and uses it to make the policy better, in blackjack. Later in this part, TD learning updates a value after every step, without waiting for the end of the episode.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_credit, result_first, result_returns):
    L.wrap_up(result_returns, result_first, choice_credit)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
