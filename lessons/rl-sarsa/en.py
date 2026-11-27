# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-sarsa. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the course's cliff, its runs of SARSA and Q-learning, and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-sarsa", "en")
    C = L.checks   # the course's cliff, its runs of SARSA and Q-learning, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Learning after every step: TD(0)](/course/open/rl-td-prediction?lang=en) and [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_td():
    quick_td = L.choice(
        "quick-td",
        question="A step goes from $S$ to $S'$ with reward 0 and $\\gamma$ = 1, while $V(S)$ = 0.5 and $V(S')$ = 0.6. What is the TD error $\\delta$?",
        options={
            "target": "0.6, the target itself",
            "right": "0.1, the target minus the old estimate",
            "minus": "−0.1, the old estimate minus the target",
            "add": "1.1, the reward, the target and the old estimate together",
        },
        correct="right",
        why={
            "right": "$\\delta = R + \\gamma V(S') - V(S) = 0 + 0.6 - 0.5 = 0.1$: the step went a little better than $V(S)$ predicted. This lesson uses the same kind of target for $Q$.",
            "target": "The target is $R + \\gamma V(S') = 0.6$, but the TD error says how far the old estimate is from it: what do you take away? See [Learning after every step: TD(0)](/course/open/rl-td-prediction?lang=en).",
            "minus": "The sign matters: a step that goes better than predicted should push $V(S)$ up. Which comes first? See [Learning after every step: TD(0)](/course/open/rl-td-prediction?lang=en).",
            "add": "The TD error is a difference, not a total: which two numbers does it compare? See [Learning after every step: TD(0)](/course/open/rl-td-prediction?lang=en).",
        },
    )
    quick_td
    return (quick_td,)


@app.cell(hide_code=True)
def quick_check_q():
    quick_q = L.choice(
        "quick-q",
        question="In Q-learning, an agent takes action $a$ in state $s$, gets reward $r$ and lands in $s'$. Which number of $s'$ goes into its target $r + \\gamma \\cdot \\ldots$?",
        options={
            "next": "The $Q$ of the action the agent will take next in $s'$",
            "right": "The largest $Q$ in $s'$, over all actions",
            "mean": "The mean of the four $Q$'s in $s'$, since the agent may take any of them",
            "table": "$V(s')$ from the table `T`",
        },
        correct="right",
        why={
            "right": "Q-learning takes $\\max_{a'} Q(s', a')$: the value of $s'$ if the agent takes the best action there, whatever it does next. This lesson's new method does take the action that comes next.",
            "next": "Q-learning does not look at the action that comes next: which $Q$ of $s'$ does its max pick? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
            "mean": "Q-learning takes one $Q$ of $s'$, not all four: which one does its max pick? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
            "table": "Q-learning needs no table of probabilities, only the Q-table it learns: which $Q$ of $s'$ does it use? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
        },
    )
    quick_q
    return (quick_q,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    An agent that learns by trying makes mistakes on purpose: with ε-greedy it takes a random action now and then, to find out what it does. In a simulation that costs nothing, but a robot that learns in a real building, next to a staircase, pays for every random step. So it matters what the agent learns: what its actions are worth if it always takes the best one, or what they are worth with its own random steps in them. Q-learning does the first, and SARSA, the method of this lesson, the second. On the same task they learn different paths. The difference has a name you will meet in all of RL, on-policy against off-policy, and it decides which data a method can learn from.

    **An everyday example.** You walk along a canal with your eyes on your phone, so now and then you take a step without looking. The shortest way runs right along the water. If you plan as if you will always look up in time, you walk along the edge; if you plan with your own careless steps in mind, you walk a meter further from the water. Both plans are right, each for a different walker.
    """)
    return


@app.cell(hide_code=True)
def predict_safe():
    guess_safe = L.guess(
        "safe",
        question="A grid of 4 by 12 has a cliff along the bottom, between the start and the goal. Every step gives −1; a step into the cliff gives −100 and puts the agent back at the start. SARSA and Q-learning both learn for 500 episodes, with ε-greedy, ε = 0.1, and η = 0.5. Over the last 100 episodes, averaged over 50 runs, which method has the higher mean return per episode while it learns?",
        options={
            "q": "Q-learning: it learns the shortest way, 13 steps along the edge, so its episodes cost the least",
            "same": "About the same: after 400 episodes both have learned the cliff, so they walk alike",
            "right": "SARSA: its episodes cost about half as much",
            "fall": "Neither does well: with ε = 0.1 both keep falling off, about once in every episode",
        },
        reveal="**SARSA: its episodes cost about half as much.** Over the last 100 episodes, SARSA's mean return is −28 and Q-learning's −50, and SARSA has the higher one in 49 of the 50 runs. Q-learning has learned the shortest way, 13 steps along the edge, but one step in ten is random, and next to the edge a random step down falls: Q-learning falls in one of every three episodes. SARSA walks the top row, 17 steps, and falls in one of every twenty.",
    )
    guess_safe
    return (guess_safe,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The cliff.** A grid of 4 rows and 12 columns. The agent starts in S, bottom left, and the episode ends in G, bottom right. The ten cells between them are the cliff. In every cell the agent can go up, down, left or right; at the edge of the grid it stays where it is. Every step gives a reward of −1, and a step into the cliff gives −100 and puts the agent back on S, where the episode goes on; $\gamma$ = 1. The return of an episode is then minus the number of steps, with 99 more off for each fall. The shortest way is up, 11 steps to the right, and down: 13 steps, a return of −13. In code a state is (row, column), counted from the top left: S is (3, 0), G is (3, 11), and the cliff runs from (3, 1) to (3, 10). Both methods start with every $Q$ at 0 and learn with ε-greedy, ε = 0.1, and with learning rate $\eta$ = 0.5.

        **SARSA.** Q-learning moves $Q(S, A)$ toward $R + \gamma \max_{a'} Q(S', a')$: the value of $S'$ if the agent takes the best action there. SARSA first chooses the action $A'$ the agent will really take in $S'$, with ε-greedy, and moves $Q(S, A)$ toward $R + \gamma Q(S', A')$:
        """),
        L.formula(r"[[param:Q(S, A)]] \leftarrow [[param:Q(S, A)]] + [[hyper:\eta]] \left( [[data:R]] + [[hyper:\gamma]] [[param:Q(S', A')]] - [[param:Q(S, A)]] \right)"),
        mo.md(r"""
        The name lists what one update needs: $S$, $A$, $R$, $S'$ and $A'$. Then the agent takes $A'$, so the action in the target is the action it does next.

        **On-policy and off-policy.** While it learns, the agent follows ε-greedy. SARSA learns the values of that very policy, random steps and all: it is **on-policy**. Q-learning learns the values of always taking the best action, while it follows ε-greedy: it is **off-policy**, it learns about a different policy from the one it follows. Next to the cliff this makes the difference. One step in ten is random, and one random step in four goes down: along the edge, the agent falls once in 40 steps. The target of SARSA sometimes holds the −100 of such a fall, so SARSA learns that the cells along the edge are worth less, and keeps away from them. The target of Q-learning takes the largest $Q$, never the step down: it learns the short way, as if it will never take a random step again.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **SARSA** | sarsa | TD learning of $Q(s, a)$ toward $R + \gamma Q(S', A')$, with $A'$ the action the agent really takes next | `sarsa_update` |
        | **on-policy** | | learning the values of the policy the agent follows while it learns, random steps included | |
        | **off-policy** | | learning the values of a different policy from the one the agent follows, such as always taking the best action | |
        | $A'$ | A prime | the action the agent will take in $S'$ | `nxt_action` |
        | $S'$ | S prime | the state after the step | `nxt` |
        """),
        L.watch_out("SARSA needs $A'$ before it can update, so it chooses $A'$ first and then really takes it in the next step. If it chose again after the update, the target would hold an action the agent never took."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    SARSA and Q-learning on the cliff, both with ε = 0.1 and $\eta$ = 0.5, 50 runs of 500 episodes each. On the left, the mean return of the last 10 episodes over the 50 runs: SARSA solid, Q-learning dashed; the first episodes lie far below −100 and stay at the bottom edge. On the right, the best action of every state in run 1 as small arrows, and in <span class="c-param">orange</span> the way they lead from S, once they lead to G.
    """)
    return


@app.cell(hide_code=True)
def explore_cliff_cell():
    explore_cliff = L.explore("play", "episodes", lambda _v: C.play_frame(int(_v), L.lang), start=C.EPISODES,
                              values=C.LOOKS, role="",
                              task="Go to 50: whose arrows lead to G first, and along which row? Then go on to 500: which line stays higher, although its way is longer?")
    explore_cliff
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    The first episode takes both agents hundreds of steps: over the 50 runs its mean return is −1,833 for SARSA and −1,800 for Q-learning. With every $Q$ at 0, the agent wanders around and falls many times. After 50 episodes, the arrows of Q-learning in run 1 lead along the edge, 13 steps; after 100 episodes, those of SARSA go up to the top row, 17 steps. From then on SARSA's line stays above: over the last 100 episodes SARSA's mean return is −28 and Q-learning's −50. Q-learning falls in one of every three episodes, and SARSA in one of every twenty.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    SARSA, for one episode:

    1. Start in S, the start cell: it is the first state $S$. Choose an action $A$ with ε-greedy.
    2. Take $A$: you get a reward $R$ and land in $S'$.
    3. Choose the next action $A'$ in $S'$, with ε-greedy.
    4. Move $Q(S, A)$ toward the target $R + \gamma Q(S', A')$, by $\eta$ times the difference.
    5. $S'$ becomes $S$ and $A'$ becomes $A$. If $S$ is G, the episode is over; otherwise, go on at step 2.

    Q-learning differs only in step 4: its target is $R + \gamma \max_{a'} Q(S', a')$, so it does not need $A'$ to update. In G, SARSA still chooses an $A'$, but every $Q$ of G stays 0: no step starts from G, so the last target is just $R$.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Suppose that after some episodes the agent is in (2, 5), on the row next to the cliff, and goes right, to (2, 6), with reward −1. $Q$((2, 5), right) is −7, and the four values of (2, 6) are: up −8, down −114, left −8 and right −6. $\eta$ = 0.5 and $\gamma$ = 1.

    - Q-learning: the target is $-1 + \max(-8, -114, -8, -6) = -1 - 6 = -7$, so $Q$((2, 5), right) $= -7 + 0.5 \cdot (-7 - (-7)) = -7$: nothing changes.
    - SARSA, when ε-greedy picks the best action in (2, 6), right: the target is $-1 + (-6) = -7$, the same, and nothing changes.
    - SARSA, when the random action is down: the target is $-1 + (-114) = -115$, and $Q$((2, 5), right) $= -7 + 0.5 \cdot (-115 - (-7)) = -7 - 54 = -61$.

    A random step down comes once in 40 steps here: 0.1 times 1/4. Each time, SARSA pulls the $Q$ of walking along the edge far down, and the next ordinary steps pull it back up, halfway each time. So on average SARSA's $Q$ for the cells along the edge stays lower than Q-learning's: SARSA factors in the falls that ε-greedy will cause, and its arrows turn away from the edge.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_update():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. One step of SARSA

        `sarsa_update(Q, state, action, reward, nxt, nxt_action, eta, gamma)` gives back the new $Q$ of `state` and `action`: step 4 of the recipe. The agent took `action` in `state`, got `reward`, landed in `nxt`, and will take `nxt_action` there. `Q` is a dict with a dict per state, as in Q-learning: `Q[state][action]`. `eta` and `gamma` are $\eta$ and $\gamma$. Replace the `...` with the `target`.

        The check calls `sarsa_update(Q, (2, 5), "right", -1, (2, 6), "down", 0.5, 1.0)`, the step of the worked example where the random action is down, with `Q` the Q-table of the worked example. Then four other steps with `Q_500`, the Q-table of the course's run 1 of SARSA after 500 episodes: on the top row, a fall into the cliff, and two with $\gamma$ = 0.9.
        """),
        L.task("Make `sarsa_update` move Q(s, a) toward the reward plus γ times the Q of the next action in the next state."),
    ])
    return


@app.cell(hide_code=True)
def exercise_update():
    box_update = L.code_box("sarsa_update", lines=4)
    box_update
    return (box_update,)


@app.cell(hide_code=True)
def check_update(box_update):
    result_update = L.check("sarsa_update", box_update.value)
    result_update
    return (result_update,)


@app.cell(hide_code=True)
def hints_update():
    L.hints(
        "sarsa_update",
        [
            "Step 4 of the recipe and the formula of *The idea*: the `target` is the `reward` plus `gamma` times one $Q$ of the next state. Which state is next, and which action will the agent take there?",
            "In the worked example, with the random step down: $-1 + 1 \\cdot Q$((2, 6), down) $= -1 - 114 = -115$. The function gets (2, 6) and down under two of the names between its parentheses: which ones?",
            "`    target = reward + gamma * ___`",
        ],
        why="`target = reward + gamma * Q[nxt][nxt_action]` is the target of SARSA: the reward plus the discounted $Q$ of the action the agent really takes next. With the largest $Q$ of `nxt` in its place, it would be Q-learning.",
    )
    return


@app.cell(hide_code=True)
def text_after_update():
    mo.md(r"""
    In the picture, your `sarsa_update` learns run 1 of the play block. After 500 episodes its arrows go up from S, along the top row and down to G: 17 steps. Run 1 is one of the less lucky runs: over its last 100 episodes the agent fell 8 times, and its mean return is −39.3.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_episode():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A whole episode

        `sarsa_episode(Q, eta, epsilon, gamma, rng)` plays one episode of SARSA on the cliff, updates `Q` after every step and gives back `total`, the return. It uses these things of the course: `step(state, action)`, the cliff, which gives back `nxt` and `reward`; `choose(Q, state, epsilon, rng)`, ε-greedy as you wrote it in Q-learning, with `rng` its random numbers; `sarsa_update`, the course's own solution of exercise 1; and `START` and `GOAL`, (3, 0) and (3, 11). The loop `while state != GOAL:` repeats its block as long as the agent is not on G. Replace the `...`, so that `nxt_action` is the action the agent will take in the state it lands on: step 3 of the recipe. Two lines further, `action = nxt_action` makes the agent really take it.

        The check calls `sarsa_episode(Q_50, 0.5, 0.1, 1.0, np.random.default_rng(1))`, with `Q_50` the Q-table of the course's run 1 after 50 episodes and a new random generator with seed 1, and compares the return and the Q-table your function leaves with those of the course. Then four other cases: seed 2 with ε = 0.2, a table of zeros, `Q_500` with $\eta$ = 0.1 and $\gamma$ = 0.9, and `Q_50` with $\eta$ = 0.3, ε = 0.05 and $\gamma$ = 0.9.
        """),
        L.task("Make `sarsa_episode` choose the next action with ε-greedy, in the state the agent has just landed on, before the update."),
    ])
    return


@app.cell(hide_code=True)
def exercise_episode():
    box_episode = L.code_box("sarsa_episode", lines=13)
    box_episode
    return (box_episode,)


@app.cell(hide_code=True)
def check_episode(box_episode):
    result_episode = L.check("sarsa_episode", box_episode.value)
    result_episode
    return (result_episode,)


@app.cell(hide_code=True)
def hints_episode():
    L.hints(
        "sarsa_episode",
        [
            "Step 3 of the recipe: `nxt_action` is the ε-greedy choice of the agent in the state it has just landed on. `choose` takes the Q-table `Q`, a state, ε and the random numbers, as on the line before the loop.",
            "Before the loop the agent chose its first action in the state where it started. Where does it stand after `step`, and which name on the line with `step` holds that state?",
            "`        nxt_action = choose(Q, ___)`",
        ],
        why="`nxt_action = choose(Q, nxt, epsilon, rng)` chooses the next action in `nxt` before the update, since the target of SARSA needs it; two lines further, `action = nxt_action` makes the agent really take it.",
    )
    return


@app.cell(hide_code=True)
def text_after_episode():
    mo.md(r"""
    In the picture, your `sarsa_episode` plays 200 episodes in a row, from a table of zeros: the first one wanders for hundreds of steps, and by the end the arrows take the top row, 17 steps. The last 50 episodes give a mean return of −21.9.

    ### All runs at once with NumPy

    One run says little. `C.course_returns()` gives the return of every episode of every run, for SARSA and for Q-learning, as two arrays of shape `(50, 500)`: a row per run, a column per episode. `C.course_falls()` gives the falls of every episode in the same shape. `_sarsa[:, 400:]` takes the last 100 episodes of every run, and `np.mean` without an axis averages all its numbers. `np.mean(..., axis=1)` averages every row: one number per run. `np.max` and `np.min` give the largest and the smallest number of an array. Run the cell.
    """)
    return


@app.cell
def example_numpy():
    import numpy as np

    _sarsa, _q = C.course_returns()               # shape (50, 500): a row per run, a column per episode
    _falls_sarsa, _falls_q = C.course_falls()
    _last_sarsa = np.mean(_sarsa[:, 400:], axis=1)   # the mean return of the last 100 episodes, one number per run
    _last_q = np.mean(_q[:, 400:], axis=1)
    print("shape:", _sarsa.shape)
    print("mean return of the last 100 episodes, SARSA:", round(float(np.mean(_sarsa[:, 400:])), 1), "Q-learning:", round(float(np.mean(_q[:, 400:])), 1))
    print("falls per episode, last 100 episodes, SARSA:", round(float(np.mean(_falls_sarsa[:, 400:])), 3), "Q-learning:", round(float(np.mean(_falls_q[:, 400:])), 3))
    print("runs in which SARSA has the higher mean return:", int((_last_sarsa > _last_q).sum()))
    print("best and worst run, SARSA:", round(float(np.max(_last_sarsa)), 1), round(float(np.min(_last_sarsa)), 1))
    print("best and worst run, Q-learning:", round(float(np.max(_last_q)), 1), round(float(np.min(_last_q)), 1))
    return (np,)


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    Over the last 100 episodes SARSA falls 0.051 times per episode and Q-learning 0.332 times, six or seven times as often. SARSA has the higher mean return in 49 of the 50 runs. The runs spread a lot: SARSA's best run has −20.8 and its worst −63.4, Q-learning's best −34.8 and its worst −61.0. With $\eta$ = 0.5 the values keep moving, and a few unlucky falls in a row can send the arrows of SARSA closer to the edge for a while.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your `sarsa_update` beside Q-learning

    The demo runs the 50 runs of the play block with your `sarsa_update` for SARSA, and the course's Q-learning beside it. Where your function does not pass its check yet, the course's takes its place. Pick ε for both methods: with 0.1 you see the play block again; then try 0.2, 0.05 and 0.01. The chart shows the mean return of the last 10 episodes over the 50 runs, SARSA solid and Q-learning dashed; beside it, the arrows of run 1 as they are now.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    epsilon = mo.ui.dropdown(options=C.eps_options(L.lang), value=C.eps_text(0.1, L.lang), label=C.DEMO_TEXT[L.lang]["eps"])
    play_button = mo.ui.run_button(label="Run 50 runs with my sarsa_update")
    mo.hstack([epsilon, play_button], justify="start", gap=1.5, wrap=True)
    return epsilon, play_button


@app.cell(hide_code=True)
def demo_play(box_update, epsilon, play_button, result_update):
    learned = None
    if not play_button.value:
        _out = mo.md("*Pick ε, then press the button to let both methods learn the cliff.*")
    else:
        _mine = result_update.state == "correct"
        _fn = run_code(box_update.value).names["sarsa_update"] if _mine else C.sarsa_update
        _w = C.DEMO_TEXT[L.lang]
        _now = {"shown": 0}

        def _progress(_n, _run):
            if "run" not in _now:
                _now["chart"].watch(lambda: C.now_picture(_now["run"], _now["n"], L.lang))
            _now["run"], _now["n"] = _run, _n
            if _n % 5 == 0 or _n == 1:
                _now["chart"].reference(C.chart_points(_run, "q", 1, _n), "", role="ink", replace=True)
                _now["chart"].extend(C.chart_points(_run, "sarsa", _now["shown"] + 1, _n), f"**Episode {_n} of {C.EPISODES}**")
                _now["shown"] = _n

        _live = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"], "alt": _w["alt"]}, C.EPISODES, role="ink",
                       low=C.LOW, high=0.0, window=1, key=C.curve_key(L.lang))
        _live.line("**Learning** …")
        _now["chart"] = _live
        _r = C.run(_fn, epsilon.value, show=_progress)
        if _r == "sarsa":
            _out = _live.stopped("*Your `sarsa_update` gave an error during the run, or gave something that is not a number. Does it still pass its check?*")
        elif _r == "slow":
            _out = _live.stopped("*The run took more than a minute. Does your function do more work than it needs?*")
        else:
            learned = _r
            _who = "your `sarsa_update`" if _mine else "the course's `sarsa_update`, since yours does not pass its check yet"
            _out = mo.vstack([
                L.explore("rewind", "episodes", lambda _v: C.rewind_frame(_r, int(_v), L.lang), start=C.EPISODES,
                          values=C.LOOKS, role="",
                          task="Go back to 20, then forward again: when do the arrows of each method first lead to G, and when does the solid line of SARSA rise above the dashed one?"),
                mo.md(f"Run with {_who} and ε = {C.eps_text(epsilon.value, 'en')}. Over the last 100 episodes, SARSA's mean return is {C.number(C.last_mean(_r, 'sarsa', C.EPISODES, 100), 1, 'en')} and Q-learning's {C.number(C.last_mean(_r, 'q', C.EPISODES, 100), 1, 'en')}."),
            ])
    _out
    return (learned,)


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With the course's functions, over the last 100 episodes: with ε = 0.2, SARSA has a mean return of −42.7 and Q-learning −106.4; with ε = 0.05, −22.5 against −30.2; with ε = 0.01, −17.0 against −16.3, and now Q-learning is ahead. The more the agent explores, the more Q-learning's way along the edge costs while it learns. With ε = 0.01 a random step down comes once in 400 steps, and the shorter way wins even while learning. On this small grid even ε = 0.01 explores enough; on bigger tasks, too little exploring can keep an agent from ever finding a better way.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_robot():
    mo.md(r"""
    ### Exercise 3. Learning in a simulator

    A team trains a delivery robot in a simulator of a building with a long balcony without a railing. A fall costs −100 there, and nothing breaks. The robot learns with ε-greedy, ε = 0.1. Afterwards it works in the real building and always takes its best action: no more random steps.
    """)
    return


@app.cell(hide_code=True)
def choice_robot_cell():
    choice_robot = L.choice(
        "robot",
        question="Which method should the team train with, for the way the robot will walk in the real building?",
        options={
            "sarsa": "SARSA: a way that keeps away from the edge is always the better one, also without random steps",
            "right": "Q-learning: it learns what the best actions are worth, and the robot will only take those",
            "same": "Either: after enough episodes both learn the same way",
            "neither": "Neither: a robot that learned with random steps cannot walk without them",
        },
        correct="right",
        why={
            "right": "Q-learning is off-policy: its values are those of always taking the best action, whatever the robot did while it learned. Without random steps its short way along the edge never falls: on the cliff of this lesson 13 steps, against 17 for the way of SARSA. SARSA is on-policy: it learned the values of the robot with its random steps, and walks around a danger that is gone.",
            "sarsa": "Without random steps, does the robot ever fall from the edge? Which method learned the values of exactly that robot?",
            "same": "On the cliff, after 500 episodes, the arrows of Q-learning take the edge and those of SARSA the top row: why do they stay apart while ε = 0.1?",
            "neither": "The arrows say which action is best in every state; does the robot need random steps to follow them?",
        },
    )
    choice_robot
    return (choice_robot,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Learning after every step: TD(0)](/course/open/rl-td-prediction?lang=en), you learned $V$ of a fixed policy after every step. SARSA does the same for $Q(s, a)$ of the policy the agent follows, while ε-greedy keeps improving that policy as the values change: TD learning that chooses. [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en) is its off-policy counterpart. The next lesson, *A map of RL methods*, puts the methods you know side by side: with or without a model of the world, learning values or the policy itself, on-policy or off-policy, from whole episodes or from their own estimates.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_robot, result_episode, result_update):
    L.wrap_up(result_update, result_episode, choice_robot)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
