# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-td-prediction. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the course's random walk, its runs of TD(0) and Monte Carlo, and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-td-prediction", "en")
    C = L.checks   # the course's random walk, its runs of TD(0) and Monte Carlo, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Learning from whole games: Monte Carlo prediction](/course/open/rl-monte-carlo-prediction?lang=en) and [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_mc():
    quick_mc = L.choice(
        "quick-mc",
        question="With Monte Carlo prediction, when can an agent update its estimate $V(s)$ for the states of an episode?",
        options={
            "step": "After every step, from the reward of that one step",
            "right": "Only at the end of the episode, when the return of every state is known",
            "ten": "After every ten steps, from the rewards of those ten steps, so that the agent learns a long episode in parts",
            "never": "Only after all episodes have been played, all at once",
        },
        correct="right",
        why={
            "right": "The return of a state counts every reward up to the end of the episode, so Monte Carlo has to wait for the end. This lesson updates after every step instead.",
            "step": "The return counts every reward from the state to the end of the episode, not only the next one: when is the last of them known? See [Learning from whole games: Monte Carlo prediction](/course/open/rl-monte-carlo-prediction?lang=en).",
            "ten": "Ten steps do not hold the rewards that come after them: which number does Monte Carlo average, and when is it complete? See [Learning from whole games: Monte Carlo prediction](/course/open/rl-monte-carlo-prediction?lang=en).",
            "never": "Monte Carlo can update after every episode, as soon as the returns of that episode are known: when is that? See [Learning from whole games: Monte Carlo prediction](/course/open/rl-monte-carlo-prediction?lang=en).",
        },
    )
    quick_mc
    return (quick_mc,)


@app.cell(hide_code=True)
def quick_check_q():
    quick_q = L.choice(
        "quick-q",
        question="In Q-learning, an agent takes action $a$ in state $s$, gets reward $r$ and lands in $s'$. What does $Q(s, a)$ move toward?",
        options={
            "return": "The return of the whole episode, once it is over",
            "reward": "The reward $r$ alone, since $Q$ counts what the action brings at once",
            "right": "$r + \\gamma \\max_{a'} Q(s', a')$: the reward plus the discounted largest $Q$ in $s'$",
            "mean": "The average of all rewards the agent got in state $s$ so far, over all actions and all episodes",
        },
        correct="right",
        why={
            "right": "Q-learning learns after every step, from its own estimate of the state it lands in. This lesson does the same for $V$ of a fixed policy, without the max.",
            "return": "Q-learning does not wait for the end of the episode: which estimate stands in for the rewards after the step? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
            "reward": "$Q(s, a)$ also counts what comes after the step, discounted by $\\gamma$: which estimate says that? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
            "mean": "$Q(s, a)$ is about one action in one state, and its target uses one step: which numbers go into it? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
        },
    )
    quick_q
    return (quick_q,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Monte Carlo prediction waits until an episode is over, since only then is the return known. For a walk of a few steps that is fine, but some tasks take very long, and some never end, such as a robot that keeps a warehouse tidy all day. Temporal-difference learning, TD learning for short, updates its estimate after every step, from the reward of the step and its own estimate of the next state. Many later methods of RL build on this idea, from Q-learning, which you know, to programs that learned to play backgammon and video games from their own play. It also turned out to match findings about the brain: dopamine cells seem to signal something very like the TD error of this lesson.

    **An everyday example.** You drive home and expect it to take 30 minutes. After 5 minutes it starts to rain and the traffic slows down; you now expect another 35 minutes, 40 in all. Do you wait until you are home to learn that 30 was too optimistic? You already know it: your new guess says so. You learn from the difference between two guesses, one a few minutes after the other, not only from the final outcome.
    """)
    return


@app.cell(hide_code=True)
def predict_closer():
    guess_closer = L.guess(
        "closer",
        question="A walk starts in the middle of five states in a row, A to E, and steps left or right at random until it leaves the row; leaving on the right pays 1, on the left 0. TD learning and Monte Carlo both estimate the values of A to E from the same 100 walks, with the same learning rate, η = 0.1. After the 100 walks, averaged over 100 runs, whose estimates are closer to the true values?",
        options={
            "mc": "Monte Carlo's: it learns from returns that really happened, while TD learns from its own guesses, which are wrong at first",
            "same": "About the same: both move their estimates toward the same values, so after 100 walks they are equally close",
            "right": "TD's, about three times as close",
            "turn": "Monte Carlo's at first, since TD needs walks to carry the news back state by state, and TD's later",
        },
        reveal="**TD's, about three times as close.** After 100 walks with η = 0.1, the estimates of TD learning are on average 0.048 from the true values, and those of Monte Carlo 0.142; TD is closer in 96 of the 100 runs, and ahead on average after every number of walks. Its guesses are wrong at first, but they change little from one walk to the next, while every return of Monte Carlo is 0 or 1.",
    )
    guess_closer
    return (guess_closer,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The random walk.** Five states in a row, A to E, with an end on both sides. Every walk starts in C, and at every step it goes one state left or right, each with a chance of 1/2, until it reaches an end. The step onto the right end pays a reward of 1, every other step 0, and $\gamma$ = 1. The value of a state is then the chance to end on the right from there: from C both ends are equally far, so $V(C) = 1/2$, and every state further right adds 1/6, from $V(A) = 1/6$ to $V(E) = 5/6$. These are the true values the course measures the estimates against; the agent never sees them. Both methods start with an estimate of 0.5 for every state. In code the states are the numbers 1 to 5, the left end 0 and the right end 6, where the value stays 0, since nothing comes after an end.

        **Monte Carlo with a fixed learning rate.** Monte Carlo prediction made $V(S)$ the average of the returns after $S$; a return $G$ holds the rewards from $S$ to the end of the episode. Here $V(S)$ moves a fixed part $\eta$ of the way toward each new $G$, as $Q$ moved toward its target in Q-learning:
        """),
        L.formula(r"[[param:V(S)]] \leftarrow [[param:V(S)]] + [[hyper:\eta]] \left( [[data:G]] - [[param:V(S)]] \right)"),
        mo.md(r"""
        **TD(0).** TD learning does not wait for $G$. After every step from a state $S$, with reward $R$, to the next state $S'$, it takes the reward of the step plus its own estimate of the rest, $R + \gamma V(S')$, as the target:
        """),
        L.formula(r"[[param:V(S)]] \leftarrow [[param:V(S)]] + [[hyper:\eta]] \left( [[data:R]] + [[hyper:\gamma]] [[param:V(S')]] - [[param:V(S)]] \right)"),
        mo.md(r"""
        The part in brackets is the **TD error**, $\delta = R + \gamma V(S') - V(S)$: how much better or worse the step went than $V(S)$ predicted. A positive $\delta$ pushes $V(S)$ up, a negative one down, and $\eta$ says how far: books on TD write α for it. Learning from your own estimate of the next state is called **bootstrapping**. Q-learning bootstraps with $\max_{a'} Q(s', a')$, and policy evaluation did it with the values of all possible next states, using their chances from the table `T`; TD(0) does it for $V$ of a fixed policy, here the random steps of the walk, from the one next state the walk really reached, without a table. **Temporal-difference learning** is the name of all methods that learn from the difference between estimates at two moments; **TD(0)** is the one that looks one step ahead.

        At the start every TD error is 0 except at the ends: when $V(S)$ and $V(S')$ are both 0.5 and the reward is 0, the step changes nothing. Only the last step of a walk brings news, and from there it travels back, one state at a time, over later walks.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **temporal-difference learning** | | learning estimates from the difference between an estimate and a better one a step later | |
        | **TD(0)** | T D zero | TD learning with one step: the target is $R + \gamma V(S')$ | `td_update` |
        | **TD error** | | how much better or worse a step went than predicted: $\delta = R + \gamma V(S') - V(S)$ | `delta` |
        | **bootstrapping** | | in RL, updating an estimate with the help of another estimate, such as $V(S')$, instead of the real return | |
        | **random walk** | | a path of random steps, here each left or right with a chance of 1/2 | `walk` |
        | $\delta$ | delta | the TD error | `delta` |
        | $S'$ | S prime | the state after the step | `s_next` |
        """),
        L.watch_out("TD(0) learns what a fixed policy is worth, as Monte Carlo prediction did; the agent here does not choose anything. Learning to choose with TD comes in the next lesson."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    TD(0) and Monte Carlo on the random walk, both with $\eta$ = 0.1 and from the same walks, 100 runs of 100 walks each. On the left, the estimates of run 1 after a number of walks, in <span class="c-param">orange</span>: TD(0) solid, Monte Carlo dashed, against the true values from 1/6 to 5/6. On the right, the average distance between the estimates and the true values over the 100 runs, in <span class="c-loss">purple</span>.
    """)
    return


@app.cell(hide_code=True)
def explore_walks_cell():
    explore_walks = L.explore("play", "walks", lambda _v: C.play_frame(int(_v), L.lang), start=C.N_WALKS,
                              values=C.LOOKS, role="",
                              task="Go to 1: which estimates did each method change after the first walk, and why only one for TD(0)? Then step to 100: whose estimates in run 1 jump around from stop to stop?")
    explore_walks
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    The first walk of run 1 visits A, B, C and D and ends on the left. Monte Carlo moves every visited state toward its return, 0: A to 0.365 and B to 0.266, a tenth of the way at each visit. TD(0) changes only A, to 0.45: on every other step both estimates were 0.5 and the reward 0, so the TD error was 0. Over the 100 runs, TD(0) is on average 0.113 from the true values after 10 walks, and Monte Carlo 0.161; after 100 walks 0.048 and 0.142. In run 1, Monte Carlo is at 0.073 after 5 walks and at 0.19 after 100: a return of 1 pulls every state of the walk up, a return of 0 pulls them all down, and $\eta$ = 0.1 moves them a tenth of the way each time. The target of TD(0), $R + V(S')$, lies close to the estimate, so its estimates move a little at a time.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    TD(0) for the values of a policy:

    1. Give every state an estimate $V$, such as 0.5, and the ends 0.
    2. Start a walk, and take a step with the policy: from $S$, with reward $R$, to $S'$.
    3. Work out the TD error, $\delta = R + \gamma V(S') - V(S)$.
    4. Move $V(S)$ by $\eta\,\delta$.
    5. If $S'$ is an end, the walk is over: start the next one at step 2. Otherwise, take the next step from $S'$ and go on at step 3.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    One walk, C → D → E → right end, with every estimate at 0.5, $\eta$ = 0.1 and $\gamma$ = 1. TD(0) updates after every step:

    - C to D, reward 0: $\delta = 0 + 0.5 - 0.5 = 0$, so $V(C)$ stays 0.5.
    - D to E, reward 0: $\delta = 0 + 0.5 - 0.5 = 0$, so $V(D)$ stays 0.5.
    - E to the right end, reward 1: $\delta = 1 + 0 - 0.5 = 0.5$, so $V(E) = 0.5 + 0.1 \cdot 0.5 = 0.55$.

    Monte Carlo waits for the end. The return after C, D and E is 1 for each, so each moves a tenth of the way toward 1: $0.5 + 0.1 \cdot (1 - 0.5) = 0.55$, all three.

    The same walk once more. TD(0): at C, $\delta = 0 + 0.5 - 0.5 = 0$ again; at D, $\delta = 0 + 0.55 - 0.5 = 0.05$, so $V(D) = 0.505$: D learns from the new estimate of E, not from the end of the walk; at E, $\delta = 1 + 0 - 0.55 = 0.45$, so $V(E) = 0.595$. Monte Carlo moves C, D and E from 0.55 to 0.595 each. The true value of C is 0.5: after two lucky walks Monte Carlo has moved C away from it, while TD(0) left C alone, since nothing it has learned so far says anything new about C.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_td():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. One step of TD(0)

        `td_update(V, s, r, s_next, eta, gamma)` does steps 3 and 4 of the recipe for one step. `V` is a list of seven estimates: the left end, A to E, and the right end. `s` and `s_next` are the states before and after the step, as numbers from 0 to 6, `r` is the reward of the step, and `eta` and `gamma` are $\eta$ and $\gamma$. The function changes `V[s]` and gives nothing back: a list that a function changes is changed for the code that called it too. Replace the `...` with the TD error `delta`.

        The check starts from `START`, 0.5 for A to E and 0 at both ends, and calls `td_update` for every step of `WALK + WALK`, the two walks of the worked example, with $\eta$ = 0.1 and $\gamma$ = 1. Then four other cases, each from `START`: `LEFT_WALK`, C, B, A and out on the left; `WALK + LEFT_WALK + WALK` with $\eta$ = 0.5 and $\gamma$ = 0.9; `LOOP_WALK`, C, D, C, D, E, D, E and out on the right, with $\eta$ = 0.2; and `LOOP_WALK + LEFT_WALK` with $\gamma$ = 0.9.
        """),
        L.task("Make `td_update` move `V[s]` by `eta` times the TD error."),
    ])
    return


@app.cell(hide_code=True)
def exercise_td():
    box_td = L.code_box("td_update", lines=5)
    box_td
    return (box_td,)


@app.cell(hide_code=True)
def check_td(box_td):
    result_td = L.check("td_update", box_td.value)
    result_td
    return (result_td,)


@app.cell(hide_code=True)
def hints_td():
    L.hints(
        "td_update",
        [
            "Step 3 of the recipe and the formula of *The idea*: `delta` is the target minus the old estimate. The target is the reward `r` plus `gamma` times the estimate of the state you reached; the old estimate is the estimate of the state you came from. Which of `V[s]` and `V[s_next]` is which?",
            "On the second walk of the worked example, at D: the reward is 0, the estimate of E is 0.55 and the estimate of D is 0.5, so the TD error is 0 + 1 · 0.55 − 0.5 = 0.05. In code, D is state 4 and E state 5.",
            "`    delta = ___`",
        ],
        why="`delta = r + gamma * V[s_next] - V[s]` is the TD error: the reward plus the discounted estimate of the next state, minus what `V[s]` said before. The line under it moves `V[s]` by `eta` times that difference.",
    )
    return


@app.cell(hide_code=True)
def text_after_td():
    mo.md(r"""
    The picture shows what your `td_update` learns from the 100 walks of run 1. After 10 walks the estimates have started to tilt, low on the left and high on the right; after 100 they lie close to the true values, on average 0.047 away.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_mc():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Monte Carlo after a whole walk

        For the comparison, Monte Carlo with a fixed learning rate. `mc_update(V, walk, eta, gamma)` gets the whole walk once it is over: `walk` is the list of its steps, each a tuple `(s, r, s_next)`. The loop goes through the walk from the end to the start with `reversed(walk)`, which gives the items of a list in reverse order without changing the list, and works out the return `G` from each step, as in Monte Carlo prediction: the reward of the step plus `gamma` times the return from the next step. Every visit of a state counts. Replace the `...` with the new `V[s]`: moved toward `G` by `eta` times the distance.

        The check starts from `START` and calls `mc_update` after each walk of `[WALK, WALK]`, with $\eta$ = 0.1 and $\gamma$ = 1. Then four other cases with `LEFT_WALK` and `LOOP_WALK`, as in exercise 1, also with $\gamma$ = 0.9.
        """),
        L.task("Make `mc_update` move every visited `V[s]` toward its return `G` by `eta` times the distance."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mc():
    box_mc = L.code_box("mc_update", lines=7)
    box_mc
    return (box_mc,)


@app.cell(hide_code=True)
def check_mc(box_mc):
    result_mc = L.check("mc_update", box_mc.value)
    result_mc
    return (result_mc,)


@app.cell(hide_code=True)
def hints_mc():
    L.hints(
        "mc_update",
        [
            "The first formula of *The idea*: `V[s]` moves toward the return `G` by `eta` times the distance between them, and `G` already holds every reward from this step to the end. What is that distance, and which of the two comes before the minus?",
            "The first walk of the worked example: `G` is 1 after C, D and E, and every estimate is 0.5, so the distance is 1 − 0.5 = 0.5, and a tenth of it takes each estimate to 0.55.",
            "`        V[s] = ___`",
        ],
        why="`V[s] = V[s] + eta * (G - V[s])` moves `V[s]` a part `eta` of the way toward the return `G`: the update of TD(0), with the whole return as the target in place of the reward plus the estimate of the next state.",
    )
    return


@app.cell(hide_code=True)
def text_after_mc():
    mo.md(r"""
    In the picture, your `mc_update` learns from the same 100 walks of run 1. After 100 walks it is on average 0.19 from the true values, four times as far as TD(0). Its estimates follow the last few walks: after 90 walks C was at 0.94, after 95 at 0.09, and after the last two walks, which both end on the right, at 0.65.

    ### All runs at once with NumPy

    One run says little. `C.course_errors()` gives the error of every run after every number of walks, for TD(0) and for Monte Carlo, both with $\eta$ = 0.1, as two arrays of shape `(100, 101)`: a row per run, a column per number of walks from 0 to 100. `np.mean(_td, axis=0)` averages every column over the 100 runs: the curve of the play block. `_td[:, 100]` is the column after 100 walks, all runs at once, and `np.std` says how far its numbers spread. `_td[:, 100] < _mc[:, 100]` is `True` in the runs where TD(0) is closer, and `.sum()` counts the `True`s. Run the cell.
    """)
    return


@app.cell
def example_numpy():
    import numpy as np

    _td, _mc = C.course_errors()        # shape (100, 101): a row per run, a column per number of walks
    _curve_td = np.mean(_td, axis=0)    # the average of every column: the curve of the play block
    _curve_mc = np.mean(_mc, axis=0)
    print("shape:", _td.shape)
    print("after 10 walks, TD(0):", round(float(_curve_td[10]), 3), "Monte Carlo:", round(float(_curve_mc[10]), 3))
    print("after 100 walks, TD(0):", round(float(_curve_td[100]), 3), "Monte Carlo:", round(float(_curve_mc[100]), 3))
    print("spread after 100 walks, TD(0):", round(float(np.std(_td[:, 100])), 3), "Monte Carlo:", round(float(np.std(_mc[:, 100])), 3))
    print("runs in which TD(0) is closer after 100 walks:", int((_td[:, 100] < _mc[:, 100]).sum()))
    return (np,)


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    After 100 walks the errors of the runs spread 0.023 for TD(0) and 0.069 for Monte Carlo, three times as much, and TD(0) is closer in 96 of the 100 runs. After 10 walks it is closer in 67: early on, a lucky or unlucky walk decides a lot for both.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your functions on 100 runs

    The demo runs the 100 runs of the play block with your `td_update` and `mc_update`, on the same walks. Where your function does not pass its check yet, the course's takes its place. Pick $\eta$ for each method: with 0.1 for both you see the play block again; then try a smaller $\eta$ for Monte Carlo, such as 0.02, and 0.05 for TD(0), then 0.01 for both. The chart shows the average distance to the true values after every walk, TD(0) solid and Monte Carlo dashed; beside it, the estimates of run 1 as they are now.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    eta_td = mo.ui.dropdown(options=C.eta_options(L.lang), value=C.eta_text(0.1, L.lang), label=C.DEMO_TEXT[L.lang]["eta_td"])
    eta_mc = mo.ui.dropdown(options=C.eta_options(L.lang), value=C.eta_text(0.1, L.lang), label=C.DEMO_TEXT[L.lang]["eta_mc"])
    play_button = mo.ui.run_button(label="Run 100 runs with my td_update and mc_update")
    mo.hstack([eta_td, eta_mc, play_button], justify="start", gap=1.5, wrap=True)
    return eta_mc, eta_td, play_button


@app.cell(hide_code=True)
def demo_play(box_mc, box_td, eta_mc, eta_td, play_button, result_mc, result_td):
    walked = None
    if not play_button.value:
        _out = mo.md("*Pick η for each method, then press the button to let both learn from the walks.*")
    else:
        _mine = [_n for _n, _r in (("td_update", result_td), ("mc_update", result_mc)) if _r.state == "correct"]
        _td = run_code(box_td.value).names["td_update"] if "td_update" in _mine else C.td_update
        _mc = run_code(box_mc.value).names["mc_update"] if "mc_update" in _mine else C.mc_update
        _w = C.DEMO_TEXT[L.lang]
        _now = {}

        def _progress(_n, _run):
            if "run" not in _now:
                _now["chart"].watch(lambda: C.now_picture(_now["run"], _now["n"], L.lang))
            _now["run"], _now["n"] = _run, _n
            _now["chart"].reference([(_k, _run["mc"][_k]) for _k in range(_n + 1)], "", role="loss", replace=True)
            _now["chart"].extend([(_k, _run["td"][_k]) for _k in ([0, 1] if _n == 1 else [_n])], f"**Walk {_n} of {C.N_WALKS}**")

        _live = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"], "alt": _w["alt"]}, C.N_WALKS, role="loss",
                       low=0.0, high=0.25, window=1, key=C.demo_key(eta_td.value, eta_mc.value, L.lang))
        _live.line("**Walking** …")
        _now["chart"] = _live
        _r = C.run(_td, _mc, eta_td.value, eta_mc.value, show=_progress)
        if _r == "td":
            _out = _live.stopped("*Your `td_update` gave an error during the run, or left `V` without a number for every state. Does it still pass its check?*")
        elif _r == "mc":
            _out = _live.stopped("*Your `mc_update` gave an error during the run, or left `V` without a number for every state. Does it still pass its check?*")
        elif _r == "slow":
            _out = _live.stopped("*The run took more than a minute. Does your function do more work than it needs?*")
        else:
            walked = _r
            _theirs = [_n for _n in ("td_update", "mc_update") if _n not in _mine]
            if len(_mine) == 2:
                _who = "your `td_update` and your `mc_update`"
            elif len(_mine) == 1:
                _who = f"your `{_mine[0]}` and the course's `{_theirs[0]}`"
            else:
                _who = "the course's `td_update` and `mc_update`, since yours do not pass their checks yet"
            _out = mo.vstack([
                L.explore("rewind", "walks", lambda _v: C.rewind_frame(_r, int(_v), L.lang), start=C.N_WALKS,
                          values=C.LOOKS, role="",
                          task="Go back to 0, then forward again: after how many walks does the solid line of TD(0) pull away from the dashed one, and how do the estimates of run 1 move on the way?"),
                mo.md(f"Run with {_who}, $\\eta$ = {C.eta_text(eta_td.value, 'en')} for TD(0) and {C.eta_text(eta_mc.value, 'en')} for Monte Carlo. After 100 walks the estimates of TD(0) are on average {C.number(_r['td'][C.N_WALKS], 3, 'en')} from the true values over the 100 runs, and those of Monte Carlo {C.number(_r['mc'][C.N_WALKS], 3, 'en')}."),
            ])
    _out
    return (walked,)


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With the course's functions and $\eta$ = 0.02 for Monte Carlo, its estimates jump around much less, and it ends at 0.076 after 100 walks; TD(0) with $\eta$ = 0.05 ends at 0.031. A smaller $\eta$ trades speed for calm, for both methods: with 0.01 for both, Monte Carlo is even ahead after 100 walks, 0.088 against 0.113, since TD(0) then carries the news back very slowly. But with each method at its own best $\eta$ from the menu, TD(0) wins on this walk, 0.031 against 0.076. Nobody has proved that TD learning always needs fewer episodes than Monte Carlo, but on tasks with random outcomes, such as this walk, it usually does.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_predictor():
    mo.md(r"""
    ### Exercise 3. The same episodes, two answers

    An agent has seen only these eight episodes, with $\gamma$ = 1: once A, reward 0, then B, reward 0, and the end; six times B, reward 1, and the end; once B, reward 0, and the end. Both methods estimate $V(B)$ = 0.75: six of the eight episodes from B ended with 1. Monte Carlo estimates $V(A)$ = 0: the one return after A was 0. Now the agent goes over the eight episodes again and again with TD(0) and a small $\eta$, until the estimates stop moving.
    """)
    return


@app.cell(hide_code=True)
def choice_predictor_cell():
    choice_predictor = L.choice(
        "predictor",
        question="Where does $V(A)$ end with TD(0)?",
        options={
            "zero": "At 0, as with Monte Carlo: the one return after A was 0, and TD(0) learns that too in the end",
            "half": "At 0.375, halfway between the return of 0 and $V(B)$, since TD(0) weighs both",
            "right": "At 0.75: A always led to B with reward 0, so TD(0) moves $V(A)$ toward $0 + V(B)$",
            "one": "At 1, since B mostly ends with 1",
        },
        correct="right",
        why={
            "right": "TD(0) only ever sees one step from A: to B, with reward 0. Its target for A is $0 + V(B)$, and $V(B)$ learns from all eight episodes. Monte Carlo fits the one return it saw after A; TD(0) fits what the steps say about each other: if A is followed by B again, B is worth 0.75 by everything the agent saw. That makes TD's estimate the better guess for new episodes, as long as the task works the way the steps suggest.",
            "zero": "The one return after A was 0, but TD(0) never looks at returns. What is the target of TD(0) for the step from A, and where does it end?",
            "half": "TD(0) has one target for a step, not two to weigh: for the step from A to B, what is it?",
            "one": "The target of TD(0) for A is the reward of the step plus $V(B)$: how large are those two?",
        },
    )
    choice_predictor
    return (choice_predictor,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Learning from whole games: Monte Carlo prediction](/course/open/rl-monte-carlo-prediction?lang=en), you estimated $V$ from whole episodes. TD(0) estimates the same values after every step, by bootstrapping: its target is the reward plus its own estimate of the next state, as the target of [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en) was, and as [What is a policy worth? Policy evaluation](/course/open/rl-policy-evaluation?lang=en) did with every possible next state and the table `T`. TD(0) only says what a fixed policy is worth. The next lesson, *Careful or bold: SARSA and Q-learning*, uses TD to choose: SARSA learns $Q(s, a)$ after every step and improves the policy as it goes, and you hold it against Q-learning on the edge of a cliff.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_predictor, result_mc, result_td):
    L.wrap_up(result_td, result_mc, choice_predictor)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
