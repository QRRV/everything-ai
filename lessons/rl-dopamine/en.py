# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-dopamine. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the course's trials with a light and juice, and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-dopamine", "en")
    C = L.checks   # the course's trials with a light and juice, the TD model of conditioning, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Learning from surprise: the Rescorla-Wagner model](/course/open/rl-rescorla-wagner?lang=en) and [Learning after every step: TD(0)](/course/open/rl-td-prediction?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_surprise():
    quick_surprise = L.choice(
        "quick-surprise",
        question="In the Rescorla-Wagner model, the sound has V = 0.8 and the light V = 0.2. Both come, and food follows. What is the surprise of this trial?",
        options={
            "light": "0.8: the food minus the V of the light",
            "right": "0: together the two predict all the food",
            "food": "1: the food came, and that is the surprise",
            "each": "0.2 for the sound and 0.8 for the light: every signal has a surprise of its own",
        },
        correct="right",
        why={
            "right": "Together the two predict 0.8 + 0.2 = 1, so the surprise is 1 − 1 = 0, and neither V moves. In this lesson the surprise is worked out at every moment within a trial, not once per trial.",
            "light": "The surprise is one number for the whole trial: the food minus what the signals present predict together. What do 0.8 and 0.2 add up to? See [Learning from surprise: the Rescorla-Wagner model](/course/open/rl-rescorla-wagner?lang=en).",
            "food": "The surprise is the food minus what was predicted, not the food alone. What do the sound and the light predict together? See [Learning from surprise: the Rescorla-Wagner model](/course/open/rl-rescorla-wagner?lang=en).",
            "each": "In the Rescorla-Wagner model, the signals present share one surprise. What is 1 − (0.8 + 0.2)? See [Learning from surprise: the Rescorla-Wagner model](/course/open/rl-rescorla-wagner?lang=en).",
        },
    )
    quick_surprise
    return (quick_surprise,)


@app.cell(hide_code=True)
def quick_check_td():
    quick_td = L.choice(
        "quick-td",
        question="In TD(0), V(S) = 0.9, the reward R is 0, γ = 1, and the next state has V(S′) = 0. What is the TD error δ?",
        options={
            "nothing": "0: no reward came, so there is nothing to compare",
            "old": "0.9: how far off the old value was",
            "right": "−0.9: the target 0 minus the old 0.9",
            "next": "1: the next state is worth less than this one, by one whole reward",
        },
        correct="right",
        why={
            "right": "δ = R + γV(S′) − V(S) = 0 + 1 × 0 − 0.9 = −0.9: the target is below the old value, so the error is below 0. In this lesson, that is what happens when juice that was expected stays away.",
            "nothing": "Without a reward, the target R + γV(S′) can still differ from V(S). What is 0 + 1 × 0 − 0.9? See [Learning after every step: TD(0)](/course/open/rl-td-prediction?lang=en).",
            "old": "The TD error is the target minus the old value, so its sign matters. Which is larger here, the target or V(S)? See [Learning after every step: TD(0)](/course/open/rl-td-prediction?lang=en).",
            "next": "δ = R + γV(S′) − V(S). What do you get with R = 0, V(S′) = 0 and V(S) = 0.9? See [Learning after every step: TD(0)](/course/open/rl-td-prediction?lang=en).",
        },
    )
    quick_td
    return (quick_td,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Dopamine is a substance that some cells in the brain use to pass on a signal, and it is often called the pleasure chemical. In the 1980s and 1990s, the neuroscientist Wolfram Schultz measured single dopamine neurons in monkeys while they learned that a small light means a drop of fruit juice, a little more than a second later. The neurons did not simply fire at the juice. At first, before the monkeys knew what the light meant, they fired a burst when the juice came. Once the monkeys had learned, the burst came at the light, and the juice itself got no response. And when the juice stayed away after the light, the neurons went quiet, below their usual rate, exactly at the moment the juice should have come. In 1997, Schultz, the theorist Peter Dayan and the neuroscientist Read Montague showed that this is the error of a TD model that learns to predict the juice: the same kind of error as in *Learning after every step: TD(0)*. An idea from reinforcement learning turned out to describe what cells in a brain do, and it is still one of the clearest links between AI and neuroscience. In this lesson you build that model in two short functions, and watch its error move from the juice to the light.

    **An everyday example.** A café gives you a free biscuit with your coffee, without saying so beforehand. The first time, the biscuit is a pleasant surprise. After a few weeks, the pleasant moment comes when you walk in, since you know a biscuit is coming; the biscuit itself is just what you expected. And on the day the biscuit stays away, you notice at once, although a coffee without a biscuit was perfectly normal before.
    """)
    return


@app.cell(hide_code=True)
def predict_shift():
    guess_shift = L.guess(
        "shift",
        question="The course's model sees a light, and 4 steps later juice comes, in 60 trials in a row, with η = 0.2. In trial 1, its error is 1 at the moment of the juice and 0 everywhere else. Where is its error largest in trial 60, do you think?",
        options={
            "juice": "At the juice, just as in trial 1",
            "right": "At the light, 4 steps before the juice",
            "half": "At the light and at the juice, about half each",
            "none": "Nowhere: after 60 trials the model predicts everything, so no moment is a surprise any more",
        },
        reveal="**At the light, 4 steps before the juice.** After 60 trials, the V's from the light up to the step before the juice are almost 1: from the light on, the model expects juice. When the juice comes, the V drops from 1 to 0, and the juice of 1 makes up for that: no error. The error of 1 sits at the light, the first moment that says juice is coming. Before the light the model predicted 0, since nothing announces when the light comes, so the light itself stays a surprise. Schultz saw the same in the dopamine neurons of monkeys.",
    )
    guess_shift
    return (guess_shift,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A V for every step.** The Rescorla-Wagner model made one prediction per trial. To say *when* something is a surprise, a model has to predict at every moment within a trial. The course counts the moments of a trial in steps from the light: step 0 is the moment the light comes on, and the juice comes at step 4. Every step gets its own V: the juice that is still to come after that step, 0 at the start. Before the light, the prediction is 0, since nothing announces when the light will come. This is the TD model of conditioning of Richard Sutton and Andrew Barto, which Read Montague, Peter Dayan and Terrence Sejnowski used for dopamine in 1996, with a separate V for each step after the light.

        **The error at every step.** At step t, the model compares what it has now with what the step before predicted:
        """),
        L.formula(r"[[loss:\delta_t]] = [[data:r_t]] + [[hyper:\gamma]] [[param:V_t]] - [[param:V_{t-1}]]"),
        mo.md(r"""
        Here $r_t$ is the juice that comes at step t, 1 or 0, and γ is the discount factor. Put together, $r_t + \gamma V_t$ is what step t says about the juice: what came now, plus what is still to come. $V_{t-1}$ is what the step before said. This is the TD error of *Learning after every step: TD(0)*, δ = R + γV(S′) − V(S), with S the step before and S′ step t. In this lesson γ is 1, until the demo: within a trial of a few seconds, the course does not discount. To learn, the step before takes a step of η times the error:
        """),
        L.formula(r"[[param:V_{t-1}]] \leftarrow [[param:V_{t-1}]] + [[hyper:\eta]] \, [[loss:\delta_t]]"),
        mo.md(r"""
        **Three trials.** Follow the course's model with η = 0.2. In trial 1, all V's are 0. At the light nothing changes: 0 + 0 − 0 = 0. At step 4 the juice comes: 1 + 0 − 0 = 1, a surprise. Step 3, the step before, learns from it, and in the trials after that the steps before step 3 learn from step 3: trial by trial, the error moves back. By trial 60, the V's from the light to step 3 are almost 1. Now the error sits at the light: 0 + 1 − 0 = 1, the jump from the 0 before the light to the 1 after it. At the juice it is 1 + 0 − 1 = 0: the juice was expected. In trial 61 the juice stays away, and at step 4 the error is 0 + 0 − 1 = −1.
        """),
        mo.Html(C.trials_picture(L.lang)),
        mo.md(r"""
        **What the neurons do.** Put the three trials next to the recordings of Schultz, and they match: a burst at the juice in the first trials, a burst at the light once the monkeys had learned, and a dip below the usual rate where the juice should have come. Dopamine neurons fire a few times per second even when nothing happens, so they can signal an error below 0 by firing less. Schultz, Dayan and Montague (1997) concluded that dopamine neurons send a reward prediction error: the reward that came plus what is now expected, minus what was expected before. The juice itself does not make them fire: once it is predicted, the monkey drinks the same juice and the neurons stay quiet. Dopamine is not the pleasure of the juice; it is the surprise about it.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **dopamine** | | a substance with which some cells in the brain pass on a signal; dopamine neurons fire like the error of a TD model that predicts reward | |
        | **reward prediction error** (RPE) | | the reward that came plus what is now expected, minus what was expected before; dopamine neurons seem to send it | `errors` |
        | $\delta_t$ | delta t | the error at step t of a trial: $r_t + \gamma V_t - V_{t-1}$ | `deltas[t]` |
        """),
        L.watch_out("The error at step t belongs to the prediction of the step before: it says how far off $V_{t-1}$ was. So when the juice comes at step 4, it is $V_3$ that learns, not $V_4$."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    One trial of the TD model of conditioning:

    1. Count the steps from the light, step 0, and give every step a V, 0 at the start. Before the light, the prediction is 0.
    2. At every step t, work out the error: $\delta_t = r_t + \gamma V_t - V_{t-1}$, with the juice $r_t$ of that step.
    3. Every step except the last takes a step of η times the error of the step after it: $V_{t-1} \leftarrow V_{t-1} + \eta \, \delta_t$. The last step has no step after it, and its V stays 0.
    4. Then the next trial, from step 2.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell takes the V's that the course's model has after many trials, 1 from the light to step 3 and 0 after that, and works out the error at the first six steps with γ = 1: once with the juice at step 4, and once without it. `old` is the V of the step before, 0 at step 0, since nothing is predicted before the light. Run the cell.
    """)
    return


@app.cell
def example_errors():
    import numpy as np

    learned = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0])   # V of every step after many trials
    juice = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])     # the juice comes at step 4
    for t in range(6):
        if t == 0:
            old = 0   # before the light, nothing is predicted
        else:
            old = learned[t - 1]
        print("step", t, "with juice", juice[t] + learned[t] - old, "without", learned[t] - old)
    return juice, learned, np, old, t


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    At step 0 the V jumps from the 0 before the light to 1: an error of 1, with or without juice, since the light comes in both. From step 1 to step 3 the V stays 1 and nothing comes: no error. At step 4 the V drops to 0. With juice, the 1 that comes makes up for the drop, and the error is 0; without juice, the drop is all there is, and the error is −1.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_errors():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The error at every step

        `errors(V, rewards, gamma)` gives back the error δ at every step of one trial. `V` is a NumPy array with the V of every step, from the light on; `rewards` is an array with the juice at every step; `gamma` is γ. `np.zeros(len(V))` makes `deltas`, an array with a 0 for every step, and the loop `for t in range(len(V)):` fills it in one step at a time. `before` holds the V of the step before: 0 before the light, and the last line of the loop moves it on to `V[t]`. Replace the `...` with the error at step t.

        The check calls `errors(np.array([0.5, 0.6, 0.7, 0.8, 0.0, 0.0]), np.array([0, 0, 0, 0, 1, 0]), 1)`: a trial of six steps in which the model has learned part of the way, with the juice at step 4. Then five other cases: the learned V's with and without juice, V's of 0, the first case with γ 0.9, and a trial with twice the juice at step 2.
        """),
        L.task("Make `deltas[t]` the error at step t."),
    ])
    return


@app.cell(hide_code=True)
def exercise_errors():
    box_errors = L.code_box("errors", lines=7)
    box_errors
    return (box_errors,)


@app.cell(hide_code=True)
def check_errors(box_errors):
    result_errors = L.check("errors", box_errors.value)
    result_errors
    return (result_errors,)


@app.cell(hide_code=True)
def hints_errors():
    L.hints(
        "errors",
        [
            "Step 2 of the recipe: `deltas[t]` is the juice of step `t`, which is in `rewards`, plus `gamma` times the V of step `t`, minus `before`, the V of the step before.",
            "In the check's first case, step 1 has V 0.6, the step before has 0.5, and no juice comes: 0 + 1 × 0.6 − 0.5 = 0.1. At step 0, `before` is still 0, so the error is the V of the light itself: 0.5.",
            "`        deltas[t] = ___`",
        ],
        why="`deltas[t] = rewards[t] + gamma * V[t] - before` is $\\delta_t = r_t + \\gamma V_t - V_{t-1}$: what step t says about the juice, minus what the step before said. With the learned V's, it gives 1 at the light, 0 at the juice, and −1 there when the juice stays away.",
    )
    return


@app.cell(hide_code=True)
def text_after_errors():
    mo.md(r"""
    The picture shows your `errors` on the three trials of the idea. In trial 1 the error stands at the juice, in trial 60 at the light, and in trial 61 the bar at step 4 points down: the juice that the step before promised did not come.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_learn():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Learning from the error

        `learn(V, rewards, eta, gamma)` does one trial of learning and gives back the new V's. Its first line works out `deltas` with `errors`; the box uses the course's `errors`, which does what yours does. Within one trial, every V is used once before it changes, so the errors of the old V's are the ones to learn from. `new = np.zeros(len(V))` makes an array for the new V's, and `for t in range(len(V) - 1):` goes over every step except the last: the last step has no step after it, and its V stays 0. `eta` is the learning rate η. Replace the `...` with the error that `V[t]` learns from.

        The check calls `learn(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.array([0, 0, 0, 0, 1, 0]), 0.2, 1)`: the first trial, from V's of 0, with the juice at step 4. Then four other cases: the V's that learned part of the way; the learned V's in a trial without juice; η 0.5 with γ 0.9; and a trial with twice the juice at step 2, with η 0.1 and γ 0.8.
        """),
        L.task("Make every `new[t]` learn from the right error."),
    ])
    return


@app.cell(hide_code=True)
def exercise_learn():
    box_learn = L.code_box("learn", lines=6)
    box_learn
    return (box_learn,)


@app.cell(hide_code=True)
def check_learn(box_learn):
    result_learn = L.check("learn", box_learn.value)
    result_learn
    return (result_learn,)


@app.cell(hide_code=True)
def hints_learn():
    L.hints(
        "learn",
        [
            "Step 3 of the recipe: `new[t]` is `V[t]` plus `eta` times the error that says how far off `V[t]` was. `deltas` holds the error of every step. The error at step t compares step t with the step before, so which step's error is about the prediction of step `t`?",
            "In the first trial, only the juice at step 4 is a surprise: δ is 1 at step 4. It is the step before, step 3, that should learn: `new` should become 0.2 at step 3 and stay 0 at step 4.",
            "`        new[t] = V[t] + eta * ___`",
        ],
        why="`new[t] = V[t] + eta * deltas[t + 1]` is $V_{t-1} \\leftarrow V_{t-1} + \\eta \\, \\delta_t$, written one step later: the error at step t + 1 says how far off the prediction of step t was. Over 60 trials, the error moves from the juice to the light.",
    )
    return


@app.cell(hide_code=True)
def text_after_learn():
    mo.md(r"""
    The picture shows the error at every step after your `learn` learned for that many trials. In trial 1 it stands at the juice. In the next trials it spreads to the steps before the juice, since every step learns from the step after it, and moves step by step to the light; by trial 60 only the light is left. The last row, trial 61 without juice, has the dip at step 4.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### The error moves, trial by trial

    The demo runs 60 trials with your `learn` if it passes its check, and with the course's otherwise, with η 0.2, and then one trial without juice. Choose at which step after the light the juice comes, 2, 4 or 6, and the discount factor γ. The chart shows the error at the moment of the juice in every trial, and dashed the error at the light; beside it, the V's and the errors of the last trial. Afterwards you can go back through the trials.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    delay_choice = mo.ui.dropdown(options=C.delay_options(L.lang), value=C.number(4, 0, L.lang), label=C.DEMO_TEXT[L.lang]["delay"])
    gamma_choice = mo.ui.dropdown(options=C.gamma_options(L.lang), value=C.number(1, 0, L.lang), label=C.DEMO_TEXT[L.lang]["gamma"])
    run_button = mo.ui.run_button(label="Run the trials")
    mo.hstack([delay_choice, gamma_choice, run_button], justify="start", gap=1.5, wrap=True)
    return delay_choice, gamma_choice, run_button


@app.cell(hide_code=True)
def demo_trials(box_learn, delay_choice, gamma_choice, result_learn, run_button):
    if not run_button.value:
        _out = mo.md("*Choose the step of the juice and γ, then press the button to run the trials.*")
    else:
        _mine = result_learn.state == "correct"
        _learn = run_code(box_learn.value, C.learn_names()).names["learn"] if _mine else C.learn
        _delay, _gamma = delay_choice.value, gamma_choice.value
        _n = C.TRIALS + 1
        _w = C.DEMO_TEXT[L.lang]
        _live = L.live(C.chart_labels(L.lang), _n, role="loss", low=-1, high=1, window=1, xticks=C.demo_xticks(),
                       key=C.chart_key(L.lang))

        def _progress(_t, _demo):
            _live.watch(lambda: C.now_picture(_demo, L.lang))
            _live.reference(_demo.points(0), "", role="loss", replace=True)
            _live.extend([(_t, float(_demo.deltas[_t - 1, _demo.delay]))], _w["learning"].format(t=_t, n=_n))

        _demo = C.demo_run(_learn, _delay, _gamma, show=_progress, pause=0.08)
        if _demo == "error":
            _out = _live.stopped("*Your `learn` gave an error, or something other than a number for every step. Does it still pass its check?*")
        else:
            _out = mo.vstack([
                L.explore("rewind", "trial", lambda _v: C.rewind_frame(_demo, int(_v), L.lang), start=_n,
                          values=list(range(_n + 1)), role="",
                          task="Go to trial 1, then forward: at which step is the error largest in trial 10, and in trial 20?"),
                mo.md(C.verdict(_demo, _mine, L.lang)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With γ 1, the error at the light ends at 0.99 or more after 60 trials, wherever the juice comes; the later the juice, the more trials the error needs to travel back to the light. With γ below 1, juice further away counts for less, so the light predicts less of it: with γ 0.9, the error at the light ends at 0.81 with the juice at step 2, at 0.66 at step 4 and at 0.52 at step 6, and with γ 0.8 at 0.64, 0.41 and 0.26. Dopamine neurons do something like this: Shunsuke Kobayashi and Schultz (2008) found that their burst at a signal is smaller the longer the juice takes to follow it. In every setting, the trial without juice drops to −1 at the moment of the juice. In the model, the error travels back to the light step by step; in recordings this was hard to see for a long time, until a study in mice found such a gradual shift in 2022 (Ryunosuke Amo and colleagues).
    """)
    return


@app.cell(hide_code=True)
def text_exercise_more():
    mo.md(r"""
    ### Exercise 3. More juice than expected

    The course's model has learned in 60 trials that the light means juice at step 4.
    """)
    return


@app.cell(hide_code=True)
def choice_more_cell():
    choice_more = L.choice(
        "more",
        question="In trial 61, twice as much juice comes as usual: r = 2 at step 4. What is the error at step 4?",
        options={
            "none": "0: the juice came at the moment the model expected it",
            "right": "1: one more than the step before predicted",
            "all": "2: all the juice that came",
            "dip": "−1: the juice came, but not the amount that was promised",
        },
        correct="right",
        why={
            "right": "$\\delta_4 = r_4 + \\gamma V_4 - V_3$ = 2 + 0 − 1 = 1: the error is how much more came than was predicted, not how much came. Dopamine neurons do the same: more juice than predicted gives a burst, less gives a dip (Philippe Tobler, Christopher Fiorillo and Schultz, 2005). The error at the light stays 1, since the light still predicts the usual juice.",
            "none": "The juice came on time, but was it the amount that the step before predicted? What is 2 + 0 − 1?",
            "all": "The error is what came minus what was predicted. What did step 3 predict?",
            "dip": "An error below 0 means that less came than was predicted. Is 2 less or more than the 1 that step 3 predicted?",
        },
    )
    choice_more
    return (choice_more,)


@app.cell(hide_code=True)
def text_exercise_unblock():
    mo.md(r"""
    ### Exercise 4. Blocking in the brain

    In blocking, from *Learning from surprise: the Rescorla-Wagner model*, a sound already predicts the food when a light joins it, and the light learns almost nothing. Pascale Waelti, Anthony Dickinson and Schultz (2001) found that dopamine neurons in monkeys hardly fire at such a blocked signal, and not at the food either, since it was expected.
    """)
    return


@app.cell(hide_code=True)
def choice_unblock_cell():
    choice_unblock = L.choice(
        "unblock",
        question="In an experiment with rats (Elizabeth Steinberg and colleagues, 2013), dopamine neurons were made to fire at the moment of the food, in the trials in which the light joined the sound. What did the rats learn about the light?",
        options={
            "right": "That the light means food: the burst worked as a surprise",
            "nothing": "Nothing: the sound already predicted the food",
            "sound": "The sound lost its V, since the burst came after it as well",
            "below": "A V below 0 for the light: the burst came at a moment the model did not expect, so it counts against the light",
        },
        correct="right",
        why={
            "right": "The burst at the food is the error signal, so making the neurons fire there works as if the food were a surprise after all: the light learned that food follows it, although the sound still predicted the food. The neurons were switched on with light through a thin fiber, a technique called optogenetics. It is one of the strongest signs that the brain does not only make a reward prediction error, but also learns from it.",
            "nothing": "Without the burst, the light would learn nothing, as in blocking. What does a burst at the food usually tell the brain?",
            "sound": "The burst came at the food, as after a surprise. Which signal still had something to learn?",
            "below": "The burst came at the food, the moment a surprise would come. Does a burst mean more or less than was predicted?",
        },
    )
    choice_unblock
    return (choice_unblock,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Learning from surprise: the Rescorla-Wagner model](/course/open/rl-rescorla-wagner?lang=en), signals learned from the surprise of a whole trial. The TD model of conditioning makes that surprise a TD error at every step, as in [Learning after every step: TD(0)](/course/open/rl-td-prediction?lang=en), and that error moves from the juice to the light while the model learns, just as the bursts of dopamine neurons do. The next lesson, *RL outside games*, leaves the brain and looks at where RL is used in the world.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_more, choice_unblock, result_errors, result_learn):
    L.wrap_up(result_errors, result_learn, choice_more, choice_unblock)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
