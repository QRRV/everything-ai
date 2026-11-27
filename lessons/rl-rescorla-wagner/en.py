# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-rescorla-wagner. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the course's dog with a sound and a light, and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-rescorla-wagner", "en")
    C = L.checks   # the course's dog with a sound and a light, its experiments, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Pavlov's dogs and Thorndike's cats: conditioning](/course/open/rl-conditioning?lang=en) and [Learning after every step: TD(0)](/course/open/rl-td-prediction?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_dog():
    quick_dog = L.choice(
        "quick-dog",
        question="For Pavlov's dog, the sound has V = 0.5, η = 0.2, and food comes after the sound. What is V after this trial?",
        options={
            "plus": "0.7: V plus η",
            "right": "0.6: a fifth of the way from 0.5 to 1",
            "jump": "1: V jumps to the food at once",
            "step": "0.1: only the step, so the dog forgets the old V",
        },
        correct="right",
        why={
            "right": "V takes a step of η times how far off the prediction was: 0.5 + 0.2 × (1 − 0.5) = 0.6. In this lesson a light comes with the sound, and the two signals share that difference.",
            "plus": "Adding η makes V grow by 0.2 every trial, past 1 and on. How big a part of the difference between the food and V does V take? See [Pavlov's dogs and Thorndike's cats: conditioning](/course/open/rl-conditioning?lang=en).",
            "jump": "With η = 1, V would jump to the food. What does η = 0.2 do with the step from 0.5 to 1? See [Pavlov's dogs and Thorndike's cats: conditioning](/course/open/rl-conditioning?lang=en).",
            "step": "The step is 0.2 × (1 − 0.5) = 0.1, and it is added to the old V. What is 0.5 + 0.1? See [Pavlov's dogs and Thorndike's cats: conditioning](/course/open/rl-conditioning?lang=en).",
        },
    )
    quick_dog
    return (quick_dog,)


@app.cell(hide_code=True)
def quick_check_td():
    quick_td = L.choice(
        "quick-td",
        question="In TD(0), V(S) = 0.5, V(S′) = 0.8, the reward R is 0 and γ = 1. What is the TD error δ?",
        options={
            "next": "0.8: the value of the next state",
            "sign": "−0.3: the old value minus the target",
            "right": "0.3: the target 0.8 minus the old 0.5",
            "reward": "0: no reward came, so nothing changes, whatever the values of the two states",
        },
        correct="right",
        why={
            "right": "δ = R + γV(S′) − V(S) = 0 + 1 × 0.8 − 0.5 = 0.3: the target minus the old value. In this lesson the same kind of difference shows up in a model of animals from 1972, with the food as the target.",
            "next": "V(S′) is part of the target R + γV(S′). What is left once the old value V(S) is taken off? See [Learning after every step: TD(0)](/course/open/rl-td-prediction?lang=en).",
            "sign": "The TD error is the target minus the old value, so the order matters. What is 0.8 − 0.5? See [Learning after every step: TD(0)](/course/open/rl-td-prediction?lang=en).",
            "reward": "Without a reward, the target R + γV(S′) can still differ from V(S): the next state may be worth more. What is 0 + 1 × 0.8 − 0.5? See [Learning after every step: TD(0)](/course/open/rl-td-prediction?lang=en).",
        },
    )
    quick_td
    return (quick_td,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The model of Pavlov's dog in *Pavlov's dogs and Thorndike's cats: conditioning* had one signal, a sound, and one number V for what it predicts. Real experiments often have more than one signal at a time: a sound and a light, a smell and a place. In 1969, the psychologist Leon Kamin trained rats with a noise followed by a short electric shock, until the rats were afraid of the noise alone. Then the noise came together with a light, followed by the same shock. Tested with the light alone, the rats showed almost no fear: the light had come before the shock many times, and yet the rats had learned almost nothing about it. Kamin called this *blocking*: what the noise already predicted, the light could not learn. In 1972, Robert Rescorla and Allan Wagner explained it with one rule: a signal learns only as far as the outcome is a surprise, and all signals present share that one surprise. Their rule became one of the best-known models in psychology. It also turned out to be the update with which gradient descent fits a linear model, and the update that TD learning spreads out over time. In this lesson you build it in two short functions.

    **An everyday example.** A dog knows that the doorbell means a visitor, and runs to the door when it rings. Then a lamp is put in the hall that lights up together with the bell. Months later the bell breaks, and only the lamp lights up when a visitor presses the button. The dog hardly looks up: the bell had always said it all, so the lamp never had anything left to teach.
    """)
    return


@app.cell(hide_code=True)
def predict_blocking():
    guess_blocking = L.guess(
        "blocking",
        question="The course's dog hears a sound before food in 20 trials, with η = 0.2, as in the model of Pavlov's dog. Then, for 20 more trials, a light comes on together with the sound, and food follows as before. Both signals start at V = 0. How much food does the light predict after these 40 trials, do you think, from 0 for none to 1 for food every time?",
        options={
            "half": "About 0.5: the light shares the food with the sound",
            "right": "Almost nothing: about 0.01",
            "full": "About 1: the light came before food 20 times, as often as the sound did at first",
            "less": "About 0.3: less than the sound, since it came later",
        },
        reveal="**Almost nothing: about 0.01.** After the 20 trials with the sound alone, the sound predicts 0.99 of the food. When the light joins, the two together already predict 0.99, so the food is hardly a surprise: 1 − 0.99 = 0.01. The light learns from that small surprise only, and after 20 trials it predicts 0.006. A dog that gets the sound and the light together from the start learns 0.50 for each. This is what Kamin found: the sound blocks the light.",
    )
    mo.vstack([mo.Html(C.plan_picture(L.lang)), guess_blocking])
    return (guess_blocking,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **One V per signal, one surprise for all.** The Rescorla-Wagner model gives every signal its own V: how much food that signal predicts, 0 at the start. In a trial, the signals present predict the food together: their V's added up, written ΣV, with the Greek capital sigma for adding up, as in the mean squared error. The surprise is the food f that came, 1 or 0, minus that prediction. Then every signal present takes the same step, η times the surprise, and a signal that is absent keeps its V:
        """),
        L.formula(r"[[param:V_i]] \leftarrow [[param:V_i]] + [[hyper:\eta]] \left( [[data:f]] - \Sigma [[param:V]] \right) \quad \text{for every signal } i \text{ present}"),
        mo.md(r"""
        With one signal, ΣV is the V of that signal, and the rule is the update of Pavlov's dog. With two, the signals compete for what there is to learn: once the sound predicts the food, the surprise is gone for the light too. Rescorla and Wagner gave every signal and every kind of food a rate of its own, α and β; the course takes them together as one learning rate η. They wrote λ for the food; here it is f, as in the model of the dog.
        """),
        mo.Html(C.shared_picture(L.lang)),
        mo.md(r"""
        **Blocking.** Follow the course's dog with η = 0.2. After 20 trials with the sound alone, V of the sound is 0.99. Then the light joins: together the two predict 0.99 + 0 = 0.99, the surprise is 0.01, and each takes a step of 0.2 × 0.01 = 0.002. The surprise only shrinks from there, so after 20 trials the light predicts 0.006. A dog that gets the sound and the light together from the start sees a surprise of 1 in the first trial; the two share every step, and after 20 trials each predicts 0.50. The picture shows both experiments.
        """),
        mo.Html(C.runs_picture(L.lang)),
        mo.md(r"""
        **A surprise below 0.** When the food stays away while the signals present predicted some, the surprise is below 0, and their V's go down. Let the sound alone bring food, and the sound with the light bring none, over and over: the sound climbs toward 1, and the light drops below 0, to −0.81 after 20 trials of each kind, so that the two together predict almost nothing. A V below 0 is a prediction that the food will not come.

        **The same rule elsewhere.** In *Learning after every step: TD(0)*, V(S) took a step toward the target R + γV(S′) after every step. The Rescorla-Wagner model makes the same move once per trial: the food f is the target, and ΣV plays the role of V(S). Richard Sutton and Andrew Barto built on this. Their TD model of classical conditioning (1987) predicts at every moment within a trial, not only once per trial, so that the timing of the signals matters, as it does for real animals. And with ΣV written as `present @ v`, the dot product of an array with a 1 for every signal present and the array of V's, the rule is a step of gradient descent for a linear model, with half the squared error as its loss: `present` is the input, `v` holds the parameters, and the food is the label. Sutton and Barto pointed this out in 1981: it is the rule that Bernard Widrow and Ted Hoff used in 1960 to train a simple electronic neuron.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **Rescorla-Wagner model** | | a model of conditioning in which every signal present takes the same step, η times the surprise: the food minus what all signals present predict together | `rw_step` |
        | **blocking** | | a signal that comes with another signal that already predicts the food learns almost nothing, since the food is hardly a surprise any more | `run(blocking, 0.2)` |
        | $\Sigma V$ | sigma V | what the signals present predict together: their V's added up | `present @ v` |
        """),
        L.watch_out("The surprise is one number for the whole trial, not one per signal. A signal does not learn from how far off its own V was: it learns from how far off all signals present were together. That is what makes blocking possible."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    One trial of the Rescorla-Wagner model:

    1. Give every signal a V, 0 at the start.
    2. In a trial, add up the V's of the signals present: ΣV, what they predict together.
    3. The surprise is the food minus that prediction: f − ΣV, with f = 1 when food came and 0 when it did not.
    4. Every signal present takes a step of η times the surprise; the others keep their V. Then the next trial, from step 2.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell follows the course's dog through the experiment of the guess, and prints the first three trials with the light. `v` holds the V of the sound and of the light, in that order. `sound` and `both` are arrays with a 1 for every signal present. `sound @ v` is the dot product, 1 × V of the sound + 0 × V of the light: what the signals present predict together. Multiplying the step by `sound` gives the light a step of 0, since it is absent. Run the cell.
    """)
    return


@app.cell
def example_blocking():
    import numpy as np

    v = np.zeros(2)            # V of the sound and of the light
    sound = np.array([1, 0])   # only the sound
    both = np.array([1, 1])    # the sound and the light together
    for _trial in range(20):
        v = v + 0.2 * (1 - sound @ v) * sound   # food came: f = 1
    print("after 20 trials with the sound alone")
    print("sound", round(float(v[0]), 4), "light", round(float(v[1]), 4))
    for _trial in range(21, 24):
        _surprise = 1 - both @ v
        v = v + 0.2 * _surprise * both
        print("trial", _trial, "surprise", round(float(_surprise), 4), "V", np.round(v, 4))
    return both, np, sound, v


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The sound reaches 0.9885 in 20 trials, and the light gets 0.0023 from its first trial: a fifth of a surprise of 0.0115. Every trial shrinks the surprise further, to 0.0069 and 0.0042, so the light gains less and less. Both V's still grow, but only by what is left to learn.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rw_step():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. One trial

        `rw_step(v, present, food, eta)` does one trial of the Rescorla-Wagner model and gives back the new V's. `v` is a NumPy array with the V of every signal; `present` is an array with a 1 for every signal present in the trial and a 0 for every signal that is absent; `food` is f, 1 when food came and 0 when it did not; `eta` is the learning rate. The line with `return` already does step 4 of the recipe: `eta * surprise * present` is 0 for an absent signal. Replace the `...` with `surprise`: the food minus what the signals present predict together.

        The check calls `rw_step(np.array([0.5, 0.0]), np.array([1, 1]), 1, 0.2)`: the sound predicts 0.5, the light 0, both are present, and food comes. Then four other cases: V 0.99 and 0, both present, with food; the sound alone with food, while the light has V −0.3; both present without food, with η 0.5; and three signals, of which the first and the last are present.
        """),
        L.task("Make `surprise` the food minus what the signals present predict together."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rw_step():
    box_rw_step = L.code_box("rw_step", lines=3)
    box_rw_step
    return (box_rw_step,)


@app.cell(hide_code=True)
def check_rw_step(box_rw_step):
    result_rw_step = L.check("rw_step", box_rw_step.value)
    result_rw_step
    return (result_rw_step,)


@app.cell(hide_code=True)
def hints_rw_step():
    L.hints(
        "rw_step",
        [
            "Steps 2 and 3 of the recipe: `surprise` is the food minus ΣV, the V's of the signals present added up. In the worked example, `sound @ v` gave that prediction for a trial with the sound alone. Which array says here which signals are present?",
            "With `v` at 0.5 and 0 and both signals present, the two predict 1 × 0.5 + 1 × 0 = 0.5 together, and the surprise is 1 − 0.5 = 0.5. Use `food`, not 1, so that a trial without food works too.",
            "`    surprise = ___`",
        ],
        why="`surprise = food - present @ v` is the food minus what the signals present predict together: `present @ v` adds up the V's of the signals present, since an absent signal counts 0 times. `food - np.dot(present, v)` says the same.",
    )
    return


@app.cell(hide_code=True)
def text_after_rw_step():
    mo.md(r"""
    In the picture, your `rw_step` in two trials. From V 0.5 and 0, the surprise is 0.5, and both signals take a step of 0.1. From V 0.99 and 0, the sound already predicts the food: the surprise is 0.01, and the light gets 0.002. The light is just as present in both trials; what differs is how much of the food was still a surprise.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_run():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A whole experiment

        `run(trials, eta)` follows the V's of the sound and the light over the trials of an experiment, and gives back a NumPy array with a row per moment: the V's at the start, and after every trial. `trials` is a list of trials, each a pair of `present` and `food`. `for present, food in trials:` gives the two parts of each pair a name of their own, as `x, y = point` does for a tuple. `v` starts as `np.zeros(2)`, `seen.append(v)` keeps the V's after every trial in the list `seen`, and `np.array(seen)` turns that list of arrays into one array with a row for each. Replace the `...` with the new `v`: one trial of the Rescorla-Wagner model, with `rw_step`. The box uses the course's `rw_step`, which does what yours does.

        The check calls `run(blocking, 0.2)`. `blocking` is the experiment of the guess: 20 trials with the sound alone and food, then 20 with the sound and the light together and food. Then four other cases: `control`, 20 trials without signals or food, then the same 20 with the sound and the light; `inhibition`, 20 times a trial with the sound alone and food followed by a trial with the sound and the light and no food; `blocking` with η 0.5; and `short`, three trials, with η 0.3.
        """),
        L.task("Make every trial update `v` with `rw_step`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_run():
    box_run = L.code_box("run", lines=7)
    box_run
    return (box_run,)


@app.cell(hide_code=True)
def check_run(box_run):
    result_run = L.check("run", box_run.value)
    result_run
    return (result_run,)


@app.cell(hide_code=True)
def hints_run():
    L.hints(
        "run",
        [
            "Step 4 of the recipe, for every trial: the new `v` is what `rw_step` gives for the V's so far, the signals `present`, the `food` and the learning rate `eta`. In which order does `rw_step` take these four?",
            "In the first trial of `blocking`, `present` is the sound alone and `food` is 1, so the V's should go from 0 and 0 to 0.2 and 0. Give `rw_step` the V's of the trial before, not a new array of zeros.",
            "`        v = ___`",
        ],
        why="`v = rw_step(v, present, food, eta)` does one trial from the V's so far, and `seen.append(v)` then keeps them. Over `blocking`, the light ends at 0.006: the sound blocks it.",
    )
    return


@app.cell(hide_code=True)
def text_after_run():
    mo.md(r"""
    In the picture, your `run` over `blocking` and `control`. In `blocking`, the line of the light stays flat on 0 while the sound learns, and hardly moves once the light comes: the sound had taken everything. In `control`, the two lines lie on top of each other and stop at 0.50: the same food, shared. In `inhibition`, the light ends at −0.81: once the sound predicts food, the light learns that the food stays away when it comes.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Blocking, trial by trial

    The demo runs the experiment of the guess with your `run` if it passes its check, and with the course's otherwise. Choose in how many trials the sound comes alone before the light joins, 0, 5, 10 or 20, and the learning rate. After those trials, the light comes with the sound for 20 trials, always followed by food. The chart shows V of the light, and dashed V of the sound; beside it, the V's after the last trial and its surprise. Afterwards you can go back through the trials.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    first_choice = mo.ui.dropdown(options=C.first_options(L.lang), value=C.number(20, 0, L.lang), label=C.DEMO_TEXT[L.lang]["first"])
    eta_choice = mo.ui.dropdown(options=C.eta_options(L.lang), value=C.number(0.2, 1, L.lang), label=C.DEMO_TEXT[L.lang]["eta"])
    run_button = mo.ui.run_button(label="Run the experiment")
    mo.hstack([first_choice, eta_choice, run_button], justify="start", gap=1.5, wrap=True)
    return eta_choice, first_choice, run_button


@app.cell(hide_code=True)
def demo_blocking(box_run, eta_choice, first_choice, result_run, run_button):
    if not run_button.value:
        _out = mo.md("*Choose the trials with the sound alone and the learning rate, then press the button to run the experiment.*")
    else:
        _mine = result_run.state == "correct"
        _run = run_code(box_run.value, C.run_names()).names["run"] if _mine else C.run
        _first, _eta = first_choice.value, eta_choice.value
        _n = _first + C.PHASE
        _w = C.DEMO_TEXT[L.lang]
        _live = L.live(C.chart_labels(L.lang), _n, role="param", low=0, high=1, window=1, xticks=C.demo_xticks(_first),
                       key=C.chart_key(L.lang))

        def _progress(_t, _demo):
            _live.watch(lambda: C.now_picture(_demo, L.lang))
            _live.reference(_demo.points(C.SOUND), "", role="param", replace=True)
            _live.extend([(_t, float(_demo.values[_t, C.LIGHT]))], _w["learning"].format(t=_t, n=_n))

        _demo = C.demo_run(_run, _first, _eta, show=_progress, pause=0.08)
        if _demo == "error":
            _out = _live.stopped("*Your `run` gave an error, or something other than a row of two V's for every moment. Does it still pass its check?*")
        else:
            _out = mo.vstack([
                L.explore("rewind", "trial", lambda _v: C.rewind_frame(_demo, int(_v), L.lang), start=_n,
                          values=list(range(_n + 1)), role="",
                          task="Go to the trial where the light joins, then forward: how big is the surprise there, and how much does the light learn from it?"),
                mo.md(C.verdict(_demo, _mine, L.lang)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With η 0.2, the light ends at 0.006 after 20 trials of the sound alone, at 0.054 after 10, at 0.16 after 5, and at 0.50 without them, when the sound and the light share the food. The longer the sound has been learning, the less surprise is left for the light. A larger η lets the sound learn faster, so it blocks more: with η 0.5 and 20 trials, the sound predicts the food completely before the light comes, and the light ends at 0.0000005. With η 0.1, the sound is still at 0.88 when the light joins, and the light gets 0.060.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_latent():
    mo.md(r"""
    ### Exercise 3. What the model misses

    Every model leaves something out. Here the light comes many times before any food.
    """)
    return


@app.cell(hide_code=True)
def choice_latent_cell():
    choice_latent = L.choice(
        "latent",
        question="Before any trial with food, the course's dog sees the light 50 times, and nothing follows. What does the Rescorla-Wagner model say the light predicts after these 50 trials?",
        options={
            "right": "0: the surprise is 0 − 0 every time",
            "below": "Below 0: the light learns that no food follows it",
            "up": "A bit above 0: the light was seen 50 times",
            "half": "0.5: the model does not know yet",
        },
        correct="right",
        why={
            "right": "With the light alone and no food, the surprise is 0 − 0 = 0, so V of the light never moves. Real animals do learn something here: a signal seen often without anything following is learned more slowly later, once food does follow it. Psychologists call this latent inhibition (Lubow and Moore, 1959), and the Rescorla-Wagner model cannot explain it: one of its known limits.",
            "below": "A V goes below 0 when food was predicted and stayed away. What did the dog predict before these trials, and what came?",
            "up": "Seeing the light changes its V only through the surprise. What is the food minus ΣV when no food comes and V is 0?",
            "half": "V of the light starts at 0. What does the surprise of each trial do to it?",
        },
    )
    choice_latent
    return (choice_latent,)


@app.cell(hide_code=True)
def text_exercise_td():
    mo.md(r"""
    ### Exercise 4. The same rule in TD learning

    The Rescorla-Wagner model and TD(0) both move a prediction toward a target.
    """)
    return


@app.cell(hide_code=True)
def choice_td_cell():
    choice_td = L.choice(
        "td",
        question="In TD(0), V(S) takes a step toward R + γV(S′). Which part of the Rescorla-Wagner model plays the role of R + γV(S′)?",
        options={
            "lr": "η, how big a step the V's take",
            "right": "f, the food that comes at the end of the trial",
            "together": "ΣV, what the signals present predict together, before the food comes",
            "present": "Which signals are present in the trial",
        },
        correct="right",
        why={
            "right": "Both rules move a prediction toward a target. In TD(0) the target is R + γV(S′), after every step; in the Rescorla-Wagner model it is f, the food, once per trial, and ΣV plays the role of V(S). The surprise f − ΣV is the TD error of a trial of one step, with nothing after it.",
            "lr": "η is the learning rate in both rules: how far the prediction moves. What does the prediction move toward?",
            "together": "ΣV is the prediction itself, as V(S) is in TD(0). What does the prediction move toward?",
            "present": "The signals present say which V's take the step. What do they step toward?",
        },
    )
    choice_td
    return (choice_td,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Pavlov's dogs and Thorndike's cats: conditioning](/course/open/rl-conditioning?lang=en), one sound learned what it predicts. With more signals, the Rescorla-Wagner model lets them share one surprise, and that explains blocking: a signal learns only what the others do not already predict. The same surprise, what came minus what was predicted, drives the TD error of [Learning after every step: TD(0)](/course/open/rl-td-prediction?lang=en) and the update of gradient descent for a linear model. The next lesson, *Dopamine as a prediction error*, shows that cells in the brain seem to signal exactly such a surprise.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_latent, choice_td, result_run, result_rw_step):
    L.wrap_up(result_rw_step, result_run, choice_latent, choice_td)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
