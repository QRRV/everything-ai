# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-conditioning. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the course's dog and cats, and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-conditioning", "en")
    C = L.checks   # the course's dog and cats in the puzzle box, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Which machine pays most? The multi-armed bandit](/course/open/rl-bandits?lang=en) and [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_explore():
    quick_explore = L.choice(
        "quick-explore",
        question="A greedy agent and an ε-greedy agent with ε = 0.1 play the same slot machines. Over the last 100 pulls, the ε-greedy agent earns more per pull. Why?",
        options={
            "luck": "Its random pulls happen to land on good machines",
            "right": "It keeps trying other machines, so it finds the best one more often",
            "never": "A greedy agent never pulls a machine twice",
            "more": "It pulls more often than the greedy agent, so it collects more rewards in the same 100 steps",
        },
        correct="right",
        why={
            "right": "A greedy agent sticks with the first machine that looked good: in the course it tried 3.3 machines on average. With ε = 0.1, one pull in ten goes to a random machine, so the agent finds out which one really pays most: 1.36 per pull over the last 100, against 1.04. In this lesson a cat explores without an ε.",
            "luck": "A random pull lands on any machine, the bad ones as well. What does the agent find out by pulling other machines now and then? See [Which machine pays most? The multi-armed bandit](/course/open/rl-bandits?lang=en).",
            "never": "A greedy agent pulls the machine with the largest estimate, again and again. What does it miss by never trying the others? See [Which machine pays most? The multi-armed bandit](/course/open/rl-bandits?lang=en).",
            "more": "Both agents pull once per step. What does the ε-greedy agent do with one pull in ten? See [Which machine pays most? The multi-armed bandit](/course/open/rl-bandits?lang=en).",
        },
    )
    quick_explore
    return (quick_explore,)


@app.cell(hide_code=True)
def quick_check_update():
    quick_update = L.choice(
        "quick-update",
        question="In Q-learning, Q(s, a) is 2, the target r + γ max Q(s′, a′) is 4, and η = 0.5. What is Q(s, a) after the update?",
        options={
            "right": "3: halfway from 2 to 4",
            "target": "4: the target itself",
            "whole": "6: Q plus the whole target",
            "away": "1: Q moved away from the target",
        },
        correct="right",
        why={
            "right": "The update moves Q a part η of the way to the target: 2 + 0.5 × (4 − 2) = 3. In this lesson the same update follows what a dog expects, with food as the target.",
            "target": "With η = 1, Q would jump to the target. What does η = 0.5 do with the step from 2 to 4? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
            "whole": "Q gets η times the difference between the target and Q, not the whole target. What is 0.5 × (4 − 2)? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
            "away": "The difference between the target and Q is 4 − 2 = 2, so Q goes up. How far, with η = 0.5? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
        },
    )
    quick_update
    return (quick_update,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Reinforcement learning did not start with computers. Around 1900, psychologists studied how animals learn from food. Ivan Pavlov, who won the Nobel Prize in 1904 for research on digestion, made a sound, such as a metronome or a buzzer, just before a dog got food. Once this had happened often enough, the sound alone made the dog's mouth water. In 1898, Edward Thorndike put hungry cats in puzzle boxes, with food outside in sight. A cat got out by pulling a loop of cord, pressing a lever or stepping on a platform. The first time, it scratched, bit and pushed until it did the right thing by chance. After that it got out faster and faster: not in one sudden moment of insight, but gradually. In the English translation of Pavlov's lectures, *Conditioned Reflexes* (1927), the food that follows the sound *reinforces* the reflex, and reinforcement learning took its name from this psychology. The two experiments are still the two halves of RL: predicting what is coming, and choosing what to do. In this lesson you build a model of each in a few lines of Python.

    **An everyday example.** A dog hears the rustle of its food bag and runs to its bowl, before there is any food in it: the sound predicts food. The same dog gets a treat every time it sits when asked, and soon sits more often: the action brings food. The first is what Pavlov studied, the second what Thorndike studied.
    """)
    return


@app.cell(hide_code=True)
def predict_trial():
    guess_trial = L.guess(
        "trial",
        question="The course's cat in a puzzle box can do 6 things, such as scratch the bars or meow; only one of them, pulling the cord, opens the door. Each action has a strength, 1 at the start, and the cat picks an action with a chance in proportion to its strength: at first each of the 6 has the same chance, so the cat needs 6 tries on average to get out. Every time it gets out, the strength of the cord grows by 1. How many tries does the cat need on average the 20th time it is put in the box, do you think?",
        options={
            "one": "1: after 19 escapes the cat knows that the cord opens the door",
            "right": "About 1.25",
            "half": "About 3, half of the first time",
            "six": "About 6: picking at random stays picking at random",
        },
        reveal="**About 1.25.** After 19 escapes the cord has strength 20, against 5 for the other actions together, so the cat pulls it with a chance of 20 out of 25 and needs 25 / 20 = 1.25 tries on average. The 1,000 cats of the course need 5.88 tries on average the first time and 1.27 the twentieth. Yet the cat never knows anything: one try in five still goes to scratching, biting or meowing.",
    )
    mo.vstack([mo.Html(C.cage_picture(L.lang)), guess_trial])
    return (guess_trial,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Classical conditioning.** In Pavlov's experiment, the food comes after the sound whatever the dog does. The dog learns what the sound predicts. A model needs one number for that: V, how much food the sound predicts, 0 at the start. After every trial, V takes a step toward the food f that came, 1 for food and 0 without:
        """),
        L.formula(r"[[param:V]] \leftarrow [[param:V]] + [[hyper:\eta]] \left( [[data:f]] - [[param:V]] \right)"),
        mo.md(r"""
        This is the update of Q-learning, with the food as the target and no next state. The difference f − V is how far off the prediction was, and η says how big a part of it V takes. With η = 0.2, V goes from 0 to 0.2, then 0.36 and 0.49: big steps while the prediction is far off, smaller ones as it comes closer. When the food stops coming, V drops back the same way. Pavlov saw that too: a sound that keeps coming without food loses its effect, which Pavlov called *extinction*.
        """),
        mo.Html(C.pavlov_picture(L.lang)),
        mo.md(r"""
        **Instrumental conditioning.** In Thorndike's puzzle box, the food comes only after the right action. The cat learns what to do. Thorndike captured what the cats did in the **law of effect** (1911): "Of several responses made to the same situation, those which are accompanied or closely followed by satisfaction to the animal will, other things being equal, be more firmly connected with the situation". In short: an action that brings something good becomes more likely in the same situation. Later psychologists also call this kind of learning *operant conditioning*.

        A model of the cat: every action gets a strength, 1 at the start, and the cat picks an action with a chance in proportion to its strength, its strength divided by the strengths of all actions together. Every time the cat gets out, the law of effect adds a gain, 1 here, to the strength of the action that opened the door. An action with chance p takes 1/p tries on average, just as a die takes 6 throws on average to show a six. At first the cord has a chance of 1/6, so 6 tries; after k escapes it has strength 1 + k against 5 for the others, so the cat needs (6 + k)/(1 + k) tries on average.
        """),
        mo.Html(C.cat_picture(L.lang)),
        mo.md(r"""
        | | Classical conditioning | Instrumental conditioning |
        |---|---|---|
        | The food comes | after the signal, whatever the animal does | only after the right action |
        | The animal learns | what the signal predicts | what to do |
        | In the model | a prediction V | a strength for every action |
        | In RL | a value, such as V(s) or Q(s, a) | a policy |

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **classical conditioning** | | learning what a signal predicts, from a reward that follows the signal whatever the animal does | `predictions` |
        | **instrumental conditioning** | | learning which action to take, from a reward that follows only the right action | `choose` |
        | **law of effect** | | an action that is followed by something good becomes more likely in the same situation | `strength[C.CORD] + C.GAIN` |
        """),
        L.watch_out("In classical conditioning the food comes anyway: the dog's watering mouth changes nothing about the food. The dog does not choose; it only predicts. So the model of the dog has one number and no actions."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    A model of each kind of conditioning:

    1. Classical: start with V = 0. After every trial, V ← V + η(f − V), with f = 1 when food came and 0 when it did not.
    2. Instrumental: give every action a strength of 1.
    3. Let the animal pick actions, each with a chance of its strength divided by all strengths together, until it picks the action that brings the reward. Count the tries.
    4. The law of effect: add the gain to the strength of that action. Then the next trial, from step 3.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell follows the course's cat through its first 10 trials, without drawing any actions: it computes what happens on average. `np.ones(C.ACTIONS)` makes an array of 6 ones, as `np.zeros` makes one of zeros: the strengths at the start. `C.CORD` is 5, the place of the cord in the array, the last one. Every trial, the cell prints the chance of the cord, its strength divided by the total of all strengths, and the tries the cat needs on average, 1 divided by that chance; then the law of effect adds `C.GAIN`, 1, to the strength of the cord. Run the cell.
    """)
    return


@app.cell
def example_strengths():
    import numpy as np

    strength = np.ones(C.ACTIONS)   # every action starts with strength 1
    for _trial in range(1, 11):
        _chance = round(float(strength[C.CORD] / strength.sum()), 2)
        _tries = round(float(strength.sum() / strength[C.CORD]), 2)   # 1 / chance
        print("trial", _trial, "chance of the cord", _chance, "tries on average", _tries)
        strength[C.CORD] = strength[C.CORD] + C.GAIN   # the law of effect: the cord opened the door
    print("strengths after 10 escapes:", strength)
    return np, strength


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The chance of the cord climbs from 0.17 to 0.67 in 10 trials, and the tries on average drop from 6 to 1.5: fast at first, then more and more slowly. The first escape takes the cat from 6 tries to 3.5, the ninth from 1.56 to 1.5. These are averages. A cat that really draws its actions can be lucky or unlucky in any trial, and in exercise 2 your cats draw them.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_predictions():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Pavlov's dog

        `predictions(food, eta)` follows the dog's V over a run of trials and gives back V before every trial. `food` is a list with a number per trial, 1 when food came after the sound and 0 when it did not; `eta` is the learning rate. `v` starts at `0.0`, and `seen` collects the values of V: in the loop, `seen.append(v)` stores V before the trial, and the next line updates `v` with the food `f` of that trial. Replace the `...` with the new V.

        The check calls `predictions(pavlov, 0.2)`. `pavlov` is a list of 40 trials: 20 with food after the sound, then 20 without, as in the picture in *The idea*. Then three other cases: `pavlov` with η 0.5; `alternate`, food after every other sound for 20 trials, with η 0.2; and `[1, 0, 1]` with η 0.5.
        """),
        L.task("Make `predictions` move `v` a part `eta` of the way to the food `f` of every trial."),
    ])
    return


@app.cell(hide_code=True)
def exercise_predictions():
    box_predictions = L.code_box("predictions", lines=7)
    box_predictions
    return (box_predictions,)


@app.cell(hide_code=True)
def check_predictions(box_predictions):
    result_predictions = L.check("predictions", box_predictions.value)
    result_predictions
    return (result_predictions,)


@app.cell(hide_code=True)
def hints_predictions():
    L.hints(
        "predictions",
        [
            "Step 1 of the recipe: the new `v` is the old `v` plus a step toward the food `f` of this trial, and the step is `eta` times the difference between the food and what V predicted. Which of `f` and `v` comes first in that difference, so that V moves toward the food?",
            "In the first trial with food, `v` is 0 and `f` is 1, so the new `v` should be 0 + 0.2 × 1 = 0.2. In the first trial without food, after 20 with, `v` is 0.99 and `f` is 0, so `v` should drop by 0.2 × 0.99. Put the difference in parentheses, so that `eta` multiplies all of it.",
            "`        v = v + eta * ___`",
        ],
        why="`v = v + eta * (f - v)` moves V a fifth of the way to the food after every trial: up toward 1 with food, down toward 0 without. It is the update of Q-learning, with the food as the target.",
    )
    return


@app.cell(hide_code=True)
def text_after_predictions():
    mo.md(r"""
    In the picture, your V over `pavlov`. It climbs fast at first and slowly near 1, and once the food stops it falls back the same way. Each step is η times how far off the prediction was: food that the sound already predicts well changes V a little, and a surprise changes it a lot. The next lesson, *Learning from surprise: the Rescorla-Wagner model*, builds on exactly that, with more than one signal at a time.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_choose():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Thorndike's cat

        `choose(strength, rng)` picks the action the cat takes, as a number: 0 for the first action, up to 5 for the cord. `strength` is a NumPy array with the strength of every action, and `rng` is a random number generator. `rng.choice(len(strength), p=chances)` draws one of the numbers 0 to `len(strength) - 1`, each with its chance from `chances`: with `p=`, `rng.choice` takes the chances, which must add up to 1. Replace the `...` with `chances`: the chance of every action, its strength divided by the strengths of all actions together.

        The check calls `trials(choose, 0)`: the course's function `trials` puts a cat in the box 20 times, with a random number generator seeded with 0, and lets your `choose` pick every action; after every escape, the law of effect adds 1 to the strength of the cord. `trials` gives back the tries of every trial. Then three other cases: seeds 1 and 2, and seed 3 with a cat of 4 actions, `trials(choose, 3, actions=4)`.
        """),
        L.task("Make `chances` hold the chance of every action, in proportion to its strength."),
    ])
    return


@app.cell(hide_code=True)
def exercise_choose():
    box_choose = L.code_box("choose", lines=3)
    box_choose
    return (box_choose,)


@app.cell(hide_code=True)
def check_choose(box_choose):
    result_choose = L.check("choose", box_choose.value)
    result_choose
    return (result_choose,)


@app.cell(hide_code=True)
def hints_choose():
    L.hints(
        "choose",
        [
            "Step 3 of the recipe: `chances` should hold the chance of every action, `strength` divided by one number, so that the chances add up to 1. Which number is that? NumPy divides a whole array by a number in one go, as `strength / 2` halves every strength.",
            "With strengths 1, 1, 1, 1, 1 and 6, the chances should be 1/11 five times and 6/11 for the cord: each strength divided by 11, all strengths added up. In the worked example, the chance of the cord was divided by that total too.",
            "`    chances = ___`",
        ],
        why="`chances = strength / strength.sum()` divides every strength by the total of all strengths: the chances add up to 1, and a twice as strong action gets twice the chance. `strength / np.sum(strength)` says the same.",
    )
    return


@app.cell(hide_code=True)
def text_after_choose():
    mo.md(r"""
    In the picture, the mean of 100 cats that choose with your `choose`, seeds 0 to 99, against the tries on average from the chances alone, (6 + k)/(1 + k). The mean stays near the dashed line, while one cat alone jumps up and down: with seed 0, the cat needs 6 tries the first time, then 2, 2, 1, 2, 2, 2, 1, 2 and 3. Thorndike's cats, too, got faster gradually, not all at once. Nothing in the cat ever understands the box: a bigger strength only makes the cord a bit more likely, every time.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### A hundred cats in the box

    The demo puts 100 cats in the box 20 times, seeds 0 to 99, with your `choose` if it passes its check, and with the course's otherwise. Choose how many actions a cat has, 4, 6 or 10, and the gain: how much stronger the law of effect makes the cord after every escape. The chart shows the mean tries of the 100 cats in every trial, and dashed the tries on average from the chances alone. Beside it, the strengths of cat 1. Afterwards you can go back through the trials.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    actions_choice = mo.ui.dropdown(options=C.action_options(L.lang), value=C.number(6, 0, L.lang), label=C.DEMO_TEXT[L.lang]["actions"])
    gain_choice = mo.ui.dropdown(options=C.gain_options(L.lang), value=C.number(1, 1, L.lang), label=C.DEMO_TEXT[L.lang]["gain"])
    cats_button = mo.ui.run_button(label="Put 100 cats in the box")
    mo.hstack([actions_choice, gain_choice, cats_button], justify="start", gap=1.5, wrap=True)
    return actions_choice, cats_button, gain_choice


@app.cell(hide_code=True)
def demo_cats(actions_choice, box_choose, cats_button, gain_choice, result_choose):
    if not cats_button.value:
        _out = mo.md("*Choose the actions and the gain, then press the button to put 100 cats in the box.*")
    else:
        _mine = result_choose.state == "correct"
        _choose = run_code(box_choose.value, C.C_NAMES).names["choose"] if _mine else C.choose
        _actions, _gain = actions_choice.value, gain_choice.value
        _w = C.DEMO_TEXT[L.lang]
        _live = L.live(C.chart_labels(L.lang), C.TRIALS, role="ink", low=0, high=C.demo_high(_actions), window=1,
                       xticks=[1, 5, 10, 15, 20])
        _live.reference(C.chance_line(_actions, _gain), _w["chance"], role="ink-2")

        def _progress(_t, _run):
            _live.watch(lambda: C.now_picture(_run, L.lang))
            _live.extend([(_t, _run.means()[-1])], _w["learning"].format(t=_t, n=C.TRIALS))

        _run = C.demo_run(_choose, _actions, _gain, show=_progress, pause=0.15)
        if _run == "error":
            _out = _live.stopped("*Your `choose` gave an error, or something other than an action. Does it still pass its check?*")
        else:
            _out = mo.vstack([
                L.explore("rewind", "trial", lambda _v: C.rewind_frame(_run, int(_v), L.lang), start=C.TRIALS,
                          values=list(range(C.TRIALS + 1)), role="",
                          task="Go back to 0, then forward again: how does the bar of the cord grow, and how much do the tries drop with each trial?"),
                mo.md(C.verdict(_run, _mine, L.lang)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With 10 actions, the first trial takes about 10 tries, and with a gain of 2 the line drops faster: after k escapes, a cat with n actions and gain g needs (n + k·g)/(1 + k·g) tries on average. However big the gain, the line does not reach 1 in 20 trials, since the other actions keep their strength of 1: with 4 actions and a gain of 2, a cat still needs 1.08 tries on average in the 20th trial, and with 10 actions and a gain of 0.5 it needs 1.86.

    The cat is an RL agent without a single number about the future: its chances are its policy, and the law of effect is how that policy learns. It explores without an ε. Every action keeps a chance above 0, so the cat goes on trying the others now and then, less and less as the cord gets stronger.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_kind():
    mo.md(r"""
    ### Exercise 3. Which kind of conditioning?

    Four kinds of learning. In three of them a signal comes first, and what follows comes whatever the learner does; in one, the reward comes only after the learner's own action.
    """)
    return


@app.cell(hide_code=True)
def choice_kind_cell():
    choice_kind = L.choice(
        "kind",
        question="Which of these is instrumental conditioning?",
        options={
            "bag": "A dog runs to its bowl when it hears the food bag rustle",
            "right": "A rat presses a lever more often once each press brings a pellet of food",
            "drill": "A person's shoulders tense at the sound of a dentist's drill",
            "bowl": "A cat's mouth waters at the sight of its bowl, which always comes just before its food",
        },
        correct="right",
        why={
            "right": "Pressing the lever is an action, and food comes only after it: the law of effect makes pressing more likely. In the other three, a signal comes first, and what follows comes whatever the animal does: classical conditioning.",
            "bag": "The food comes whether the dog runs or not; the rustle only predicts it. Which of the four gets its reward only after its own action?",
            "drill": "The drill comes whether the shoulders tense or not; the sound only predicts it. Which of the four gets its reward only after its own action?",
            "bowl": "The food comes whatever the cat does; the bowl only predicts it. Which of the four gets its reward only after its own action?",
        },
    )
    choice_kind
    return (choice_kind,)


@app.cell(hide_code=True)
def text_exercise_link():
    mo.md(r"""
    ### Exercise 4. The dog in RL

    Pavlov's dog learned what a sound predicts, with the update of Q-learning.
    """)
    return


@app.cell(hide_code=True)
def choice_link_cell():
    choice_link = L.choice(
        "link",
        question="Which part of an RL agent matches classical conditioning?",
        options={
            "policy": "The policy: which action to take in a state",
            "right": "A value: how much reward a state predicts",
            "explore": "Exploration: trying other actions now and then, even when one of them already looks best",
            "discount": "The discount: rewards later on count for less",
        },
        correct="right",
        why={
            "right": "The dog learns what the sound predicts, as V(s) learns what a state predicts, with the same update: a step toward what came. Thorndike's cat matches the policy: which action to take.",
            "policy": "The dog does not choose anything: the food comes anyway. What does the dog learn about the sound?",
            "explore": "The dog has no actions to try. What does it learn about the sound?",
            "discount": "The model of the dog has one step: the sound, then food or none. What does V learn about the sound?",
        },
    )
    choice_link
    return (choice_link,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Which machine pays most? The multi-armed bandit](/course/open/rl-bandits?lang=en) an agent learned which machine pays most; here a cat learned which action opens the door, with a rule that psychology wrote down in 1911. Pavlov's dog learns a prediction with the update of Q-learning; Thorndike's cat learns a policy with the law of effect. RL brings the two together: an agent learns values, and chooses its actions with them. The next lesson, *Learning from surprise: the Rescorla-Wagner model*, gives the dog more than one signal, and shows that a signal learns only as far as the food is a surprise.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_kind, choice_link, result_choose, result_predictions):
    L.wrap_up(result_predictions, result_choose, choice_kind, choice_link)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
