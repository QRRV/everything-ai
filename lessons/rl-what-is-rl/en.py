# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-what-is-rl. Route: course/route/lessen/overige-takken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("rl-what-is-rl", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Kinds of learning](/course/open/ml-kinds-of-learning?lang=en) and [Repeating: for loops](/course/open/py-for?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kind():
    quick_kind = L.choice(
        "quick-kind",
        question="A robot learns to walk. Nobody shows it how, but it gets a point for every metre it moves forward without falling. Which kind of learning is that?",
        options={
            "supervised": "Supervised learning: the points are its labels",
            "right": "Reinforcement learning: the points are rewards",
            "unsupervised": "Unsupervised learning: there are no right answers",
        },
        correct="right",
        why={
            "supervised": "A label is the right answer for an example, written down beforehand. Nobody wrote down the right step; the points come after the robot acts. See [Kinds of learning](/course/open/ml-kinds-of-learning?lang=en).",
            "unsupervised": "Unsupervised learning finds groups or patterns without any feedback; here the robot acts and gets points for it. See [Kinds of learning](/course/open/ml-kinds-of-learning?lang=en).",
            "right": "The robot acts, and the points it gets for what it does are rewards.",
        },
    )
    quick_kind
    return (quick_kind,)


@app.cell(hide_code=True)
def quick_check_loop():
    quick_loop = L.choice(
        "quick-loop",
        question="What does this program show?\n\n```python\nw = 1\nfor k in range(3):\n    w = w * 0.5\nprint(w)\n```",
        options={
            "right": "0.125",
            "plus": "2.5",
            "once": "0.5",
        },
        correct="right",
        why={
            "plus": "`w * 0.5` halves `w`; it does not add 0.5 to it. See [Repeating: for loops](/course/open/py-for?lang=en).",
            "once": "The loop runs three times, for `k` = 0, 1 and 2, and every time it halves `w` again. See [Repeating: for loops](/course/open/py-for?lang=en).",
            "right": "Three times halved: 1, then 0.5, 0.25 and 0.125.",
        },
    )
    quick_loop
    return (quick_loop,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In supervised learning, people write down the right answer for every example. For many tasks nobody can: which move is right in this chess position, how hard should a robot leg push right now, how much should a heating system heat at 7:14? Afterwards, though, you can say how well it went: a game won, a metre walked, a warm house for little energy. Reinforcement learning learns from exactly that, a number after every step that says how good it was.

    **An everyday example.** A dog learns to sit. You cannot tell it in words what you want. But when it sits, it gets a treat. After a while it sits more often, because sitting pays off.
    """)
    return


@app.cell(hide_code=True)
def predict_credit():
    guess_credit = L.guess(
        "credit",
        question="A program plays chess. It gets a reward only at the end: +1 if it wins and −1 if it loses, and 0 after every move before that. What do you think it can learn about its first move?",
        options={
            "nothing": "Nothing: the first move always got 0",
            "right": "Something, from the reward at the end",
            "all": "Everything: the first move decides the game",
        },
        reveal="**Something, from the reward at the end.** The program does not look only at the reward right after a move, but at all the rewards that follow it: the return of *The idea*. So the win at the end also counts for the first move. But which of its 40 moves won the game? That is the hard part of reinforcement learning, called *credit assignment*. Over many games, the program finds out which moves tend to come before a win.",
    )
    guess_credit
    return (guess_credit,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        In reinforcement learning, an **agent** learns by acting. Everything outside it is the **environment**. They take turns, step after step:

        1. the agent sees the situation it is in and chooses an **action**;
        2. the environment changes, and gives back the new situation and a **reward**: a number that says how good that step was.

        The rewards are the only feedback: nobody says which action would have been right. The *reward hypothesis* says that every goal can be put this way: get as much reward as possible, in total.

        The agent does not try to get the largest next reward, but the largest **return**: the sum of all the rewards from now on. Later rewards count less. For a game of $T$ steps, with reward $r_1$ after the first step, $r_2$ after the second, and so on:
        """),
        L.formula(r"[[loss:G_0]] = [[data:r_1]] + [[hyper:\gamma]] [[data:r_2]] + [[hyper:\gamma]]^2 [[data:r_3]] + \dots + [[hyper:\gamma]]^{T-1} [[data:r_T]]"),
        mo.md(r"""
        The **discount factor** $\gamma$ is a number from 0 to 1. Every step further away multiplies a reward by $\gamma$ once more: with $\gamma$ = 0.5, a reward one step later counts half, two steps later a quarter. With $\gamma$ close to 1 the agent is patient: rewards far away still count. With $\gamma$ = 0 only the next reward counts. Discounting also fits the idea that a reward now is surer than one far ahead, and with $\gamma$ below 1 it keeps the return from growing without end in a task that never stops, such as the heating.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **agent** | | the learner in reinforcement learning: it chooses actions and gets rewards | |
        | **environment** | | everything outside the agent: it takes the action, and gives back the new situation and a reward | |
        | **action** | | one choice of the agent at a step, such as a move in a game | |
        | **reward** | | a number from the environment after every step: how good that step was | `r` |
        | **return** | | the sum of the rewards from now on, later ones discounted: what the agent tries to make as large as possible | `G` |
        | **discount factor** | | a number from 0 to 1: a reward one step later counts $\gamma$ times as much | `gamma` |
        | $G_0$ | G zero | the return from the start | `G` |
        | $r_1, r_2, \dots$ | r one, r two | the rewards after step 1, step 2, and so on | `rewards` |
        | $\gamma$ | gamma | the discount factor | `gamma` |
        """),
        L.watch_out("*Agent* and *return* mean other things elsewhere. Here an agent is the learner in reinforcement learning; an *LLM agent* is a language model that calls tools, which is something else. And the return is the discounted sum of the rewards; Python's `return` statement, in code font, is what ends a function and gives back its result."),
        mo.md(r"""
        The picture: one step of the loop between agent and environment. It repeats until the task ends.
        """),
        draw.flow(
            [("Agent", ["sees the situation,", "chooses an action"]),
             ("Environment", ["changes, and gives", "back a reward"])],
            forward=["action"],
            back=["reward and\nnew situation"],
            title="One step, again and again"),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To work out the return of a game:

    1. Write down the rewards of the game, one per step, in order.
    2. Start with a return `G` of 0, and a factor of 1 for the first reward.
    3. For every reward: add the reward times the factor to `G`, then multiply the factor by `gamma` for the next reward.
    4. Compare returns: the agent prefers the larger one.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Two games of four steps: one gives a point at once, the other 4 points at the last step. Which one has the larger return, with $\gamma$ = 0.5 and with $\gamma$ = 0.75? `_discounted` works out the return of a list of rewards. Each `print` shows $\gamma$, then the return of the point at once, then that of the 4 points at the end. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_return():
    # 1. the rewards of two games, one per step: a point at once, or 4 points at the end
    _now = [1, 0, 0, 0]
    _late = [0, 0, 0, 4]

    def _discounted(rewards, gamma):
        # 2. a return of 0, and a factor of 1 for the first reward
        G = 0
        factor = 1
        # 3. every reward times its factor; the next factor is gamma times as large
        for r in rewards:
            G = G + factor * r
            factor = factor * gamma
        return G

    # 4. compare the returns: gamma, the point at once, the 4 points at the end
    print(0.5, _discounted(_now, 0.5), _discounted(_late, 0.5))
    print(0.75, _discounted(_now, 0.75), _discounted(_late, 0.75))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    With $\gamma$ = 0.5, the 4 at step 4 counts 0.5 · 0.5 · 0.5 = 0.125 times as much: a return of 0.5, less than the point at once. With $\gamma$ = 0.75 it counts 0.75 · 0.75 · 0.75 ≈ 0.42 times as much: a return of 1.6875, more than the point at once. The same rewards, another $\gamma$, another choice: the discount factor sets how patient the agent is.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_parts():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. Which part is which?

    Two tasks: a robot vacuum cleaner that learns to clean, and a program that learns to play chess. Put each part in its group.
    """)
    return


@app.cell(hide_code=True)
def exercise_parts():
    sort_parts = L.sort("parts")
    sort_parts
    return (sort_parts,)


@app.cell(hide_code=True)
def check_parts(sort_parts):
    result_parts = L.check("parts", sort_parts)
    result_parts
    return (result_parts,)


@app.cell(hide_code=True)
def hints_parts():
    L.hints(
        "parts",
        [
            "For each part, ask: does it choose, is it chosen, does it get acted on, or is it a number that says how good it went?",
            "Every task has one agent, one environment, one action here and one reward. The agent is what learns; the rewards are numbers.",
            "The robot vacuum cleaner and the chess program are the agents: they choose. What do they act on, what do they do, and which numbers say how well it went?",
        ],
        why="**Agent:** the robot vacuum cleaner, the chess program. **Environment:** the rooms with their dust and furniture, the board and the opponent. **Action:** turn left, move a knight. **Reward:** +1 for every square metre that gets clean, +1 for a won game and −1 for a lost one.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_return():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The return of any game

        `discounted_return(rewards, gamma)` gives the return of a list of rewards. Replace the two `...` with step 3 of the recipe. The check calls `discounted_return([0, 0, 0, 1], 0.9)`: a reward of 1, three steps after the first, so that should give 0.9 · 0.9 · 0.9 = `0.729`. It also tries other rewards, negative ones too, and other discount factors.
        """),
        L.task("Make `discounted_return` add every reward times its factor, and make the factor `gamma` times as large at every step."),
    ])
    return


@app.cell(hide_code=True)
def exercise_return():
    box_return = L.code_box("discounted-return", lines=8)
    box_return
    return (box_return,)


@app.cell(hide_code=True)
def check_return(box_return):
    result_return = L.check("discounted-return", box_return.value)
    result_return
    return (result_return,)


@app.cell(hide_code=True)
def hints_return():
    L.hints(
        "discounted-return",
        [
            "Step 3 of the recipe: two lines, in that order.",
            "The first line adds this reward to `G`, but not in full: times `factor`. The second line makes `factor` ready for the next reward, which counts `gamma` times as much.",
            "`G = G + factor * ___` and `factor = factor * ___`.",
        ],
        why="With `[0, 0, 0, 1]` and 0.9, the factors are 1, 0.9, 0.81 and 0.729. The first three rewards are 0, so only the last one adds anything: 0.729 · 1 = 0.729.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_zero():
    mo.md(r"""
    ### Exercise 3. No patience at all
    """)
    return


@app.cell(hide_code=True)
def choice_zero_cell():
    choice_zero = L.choice(
        "gamma-zero",
        question="An agent with $\\gamma$ = 0 can choose between two actions. One gives a reward of 1 at the next step and nothing after that; the other gives 0 at the next step and 100 at the step after. Which does it choose?",
        options={
            "later": "The 100: it is a hundred times as much",
            "either": "Either one: they are both rewards",
            "right": "The 1: only the next reward counts",
        },
        correct="right",
        why={
            "later": "The 100 comes one step later, so it counts $\\gamma$ times as much, and $\\gamma$ is 0 here. See *The idea*.",
            "either": "The two returns differ. Work them out with the formula of *The idea*.",
            "right": "With $\\gamma$ = 0, every reward after the next one is multiplied by 0: the return is 1 against 0. Such an agent only sees the next reward.",
        },
    )
    choice_zero
    return (choice_zero,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Kinds of learning](/course/open/ml-kinds-of-learning?lang=en) named reinforcement learning as learning from rewards; now you know its parts. An agent acts in an environment, and it is not the next reward that counts, but the return: all the rewards from now on, with later ones discounted by $\gamma$. Which action leads to a large return is what the agent has to learn, by trying. In the next lesson, *A language model predicts the next token*, you see how a model learns from text alone, with no rewards and no labels written by people.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_zero, result_parts, result_return):
    L.wrap_up(result_parts, result_return, choice_zero)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
