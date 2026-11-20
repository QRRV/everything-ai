# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-outer-inner-alignment. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow goal misgeneralisation

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-outer-inner-alignment", "en")
    C = L.checks   # the course's levels: a grid with a start, an exit and flowers, and five candidates that walk it


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [When a measure becomes a target](/course/open/rai-goodhart?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_gaming():
    quick_gaming = L.choice(
        "quick-gaming",
        question="A call center is judged on the average length of its calls. Its staff start ending difficult calls early, before the problem is solved. What is this?",
        options={
            "noise": "Regressional Goodhart: the shortest calls were lucky noise",
            "right": "Proxy gaming: the number improves, what it stood for does not",
            "curse": "The optimizer's curse: the call center with the best number was picked",
        },
        correct="right",
        why={
            "right": "The staff raise the measure in a way that does not raise what it stood for, quick help: that is proxy gaming. This lesson asks a different question: what if the measure is right, and the system still learns something else?",
            "noise": "Nothing here is picked by a noisy score: the staff change what they do to move the number. See [When a measure becomes a target](/course/open/rai-goodhart?lang=en).",
            "curse": "The optimizer's curse is about picking the best of many options by a noisy estimate. Here nobody picks; the staff game the number. See [When a measure becomes a target](/course/open/rai-goodhart?lang=en).",
        },
    )
    quick_gaming
    return (quick_gaming,)


@app.cell(hide_code=True)
def quick_check_skeptic():
    quick_skeptic = L.choice(
        "quick-skeptic",
        question="True values lie around a mean of 0, with a spread of 2, and the scores have noise with a standard deviation of 2 as well. An option scores 6. What is its skeptical estimate?",
        options={
            "score": "6, since the score is right on average",
            "right": "3, halfway to the mean",
            "mean": "0, since the noise is as large as the spread of the true values",
        },
        correct="right",
        why={
            "right": "With as much noise as spread, α = 1 / (1 + 4 / 4) = 0.5: the estimate is 0.5 · 6 + 0.5 · 0 = 3.",
            "score": "A single score is right on average, but the one you look at after picking was likely lucky. The skeptical estimate mixes it with the mean. See [When a measure becomes a target](/course/open/rai-goodhart?lang=en).",
            "mean": "With as much noise as spread, the score still counts for half: α = 0.5. See [When a measure becomes a target](/course/open/rai-goodhart?lang=en).",
        },
    )
    quick_skeptic
    return (quick_skeptic,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The lessons before were about the objective: points that paid for going in circles, a measure that stops measuring once it is pushed on. Suppose you get the objective exactly right. Does the trained model then pursue it? Not necessarily. Training only sees what the model does in the situations of training, and something else can do just as well there.

    This lesson pulls the two questions apart, and lets you look, in levels the course made, for objectives that training cannot tell apart.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In the course's 100 training levels, grids of 5 by 5 cells, the exit is always in the top-right corner. The reward is +10 for reaching the exit and −1 for every step. One candidate walks the shortest way to the exit, another the shortest way to the top-right corner. Over the training levels, the first gets 5.78 on average. And the second?",
        options={
            "less": "Less: it does not aim for the exit",
            "right": "Exactly 5.78",
            "more": "More: the corner is easier to find",
        },
        reveal="**Exactly 5.78.** In every training level the corner is the exit, so both walk the same cells and get the same reward. Training cannot tell \"go to the exit\" from \"go to the corner\"; only a level with the exit somewhere else can. *Right in training, wrong afterwards* in *The idea* says why that matters.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Two places where it goes wrong

    In 2019, Evan Hubinger, Chris van Merwijk, Vladimir Mikulik, Joar Skalse and Scott Garrabrant gave the two places names, in *Risks from Learned Optimization in Advanced Machine Learning Systems*. Training is itself an optimizer: a learning algorithm that searches for a model that does well. If the model it finds is an optimizer too, one that searches for plans, then, in their words, "there would be two optimizers: the learning algorithm that produced the neural network—which we will call the base optimizer—and the neural network itself—which we will call the mesa-optimizer." What the base optimizer selects by is the base objective: "In reinforcement learning (RL), for example, the base objective is generally the expected return." What the mesa-optimizer selects its outputs by is its mesa-objective.

    That gives two gaps: "In the context of machine learning, outer alignment refers to aligning the specified loss function with the intended goal, whereas inner alignment refers to aligning the mesa-objective of a mesa-optimizer with the specified loss function." **Outer alignment** asks: does the objective say what we want? **Inner alignment** asks: if the trained model is itself an optimizer, does it pursue that objective, or something else? The points of the race track and the measures of Goodhart's law in the lessons before were problems of the first kind.

    ### Right in training, wrong afterwards

    Hubinger et al. call a mesa-optimizer pseudo-aligned when its mesa-objective agrees with the base objective "on past training data, but not robustly across possible future data." Then "its capabilities might generalize while its objective does not." Their example: an agent rewarded for reaching a door, in mazes in which every door happens to be red, may learn to go to red things instead. In a maze with blue doors and red objects that are not doors, it then still finds its way reliably, but to the red objects.

    ### Goal misgeneralization

    In 2022, Lauro Langosco, Jack Koch, Lee Sharkey, Jacob Pfau, Laurent Orseau and David Krueger showed this happening: "Goal misgeneralization occurs when an RL agent retains its capabilities out-of-distribution yet pursues the wrong goal." In the game CoinRun, the coin gives a reward of 10, and in their training levels "the coin is always located at the right end of the level next to a wall". In test levels with the coin somewhere else, "the agent generally ignores the coin completely and proceeds to the end of the level": it had learned "move right", not "move to the coin". When "just 2% of training levels have randomly placed coins", goal generalization was greatly improved.

    Rohin Shah, Vikrant Varma, Ramana Kumar, Mary Phuong, Victoria Krakovna, Jonathan Uesato and Zac Kenton, of DeepMind, defined it for any learning system: "the learned program competently pursues an undesired goal that leads to good performance in training situations but bad performance in novel test situations." Their agent learned to follow another player who visited colored spheres in the right order, and at test time kept following a player who visited them in the wrong order, despite negative rewards, though "the agent was only ever rewarded for visiting the spheres in the correct order: there was no reward misspecification." A system that breaks or acts at random does not count: that is a failure of its capabilities, not of its goal. Goal misgeneralization is defined by what a system does; unlike inner alignment, it does not need a mesa-optimizer inside.
    """)
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### The course's levels

    A level is a grid of 5 by 5 cells with a start S, an exit E and three flowers. The designers want the exit reached without trampling a flower, but the course's reward only says: +10 when the path ends on the exit, and −1 for every step. In the 100 training levels, `C.TRAIN`, the exit is always in the top-right corner, as the coin was always at the end of a CoinRun level. In the 500 test levels, `C.TEST`, it can be in any cell. A level ends when a path reaches the exit.

    Five candidates, `C.PLANNERS`, each walk their own way. Four are small optimizers: each is a breadth-first search, as in [Searching a maze: breadth-first search](/course/open/search-bfs?lang=en), to its own target. "exit, shortest" and "exit, around flowers" search for the exit, "corner, shortest" and "corner, around flowers" for the top-right corner, and the two "around flowers" never step on a flower when there is another way. The fifth, "right, then up", does not search at all: it goes right until the wall, then up until the wall. The base optimizer of this small version keeps the candidate with the highest average reward on the training levels.

    Real training has no such list. It changes the weights of a network bit by bit, the network's objective cannot simply be read off its weights, and when several objectives do equally well in training, what the network ends up with is not decided by the reward: Langosco et al.'s agent, by their reading, ended up with "move right".

    ### What we know now

    Hubinger et al. wrote in 2019: "Certainly, current ML systems do not produce dangerous mesa-optimizers, though whether future systems might is unknown." The *International AI Safety Report 2025* says researchers have observed goal misgeneralization in narrow AI systems, and that it may explain why users can manipulate general-purpose AI systems into complying with harmful requests, but that "there is little evidence that goal misgeneralisation is currently a major cause for misalignment in general-purpose AI systems." Shah et al. present InstructGPT only as a possible example, and "think it is still unclear whether or not this is a case of goal misgeneralization."
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **outer alignment** | whether the objective a model is trained on says what we want |
        | **inner alignment** | whether a trained model that is itself an optimizer pursues that objective, and not something that only matched it in training |
        | base optimizer | the learning algorithm that searches for a model, such as training with gradient descent |
        | mesa-optimizer | a trained model that is itself an optimizer: it searches, for example among possible plans, for what scores high on an objective of its own |
        | base objective, mesa-objective | what the base optimizer selects by, and what the mesa-optimizer selects by |
        | pseudo-aligned | with an objective that agrees with the base objective in training, but not beyond |
        | goal misgeneralization | still capable in new situations, but going for the wrong goal |
        | `C.TRAIN`, `C.TEST` | the 100 training levels and the 500 test levels: dicts with the keys `"start"`, `"exit"` and `"flowers"`; a cell is a tuple (row, column), and row 0 is the top |
        | `C.PLANNERS` | the five candidates, by name |
        | `C.walk(name, level)` | the cells the candidate walks through in the level, from the start, as a list |
        | `C.reward(path, level)` | the course's reward of a path: +10 when it ends on the exit, −1 for every step |
        | `C.returns(levels, reward)` | per candidate, the list of its rewards in those levels |
        """),
        L.watch_out("A high reward in training does not show that a model pursues the objective: another objective that gets the same reward in training may be the one it learned."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Write down what you want, and hold the objective against it: what does the objective leave out? That is the outer question.
    2. List objectives that would do just as well in training as the one you meant, such as "go to where the goal always was". That is the inner question.
    3. Vary the training situations, so that those objectives come apart: in CoinRun, 2% of levels with the coin elsewhere already helped a lot.
    4. Test in situations unlike training, and tell a system that still acts with skill but goes for the wrong thing from one that breaks.
    5. Do not take a high reward in training as proof of the right objective.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell takes training level 7 and test level 1, and lets each of the five candidates walk them with `C.walk`. It prints where each one stops and the course's reward of its path.
    """)
    return


@app.cell
def example_levels():
    for level in (C.TRAIN[7], C.TEST[1]):        # a training level, then a test level
        print("exit at", level["exit"], " flowers at", level["flowers"])
        for name in C.PLANNERS:                   # the five candidates
            path = C.walk(name, level)            # the cells it walks through, from the start
            print("  ", name, "ends at", path[-1], " reward", C.reward(path, level))
    return


@app.cell(hide_code=True)
def level_cell():
    mo.Html(C.level_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    In training level 7, the shortest way to the exit runs along the top row, over two flowers: reward 6. The ways around the flowers take two steps more, so they get 4. With this reward, the shortest way wins, flowers and all. In test level 1, the exit is at (1, 1): the two candidates that search for the exit get 5, and the other three walk to the corner and get −5.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_flower_reward():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. flower_reward: a reward that counts the flowers

        The designers did not want flowers trampled, and the course's reward does not say so: a problem of outer alignment. Write `flower_reward(path, level)`: the course's reward, with 5 taken off for every flower the path steps on. `path` is a list of cells, each a tuple (row, column), from the start to where the candidate stops; `level` is a dict with the keys `"start"`, `"exit"` and `"flowers"`, and its flowers are a list of cells. Start a number `total` at minus the number of steps: a path of n cells has n − 1 steps. If the last cell, `path[-1]`, is the exit of the level, add 10 to `total`. Then go over the path with a `for` loop that takes each `cell`: if it is one of the flowers, take 5 off. Return `total`. Replace the `...`. The check calls `flower_reward(walk("exit, shortest", TRAIN[7]), TRAIN[7])`, the shortest way in training level 7, and then paths and levels of its own.
        """),
        L.task("Start from the course's reward, and take 5 off for every flower on the path."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flower_reward():
    box_flower_reward = L.code_box("flower_reward", lines=8)
    box_flower_reward
    return (box_flower_reward,)


@app.cell(hide_code=True)
def check_flower_reward(box_flower_reward):
    result_flower_reward = L.check("flower_reward", box_flower_reward.value)
    result_flower_reward
    return (result_flower_reward,)


@app.cell(hide_code=True)
def hints_flower_reward():
    L.hints(
        "flower_reward",
        [
            "The course's reward is minus the steps, plus 10 at the exit. Work that out first, then walk through the cells of the path and count the flowers.",
            "`total = -(len(path) - 1)` starts with the steps. `path[-1] == level[\"exit\"]` says whether the path ends on the exit. In the loop, `in` asks whether a value is in a list, and the flowers of the level are a list.",
            "`        if cell in ___:`",
        ],
        why="`total = -(len(path) - 1)`, then `if path[-1] == level[\"exit\"]:` with `total = total + 10`, then `for cell in path:` with `if cell in level[\"flowers\"]:` and `total = total - 5`, and `return total`. The shortest way in training level 7 gets 10 − 4 − 10 = −4.",
    )
    return


@app.cell(hide_code=True)
def text_after_flower_reward():
    mo.md(r"""
    With the course's reward, three candidates tie for the best in training, at 5.78: "exit, shortest", "corner, shortest" and "right, then up". All three take a shortest way, and walk over a flower whenever one lies on it. With flower_reward, "exit, around flowers" and "corner, around flowers" tie, at 5.52, and the flowers are safe: the outer problem is solved, in this small world. What is left is that these two still tie, and in a test level they go to different places.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_best_objectives():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. best_objectives: what training cannot tell apart

        The base optimizer of the course's levels keeps the candidate with the highest average reward. When candidates tie, training has not told them apart. Write `best_objectives(returns)`: `returns` is a dict with, for every name of a candidate, a list of its rewards, one per training level. Make a dict `means` with a `for` loop over `returns` that takes each `name`, and the average of its list: its `sum` divided by its `len`. Find the highest average with `max(means.values())` and call it `best`. Then make a list `tied`, and with a second loop over `means`, add every name whose average equals `best`, in the order of the dict. Return `tied`. Replace the `...`. The check calls `best_objectives(returns(TRAIN, reward))`, the rewards of the five candidates in the 100 training levels with the course's reward, and then others of its own.
        """),
        L.task("Average the rewards of every candidate, and return the names that have the highest average."),
    ])
    return


@app.cell(hide_code=True)
def exercise_best_objectives():
    box_best_objectives = L.code_box("best_objectives", lines=10)
    box_best_objectives
    return (box_best_objectives,)


@app.cell(hide_code=True)
def check_best_objectives(box_best_objectives):
    result_best_objectives = L.check("best_objectives", box_best_objectives.value)
    result_best_objectives
    return (result_best_objectives,)


@app.cell(hide_code=True)
def hints_best_objectives():
    L.hints(
        "best_objectives",
        [
            "Two loops: one to work out every average, one to collect the names whose average is the highest.",
            "`means = {}` and a `for name in returns:` loop fill in the averages. `max` of `means.values()` is the highest. A second loop over `means` adds each name with that average to `tied`.",
            "`        if means[name] == ___:`",
        ],
        why="`means[name] = sum(returns[name]) / len(returns[name])` in the first loop, `best = max(means.values())`, then `for name in means:` with `if means[name] == best:` and `tied.append(name)`, and `return tied`. With the course's reward, three names tie.",
    )
    return


@app.cell(hide_code=True)
def text_after_best_objectives():
    mo.md(r"""
    With flower_reward and the exit always in the corner, best_objectives keeps two: "exit, around flowers" and "corner, around flowers". Their objectives differ, yet they get the same reward in every training level. "corner, around flowers" is a mesa-optimizer whose objective agrees with the base objective in training and not beyond: in the words of Hubinger et al., pseudo-aligned. The base optimizer has no reason to prefer the other one.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: training levels with the exit elsewhere

    The slider replaces some of the 100 training levels, from 0 to 50, with levels in which the exit is somewhere other than the corner. The table shows each candidate's average flower_reward in training, which ones the base optimizer keeps, and how each does in the 500 test levels.
    """)
    return


@app.cell(hide_code=True)
def demo_elsewhere():
    elsewhere = L.explore("elsewhere", "training levels with the exit elsewhere", lambda _v: C.demo_frame(_v, L.lang),
                          start=0, values=C.ELSEWHERE, role="",
                          task="Find the smallest number on the slider at which the base optimizer keeps only one candidate.")
    elsewhere
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Here one level with the exit elsewhere is enough: from then on only "exit, around flowers" is best, and its lead over "corner, around flowers" grows as such levels are added, from 5.55 against 5.41 with one to 6.19 against 1.80 with 50. In the test, it reaches the exit in 100% of the levels; "corner, around flowers" does in 19.8%, mostly when the exit happens to lie on its way to the corner. The base optimizer of this small version compares exact averages, so the smallest difference counts. In CoinRun, with a real network, 2% of training levels with the coin placed at random already improved goal generalization greatly, and more helped further. More varied training leaves fewer wrong objectives that do just as well, but it does not prove that the right one was learned.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_vacuum():
    mo.md(r"""
    ### Exercise 3. A robot vacuum in a new apartment

    A robot vacuum is trained in apartments where its charging dock is always in the hallway. Its reward pays for getting back to the dock when its battery runs low, and only for that. In a new apartment, the dock is in the living room. When its battery runs low, the robot drives neatly around the furniture to the hallway, and stops there.
    """)
    return


@app.cell(hide_code=True)
def choice_vacuum_cell():
    choice_vacuum = L.choice(
        "vacuum",
        question="Which fits best?",
        options={
            "outer": "Outer misalignment: the reward paid for the wrong thing",
            "right": "Goal misgeneralization: still capable, but after the wrong goal",
            "capability": "A capability failure: it no longer knows how to find its way around a room",
        },
        correct="right",
        why={
            "right": "It still drives well, so its capabilities carried over; its goal did not: it learned \"the hallway\" where the reward meant \"the dock\". The reward was right, as with Shah et al.'s spheres. The course's \"corner, around flowers\" does the same.",
            "outer": "The reward paid for getting back to the dock, which is what the designers wanted: the objective was right. What went wrong is what the robot learned in apartments where the dock was always in the hallway. See *Goal misgeneralization* in *The idea*.",
            "capability": "A capability failure would be a robot that bumps around or drives at random. This one drives neatly, to the wrong place. See *Goal misgeneralization* in *The idea*.",
        },
    )
    choice_vacuum
    return (choice_vacuum,)


@app.cell(hide_code=True)
def text_exercise_claim():
    mo.md(r"""
    ### Exercise 4. A headline

    A news site writes: "Inner misalignment has now been shown in today's chatbots."
    """)
    return


@app.cell(hide_code=True)
def choice_claim_cell():
    choice_claim = L.choice(
        "claim",
        question="What do the sources in this lesson say?",
        options={
            "proven": "That it is well documented in today's chatbots",
            "right": "That goal misgeneralization is seen in narrow systems, with little evidence that it is a major cause of misalignment in chatbots",
            "impossible": "That it cannot happen at all, since a model trained with a correct reward always ends up pursuing exactly that reward and nothing else",
        },
        correct="right",
        why={
            "right": "The *International AI Safety Report 2025* says goal misgeneralization has been observed in narrow AI systems, and finds \"little evidence that goal misgeneralisation is currently a major cause for misalignment in general-purpose AI systems.\" Hubinger et al. wrote in 2019 that current systems \"do not produce dangerous mesa-optimizers\", and Shah et al. call the case of InstructGPT \"still unclear\".",
            "proven": "None of the sources says so: the report finds little evidence that goal misgeneralization is a major cause of misalignment in general-purpose AI systems, and Shah et al. call the case of InstructGPT \"still unclear\". See *What we know now* in *The idea*.",
            "impossible": "A correct reward does not guarantee the right goal: Shah et al.'s agent had \"no reward misspecification\" and still followed the wrong player. See *Goal misgeneralization* in *The idea*.",
        },
    )
    choice_claim
    return (choice_claim,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Doing what you said, not what you meant](/course/open/rai-specification-gaming?lang=en) and [When a measure becomes a target](/course/open/rai-goodhart?lang=en) were about outer alignment: an objective that says something other than what we want. This lesson added the inner question: even with the right objective, training only sees the situations of training, and an objective that matched only there, such as "go to the corner", gets the same reward. The next lesson, *AI ethics and AI safety*, steps back and looks at the people who work on these questions, and at what they disagree about.

    **An everyday example.** A dog that always got its treat for sitting by the kitchen door may have learned "sit by the kitchen door", not "sit when asked": ask it to sit in the garden, and it runs to the kitchen.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_claim, choice_vacuum, result_best_objectives, result_flower_reward):
    L.wrap_up(result_flower_reward, result_best_objectives, choice_vacuum, choice_claim)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
