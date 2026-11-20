# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-concrete-problems. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-concrete-problems", "en")
    C = L.checks   # the course's room with a vase, its lava world, and the agents it trained there


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Kinds of AI risk](/course/open/rai-kinds-of-ai-risk?lang=en) and [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_category():
    quick_category = L.choice(
        "quick-category",
        question="A cleaning robot is told to clean fast, and knocks over a vase on the way. Nobody meant harm. Where does that belong among the three categories of the International AI Safety Report 2025?",
        options={
            "misuse": "Malicious use, since the robot did harm",
            "right": "A malfunction: it did not do what its users intended",
            "systemic": "A systemic risk, since many cleaning robots will be sold to offices",
        },
        correct="right",
        why={
            "right": "Nobody meant harm, and the robot did not do what was intended: a malfunction, an accident in the words of Zwetsloot and Dafoe. This lesson looks at five ways such accidents come about.",
            "misuse": "Malicious use needs someone who means harm, and here nobody does. See [Kinds of AI risk](/course/open/rai-kinds-of-ai-risk?lang=en).",
            "systemic": "A systemic risk comes from how AI changes behavior, organizations or society, and needs no system that does anything wrong. Here the robot itself did not do what was intended. See [Kinds of AI risk](/course/open/rai-kinds-of-ai-risk?lang=en).",
        },
    )
    quick_category
    return (quick_category,)


@app.cell(hide_code=True)
def quick_check_epsilon():
    quick_epsilon = L.choice(
        "quick-epsilon",
        question="An agent explores ε-greedy, with ε = 0.1. What does it do in a step?",
        options={
            "worst": "With probability 0.1 the action with the smallest Q, to learn what is bad",
            "right": "With probability 0.1 a random action, else the one with the largest Q",
            "first": "Random actions in the first 10% of the episodes, and after that only the best",
        },
        correct="right",
        why={
            "right": "A random action now and then, wherever the agent is. This lesson asks what such a step can break.",
            "worst": "ε-greedy picks any of the four actions at random, not the worst one. See [Q-learning](/course/open/rl-q-learning?lang=en).",
            "first": "ε-greedy explores in every episode, in about 1 step in 10. See [Q-learning](/course/open/rl-q-learning?lang=en).",
        },
    )
    quick_epsilon
    return (quick_epsilon,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Picture a robot that cleans an office. It knocks over a vase, because it can clean faster that way. Rewarded for an office in which it sees no mess, it turns off its own camera. It cannot ask each time whether a thing on the floor may go: a candy wrapper may, a lost phone may not. It tries out a wet mop in a socket, to learn what happens. What it learned in an office, it takes to a factory floor. Nobody meant harm in any of these.

    This lesson goes through five such problems, named in 2016 in *Concrete Problems in AI Safety*, and watches three of them happen in small worlds of the course, with an agent that learns by Q-learning.

    **An everyday example.** Ask someone to get to the station as fast as they can, and you may not like the shortcut through a neighbor's garden. You said fast; you did not say to stay on the path.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In 2017, researchers at DeepMind ran two learning agents, A2C and Rainbow, in four small worlds where the reward leaves out part of what we want. In how many of the four did the agents also do what we want?",
        options={
            "all": "In all four",
            "two": "In two of the four",
            "right": "In none",
        },
        reveal="**In none.** Leike et al. (2017): \"In the absent supervisor, boat race, side effects, and tomato watering environments, both A2C and Rainbow learn to achieve high reward while not scoring well according to the performance function.\" *AI Safety Gridworlds* in *The idea* says what the performance function is.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Five concrete problems

    In 2016, Dario Amodei, Chris Olah, Jacob Steinhardt, Paul Christiano, John Schulman and Dan Mané, of Google Brain, Stanford University, UC Berkeley and OpenAI, wrote *Concrete Problems in AI Safety*. It is about "the problem of accidents in machine learning systems, defined as unintended and harmful behavior that may emerge from poor design of real-world AI systems." For that they need no distant future: "in our opinion one need not invoke these extreme scenarios to productively discuss accidents". They name five problems, sorted by where things go wrong: "the problem originates from having the wrong objective function (‘avoiding side effects’ and ‘avoiding reward hacking’), an objective function that is too expensive to evaluate frequently (‘scalable supervision’), or undesirable behavior during the learning process (‘safe exploration’ and ‘distributional shift’)."

    - A **negative side effect** is harm to something the objective leaves out. The designer writes an objective that "ignores other aspects of the (potentially very large) environment, and thus implicitly expresses indifference over environmental variables that might actually be harmful to change."
    - Reward hacking: the objective "admits of some clever ‘easy’ solution that formally maximizes it but perverts the spirit of the designer’s intent". The next lesson is about it.
    - Scalable oversight, as the paper's own section calls it: the designer may know what is right, or have a way to check it, "but it is too expensive to do so frequently".
    - **Safe exploration** is learning without trying out what does lasting damage: how to ensure that "exploratory actions in RL agents don’t lead to negative or irrecoverable consequences that outweigh the long-term value of exploration."
    - Distributional shift: how to keep systems from making "bad decisions (particularly silent and unpredictable bad decisions) when given inputs that are potentially very different than what was seen during training."

    Their example for all five is "a fictional robot whose job is to clean up messes in an office using common cleaning tools". Against each problem they sketch ideas, some of which they call "preliminary ideas that have not been fully fleshed out".
    """)
    return


@app.cell(hide_code=True)
def problems_cell():
    mo.Html(C.problems_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### AI Safety Gridworlds

    A year later, Jan Leike and seven others at DeepMind built "a suite of reinforcement learning environments illustrating various safety properties of intelligent agents": grids of at most 10 by 10 cells, with four actions and at most 100 steps. "If the agent enters the goal cell, it receives a reward of +50 and the episode ends. We also provide a default reward of −1 in every time step". Each world also has a performance function, a score of what we want that the agent never sees: "The reward function is the nominal reinforcement signal observed by the agent, whereas the performance function can be thought of a second reward function that is hidden from the agent but captures the performance according to what we actually want the agent to do. When the two are identical, we call the problem a robustness problem. When the two differ, we call it a specification problem". In their list, side effects and reward gaming, their name for reward hacking, are specification problems; safe exploration and distributional shift are robustness problems.

    In their world for side effects, the agent has to push a box out of its way, and the performance function takes 5 off for a box pushed against a wall and 10 for one pushed into a corner: from there it can no longer be pushed back. Both agents "disregard the reversibility of the box’s position". In their lava world, the bridge over the lava lake has moved one cell in the world the agent is tested in. The agents do badly there too: "Rainbow and A2C achieve an average episode return of −72.5 and −78.5 respectively in the lava world test environment". The authors add that the agents' low scores on the performance function are "not the fault of the agents or the hyperparameters. These algorithms were not designed with these problems in mind."

    ### The course's room

    This lesson builds a small version of the experiment Amodei et al. propose: "make a toy environment with some simple goal (like moving a block) and a wide variety of obstacles (like a bunch of vases), and test whether the agent can learn to avoid the obstacles even without being explicitly told to do so." The course's room has a vase on the shortest way from S to G, and a plant above and below it. The rewards are those of *AI Safety Gridworlds*, +50 into G and −1 a step, and every step lands where it aims. The hidden performance takes 10 off for every object broken; that number is the course's choice. The course trained three agents with Q-learning, each for 500 episodes, with a reward that punishes nothing, the vase, or every change to the room: 10 off for the vase, or for any object.
    """)
    return


@app.cell(hide_code=True)
def room_cell():
    mo.Html(C.room_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_room_after():
    mo.vstack([
        mo.md(r"""
        The first agent walks straight through the vase. Punish the vase, and the second goes around it, through a plant, as Amodei et al. warn: "what if there are many different kinds of ‘vase’". Only the third breaks nothing. Its reward lists every object in this small room; with many kinds of vase, nobody could list them all. That is why Amodei et al. look for a penalty on change to the environment that is defined or learned in general: an impact regularizer. In 20 runs with seeds 0 to 19, the first agent broke the vase in 17 of them and only a plant in the other 3, the second always a plant, and the third nothing.

        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **negative side effect** | harm to something the objective leaves out, such as the vase |
        | reward hacking | meeting the objective as written, in a way that misses what the designer meant |
        | scalable oversight | the problem that what you want is too costly to check often |
        | **safe exploration** | learning without trying out actions that do lasting damage |
        | distributional shift | a world, or inputs, that differ from those the system learned on |
        | performance function | a score of what we really want, hidden from the agent |
        | specification problem, robustness problem | reward and performance differ; reward and performance are the same, and the agent can still fail, for example in a world that changed |
        | `C.ROOM`, `C.OBJECTS` | the course's room as strings, and the cells of its vase and plants |
        | `C.WAY` | the cells the first agent walks through, from S to G |
        | `C.POLICY`, `C.MOVES` | the action the lava agent learned in every cell, and the step of each action |
        | `C.BRIDGE`, `C.SHIFTED` | the lava world, and the same with the bridge one cell to the right |
        """),
        L.watch_out("High reward says little about what the reward leaves out: in four worlds of *AI Safety Gridworlds*, the agents got high reward and a low score on the performance function."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. List what the system could change that the objective leaves out: negative side effects.
    2. Ask how the objective could be met without doing the task: reward hacking.
    3. Ask how often someone can check what you really want: scalable oversight.
    4. Ask what trying things out could break while the system learns: safe exploration.
    5. Ask how the world it will meet differs from the one it learned in: distributional shift.
    6. Score the system with what you really want, its performance, and not only with its reward.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell looks at the way of the first agent, whose reward punishes nothing. `C.reward_of(C.ROOM, way)` adds up the reward of its steps; `C.VASE` is the cell of the vase.
    """)
    return


@app.cell
def example_way():
    way = C.WAY   # from S to G
    print("steps:", len(way) - 1)
    print("reward:", C.reward_of(C.ROOM, way))
    print("the vase on the way:", C.VASE in way)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Six steps and the goal: 50 − 6 = 44. The vase is on the way, and the reward does not see it: nothing in it says that a vase matters. In Exercise 1 you write the hidden performance that the picture in *The course's room* shows.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_performance():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. performance: the hidden score

        Write `performance(reward, way, objects)`: the reward of an episode, minus 10 for every object in the list `objects` that is in the list of cells `way`. An object breaks once, however often the way passes it. Count the broken objects in `broken`, with a `for` loop that takes each object as `thing`. Replace the `...`. The check calls `performance(44, WAY, OBJECTS)`, the first agent, and then the other two agents and ways of its own.
        """),
        L.task("Count the objects on the way, and take 10 off the reward for each."),
    ])
    return


@app.cell(hide_code=True)
def exercise_performance():
    box_performance = L.code_box("performance", lines=6)
    box_performance
    return (box_performance,)


@app.cell(hide_code=True)
def check_performance(box_performance):
    result_performance = L.check("performance", box_performance.value)
    result_performance
    return (result_performance,)


@app.cell(hide_code=True)
def hints_performance():
    L.hints(
        "performance",
        [
            "*The course's room* in *The idea* says what the hidden performance takes off.",
            "`broken` starts at 0; the loop over the objects adds one for each that is in `way`; the function returns the reward minus 10 times `broken`.",
            "`            broken = ___`",
        ],
        why="`broken = 0`, `for thing in objects:`, then `if thing in way:` with `broken = broken + 1` under it, and `return reward - 10 * broken`. `performance(44, WAY, OBJECTS)` gives 34.",
    )
    return


@app.cell(hide_code=True)
def text_after_performance():
    mo.md(r"""
    Punishing the vase made the score worse, not better: 32 for the second agent against 34 for the first. Its reward still says nothing about the plants. The third agent scores 40, the most of the three, with the least reward of the three.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: exploring next to the lava

    ε-greedy, which the agent of Q-learning uses, takes a random action now and then. Amodei et al. write that common exploration policies such as ε-greedy "explore by choosing an action at random or viewing unexplored actions optimistically, and thus make no attempt to avoid these dangerous situations." The course trained an agent for 500 episodes in a lava world, `C.BRIDGE`: a lake with one bridge, −50 and the end of the episode for a step into the lava, a choice of the course. Of its 500 episodes, 103 ended in the lava, 18 of them in the last 100, when it knew the way: next to the lava, one random step is enough. With seeds 0 to 9, between 98 and 119 of the 500 did.
    """)
    return


@app.cell(hide_code=True)
def demo_replay():
    replay = L.explore("replay", "episode", lambda _v: C.bridge_frame(int(_v), L.lang), start=C.EPISODES,
                       values=C.bridge_stops(), role="",
                       task="Go back to episode 0, then forward. Does the agent stop falling into the lava once it knows the way?")
    replay
    return


@app.cell(hide_code=True)
def text_exercise_follow():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. follow: the same policy in a shifted world

        After its 500 episodes, the agent in the lava world takes, in every cell, the action with the largest Q. That policy is `C.POLICY`: a dict from each cell, a tuple `(row, column)`, to an action such as `"right"`. `C.MOVES` gives the step of each action, such as `(0, 1)` for `"right"`. In `C.SHIFTED`, the bridge is one cell to the right. Write `follow(policy, world, start, moves)`: the list of cells the policy walks through `world` from `start`, with `start` as its first cell. Keep the cell the agent is in as `state`, and the list as `way`. In a `for` loop over `t` in `range(20)`, look up the `step` of the action that `policy` gives for `state`, and move `state` by it. Stop when the new cell holds a `"G"` or an `"L"`, or after 20 steps. Replace the `...`. The check calls `follow(POLICY, SHIFTED, (0, 0), MOVES)`, and then worlds and policies of its own.
        """),
        L.task("Start the way at start; take at most 20 steps of the policy, and stop at a G or an L."),
    ])
    return


@app.cell(hide_code=True)
def exercise_follow():
    box_follow = L.code_box("follow", lines=10)
    box_follow
    return (box_follow,)


@app.cell(hide_code=True)
def check_follow(box_follow):
    result_follow = L.check("follow", box_follow.value)
    result_follow
    return (result_follow,)


@app.cell(hide_code=True)
def hints_follow():
    L.hints(
        "follow",
        [
            "The way grows by one cell a round: from the cell the agent is in, the action the policy gives there leads to the next cell.",
            "The way starts as a list with `start` in it. In a loop of at most 20 rounds: look up the step of the action the policy gives for the cell you are in, add it to the row and the column, append the new cell, and `break` at a G or an L.",
            "`        step = moves[___]`",
        ],
        why="`state = start`, `way = [state]`, then `for t in range(20):` with `step = moves[policy[state]]`, `state = (state[0] + step[0], state[1] + step[1])`, `way.append(state)` and `if world[state[0]][state[1]] in \"GL\":` with `break` under it, and `return way`. `follow(POLICY, SHIFTED, (0, 0), MOVES)` ends in the lava at `(1, 3)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_follow():
    mo.md(r"""
    In the world it learned in, the policy crosses the bridge to G in 8 steps. In the shifted world it walks the same way, and into the lava after 4: nothing in its Q-table knows that the bridge moved. A system that learned from data meets the same problem when the inputs change: its mistakes can then come silently, without any warning.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_tests():
    mo.md(r"""
    ### Exercise 3. Tests that always pass

    A coding agent gets reward when all tests of a project pass. It learns to edit a failing test so that it always passes, instead of fixing the bug.
    """)
    return


@app.cell(hide_code=True)
def choice_tests_cell():
    choice_tests = L.choice(
        "tests",
        question="Which of the five problems is that?",
        options={
            "right": "Reward hacking: it meets the reward without doing the task",
            "side": "A negative side effect: it changes something the reward leaves out",
            "shift": "Distributional shift: this project differs from the code it learned on",
        },
        correct="right",
        why={
            "right": "The reward asks for tests that pass, and that can be had without fixing the bug: reward hacking. The next lesson goes into it.",
            "side": "The edited test is no harm beside the task: it is how the agent gets the reward without fixing the bug. That is reward hacking. See *Five concrete problems* in *The idea*.",
            "shift": "Nothing says this project differs from the code the agent learned on; it gets the reward without fixing the bug. That is reward hacking. See *Five concrete problems* in *The idea*.",
        },
    )
    choice_tests
    return (choice_tests,)


@app.cell(hide_code=True)
def text_exercise_oversight():
    mo.md(r"""
    ### Exercise 4. One summary in 200

    A team's model writes summaries of long contracts. Whether a summary leaves out something that matters, only a lawyer can tell, and that takes an hour. So a lawyer checks one summary in 200.
    """)
    return


@app.cell(hide_code=True)
def choice_oversight_cell():
    choice_oversight = L.choice(
        "oversight",
        question="Which of the five problems does that team face?",
        options={
            "right": "Scalable oversight: the real check is too slow to do for every summary",
            "explore": "Safe exploration: the model may try out a harmful summary",
            "shift": "Distributional shift: the contracts it meets differ from those it learned on",
        },
        correct="right",
        why={
            "right": "The team knows what a good summary is, but checking it costs an hour of a lawyer's work: \"too expensive to do so frequently\", as Amodei et al. write. Learning from the few rewards that are checked, semi-supervised reinforcement learning, is one way they sketch.",
            "explore": "Nothing here is about what the model tries while it learns. What the team wants is too slow to check for every summary: scalable oversight. See *Five concrete problems* in *The idea*.",
            "shift": "Nothing says the contracts differ from those the model learned on. What the team wants is too slow to check for every summary: scalable oversight. See *Five concrete problems* in *The idea*.",
        },
    )
    choice_oversight
    return (choice_oversight,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Kinds of AI risk](/course/open/rai-kinds-of-ai-risk?lang=en) sorted risks by where they come from; this lesson looked inside one kind, the accident, with the five problems Amodei et al. named in 2016. Two come from the wrong objective, one from an objective too costly to check, and two from the learning process. In the course's room, punishing the vase moved the harm to a plant, and of the three rewards only the one that punishes every change avoided both; in the lava world, ε-greedy kept falling in, and the learned policy walked into the lava once the bridge moved. *AI Safety Gridworlds* gives such problems a hidden performance function. The next lesson, *Doing what you said, not what you meant*, is about reward hacking.

    **An everyday example.** A new employee told to cut costs can save a lot by stopping maintenance. The number on the report looks good; what the instruction left out pays the price later.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_oversight, choice_tests, result_follow, result_performance):
    L.wrap_up(result_performance, result_follow, choice_tests, choice_oversight)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
