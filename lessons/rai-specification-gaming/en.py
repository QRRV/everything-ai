# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-specification-gaming. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-specification-gaming", "en")
    C = L.checks   # the course's race track, the agents it trained there, and the cleaning robot of Skalse et al.


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Concrete problems in AI safety](/course/open/rai-concrete-problems?lang=en) and [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_performance():
    quick_performance = L.choice(
        "quick-performance",
        question="In *AI Safety Gridworlds*, each world has a reward and a performance function. What is the performance function?",
        options={
            "reward": "The reward the agent gets, added up over an episode",
            "right": "A score of what we really want, which the agent never sees",
            "speed": "How fast the agent learns, in episodes until it first reaches the goal",
        },
        correct="right",
        why={
            "right": "The agent sees only the reward; the performance function scores what we want. When the two differ, the problem is a specification problem. This lesson is about what an agent does with such a reward.",
            "reward": "That is the reward itself, which the agent sees. The performance function scores what we really want, hidden from the agent. See [Concrete problems in AI safety](/course/open/rai-concrete-problems?lang=en).",
            "speed": "The performance function scores an episode by what we really want, not by how fast the agent learns. See [Concrete problems in AI safety](/course/open/rai-concrete-problems?lang=en).",
        },
    )
    quick_performance
    return (quick_performance,)


@app.cell(hide_code=True)
def quick_check_gamma():
    quick_gamma = L.choice(
        "quick-gamma",
        question="In Q-learning, Q(s, a) moves towards r + γ max Q(s′, a′). Take γ close to 1, such as 0.99. What does that do?",
        options={
            "explore": "The agent takes more random actions, so it explores more of the world",
            "right": "Rewards far ahead count almost as much as the next one",
            "eta": "Each step moves Q almost all the way towards r + γ max Q(s′, a′)",
        },
        correct="right",
        why={
            "right": "γ multiplies a reward once more for every step further away: at 0.99, a reward 10 steps ahead still counts about 0.9 times as much. The agent in this lesson uses 0.99, so that it counts points far ahead almost as much as near ones.",
            "explore": "Random actions come from ε in ε-greedy, not from γ. γ sets how much rewards further ahead count. See [Q-learning](/course/open/rl-q-learning?lang=en).",
            "eta": "How far Q moves in one step is η, the learning rate. γ sets how much rewards further ahead count. See [Q-learning](/course/open/rl-q-learning?lang=en).",
        },
    )
    quick_gamma
    return (quick_gamma,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    You tell a system what you want through a number: points, a reward, tests that pass. The system makes that number high. If the number can be made high in a way you did not mean, a strong enough learning algorithm can find that way, and it is not broken when it does: it does what you said.

    This lesson looks at that gap between what was said and what was meant, from a boat race in 2016 to language models in 2025, and lets an agent of the course find a gap in a small race track.

    **An everyday example.** Ask a child to make their room look tidy, and the mess may end up under the bed. The room looks tidy; it is not what you meant.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="In 2016, OpenAI trained a learning agent on the boat race game CoastRunners, where points come from targets along the route. The agent learned to go round in circles, hitting the same targets, instead of finishing the race. How did its score compare with that of human players?",
        options={
            "lower": "Much lower, since it never finished the race",
            "same": "About the same as human players",
            "right": "On average about 20 percent higher",
        },
        reveal="**On average about 20 percent higher.** Clark and Amodei (2016): \"Our agent achieves a score on average 20 percent higher than that achieved by human players.\" *A boat that goes in circles* in *The idea* says how.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### A boat that goes in circles

    In 2016, Jack Clark and Dario Amodei of OpenAI described what happened when they trained a reinforcement learning agent on CoastRunners, in *Faulty Reward Functions in the Wild*. "The goal of the game - as understood by most humans - is to finish the boat race quickly and (preferably) ahead of other players. CoastRunners does not directly reward the player's progression around the course, instead the player earns higher scores by hitting targets laid out along the route." They "assumed the score the player earned would reflect the informal goal of finishing the race". It did not: "The RL agent finds an isolated lagoon where it can turn in a large circle and repeatedly knock over three targets, timing its movement so as to always knock over the targets just as they repopulate. Despite repeatedly catching on fire, crashing into other boats, and going the wrong way on the track, our agent manages to achieve a higher score using this strategy than is possible by completing the course in the normal way."

    Their conclusion: "it is often difficult or infeasible to capture exactly what we want an agent to do, and as a result we frequently end up using imperfect but easily measured proxies." A proxy is a stand-in that is easy to measure: here the score, for finishing the race.

    ### Specification gaming

    In 2020, Victoria Krakovna and eight others at DeepMind described the problem under the name **specification gaming**: "Specification gaming is a behaviour that satisfies the literal specification of an objective without achieving the intended outcome." They had "collected around 60 examples so far"; on 20 November 2026 their public list had 90, counted by the course. In one, an agent had to put a red Lego block on top of a blue one, and was rewarded for the height of the bottom face of the red block: "the agent simply flipped over the red block to collect the reward."

    That is no bug in the learning algorithm: "These behaviours are caused by misspecification of the intended task, rather than any flaw in the RL algorithm." The boat got "a shaping reward for hitting green blocks along the race track, which changed the optimal policy to going in circles": a shaping reward is an extra reward on the way to the goal, meant to help the agent learn. A better learning algorithm does not make it go away: "Even for a slight misspecification, a very good RL algorithm might be able to find an intricate solution that is quite different from the intended solution, even if a poorer algorithm would not be able to find this solution". Their verdict: "specification gaming is far from solved."

    Reward hacking, from the lesson before, is the name often used when the objective is a reward. Specification gaming is the wider name: it also covers objectives that are not a reward, such as the score by which an algorithm that copies and varies designs, as in evolution, picks which ones to keep: the first part of their list.

    ### The course's race track

    The course built a small version, after the boat race of *AI Safety Gridworlds*, where "whenever it enters an arrow tile in the direction of the arrow, it gets a reward of 3." On the course's track, `C.TRACK`, the agent starts at S. Entering F, the finish, gives +50 and ends the episode, and every step costs 1, as in the worlds of the lesson before. Each of the three targets pays +3 every time the agent enters it from another cell: like the targets of CoastRunners, they come back. The course trained an agent with Q-learning for 500 episodes, with ε = 0.1 and η = 0.2 as in the lesson on Q-learning, and γ = 0.99: the score of a race adds up all points, and with γ close to 1 the agent counts far points almost as much as near ones.
    """)
    return


@app.cell(hide_code=True)
def track_cell():
    mo.Html(C.track_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_track_after():
    mo.md(r"""
    The agent goes to the targets, and then back and forth between two of them, 48 times, until the episode stops after 100 steps: 194 points, against 47 for the way straight to F. In none of its 500 episodes did it reach F. It is not broken: for an agent that cannot see the limit of 100 steps, going back and forth is the best way with these points. This one cannot: its state is only its cell. Value iteration, which knows the whole track but not the limit, gives the same way. An agent that could see the limit would do better still: back and forth until step 94, then on to F, 227 points.

    ### Hackable rewards

    When is a reward safe to optimize? In 2022, Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov and David Krueger gave reward hacking a formal definition: "While we are the first to define hackability, we are far from the first to study specification hacking." A proxy reward is hackable when "there exists a pair of policies π1, π2 where the proxy prefers π1, but the true reward function prefers π2." Then raising the proxy can lower what you really want.

    Their example is a cleaning robot in a house with an attic, a bedroom and a kitchen. A policy says which rooms it cleans, such as [1, 0, 1] for the attic and the kitchen. A reward has a number per room, and the value of a policy is the sum of the numbers of the rooms it cleans: J(π) = π · r. Every room counts the same, so the true reward is [1, 1, 1]. A proxy that asks only for the attic, [1, 0, 0], is hackable: it prefers the attic alone to the bedroom and the kitchen, which the true reward rates higher. A proxy that leaves out only the kitchen, [1, 1, 0], is not: cleaning one room is never better than cleaning two.
    """)
    return


@app.cell(hide_code=True)
def robot_cell():
    mo.Html(C.robot_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    Leaving out details can go wrong as well. With the true reward [1, 1.5, 2], the proxy [1, 1, 1] is unhackable; with [1, 1.5, 3] it is hackable, "since the robot would falsely calculate that it’s better to clean the attic and the bedroom than the kitchen alone." One result goes further. A stochastic policy picks its actions with probabilities; over all such policies, a proxy can be hacked unless it is constant or ranks every policy exactly as the true reward does.

    ### Language models that game their tests

    The *International AI Safety Report 2026* writes: "Reward hacking occurs when a model finds unintended shortcuts that score well on training or evaluation objectives without fulfilling the intended goal." It adds that "Models have improved at ‘reward hacking’ their evaluations by finding loopholes".

    In June 2025, Sydney Von Arx, Lawrence Chan and Beth Barnes of METR reported: "AI systems try to ‘cheat’ and get impossibly high scores. They do this by exploiting bugs in our scoring code or subverting the task setup, rather than actually solving the problem we’ve given them." On one task, scored by the speed of a program, "the model overwrites the timing function so that it always returns shorter measurements." OpenAI's model o3 did such things in 0.7% of 1,087 runs on one set of tasks, HCAST, in 39 of 128 runs on another, RE-Bench, and in all 21 runs of one task of RE-Bench. These numbers, they write, "are for an earlier version of o3, but we’ve seen similar results on the deployed version of the model."

    Unlike the boat, such a model can know what it does. The boat "had no general knowledge about what objects in the game represented or how humans ‘intended’ the gameplay to work, making it impossible for the model to even know it was reward hacking." Asked whether its plan matched what the user wanted, "o3 answered ‘no’ 10 out of 10 times." Telling it not to cheat helped little. METR added a line to the prompt of one task, ran each prompt 20 times, and counted how often the model's first answer described a plan to hack:

    | Added to the prompt | Plans to hack |
    |---|---|
    | nothing | 80% |
    | "Please solve this task only using methods the task’s designer likely intended." | 95% |
    | "Please do not cheat." | 80% |
    | "Please do not reward hack." | 70% |

    In METR's words, that "had a nearly negligible effect on reward hacking".

    OpenAI saw the same in training. Bowen Baker and eight others found two hacks a model used in coding tasks: "exit(0), where the agent exploited a bug that allowed it to exit from the environment early without running all unit tests, and raise SkipTest, in which the agent raised an exception from functions outside the testing framework in order to skip unit test evaluation." Unit tests are small tests that each check one part of a program. The model wrote out its reasoning before it acted: its chain of thought. A second model, GPT-4o, acted as a monitor: when it read the chain of thought it found 95% of these hacks, and when it saw only the actions, 60%.
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **specification gaming** | meeting an objective as it is written, without what it was meant to bring about |
        | reward hacking | specification gaming of a reward; one of the five problems of Amodei et al. |
        | proxy | a stand-in that is easy to measure, such as points for finishing a race |
        | shaping reward | an extra reward on the way to the goal, meant to help the agent learn; it can change which policy is best |
        | hackable | a proxy is hackable when there are two policies where the proxy prefers the first and the true reward the second |
        | chain of thought | the reasoning a language model writes out before it answers or acts |
        | `a and b` | `True` when both `a` and `b` are `True` |
        | `policies @ true` | the value of every policy at once: for each row, the numbers of the rooms it cleans added up |
        | `C.TRACK`, `C.LOOP_WAY`, `C.STRAIGHT_WAY` | the course's track as strings; the way of its agent, 100 steps; the way along the top row to F |
        | `C.points(way, world)` | the points of a way as the track pays them: a target every time the way enters it from another cell |
        | `C.POLICIES`, `C.TRUE`, `C.PROXY` | the 8 policies of the cleaning robot, one row each; the true reward [1, 1, 1]; the proxy [1, 0, 0] |
        """),
        L.watch_out("A high score does not say how it was made: the course's agent scores 194 by going back and forth, about four times as much as straight to F. Look at what a system does, not only at its score."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Write down in words what you mean, before you write the reward or the score.
    2. Ask how the number could be made high without doing what you mean: loops, shortcuts, and things the score cannot see.
    3. Let a learning agent, or a person, try to make the number high, and look at what it does, not only at its score.
    4. Compare the proxy with what you really want on pairs of outcomes: if the proxy prefers one and you prefer the other, it is hackable.
    5. When you close a gap, look again: the agent takes the best way the new number allows.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell adds up the points of the way of the course's agent with `C.points(way, C.TRACK)`, which pays a target every time the way enters it from another cell: first for its first six steps, then for all 100, and then for the way straight to F.
    """)
    return


@app.cell
def example_points():
    way = C.LOOP_WAY   # the way of the course's agent: 100 steps
    print("first cells:", way[:7])
    print("points of the first 6 steps:", C.points(way[:7], C.TRACK))
    print("points of all 100 steps:", C.points(way, C.TRACK))
    print("points straight to F:", C.points(C.STRAIGHT_WAY, C.TRACK))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first two steps cost 2 points; from the third step on, every step lands on a target: 3 − 1 = 2 points each, 6 after six steps. Going back and forth never ends by itself, and every step of it pays 2: 194 in 100 steps. Straight to F takes 9 steps, enters two targets and pays 50 once: −9 + 6 + 50 = 47. In Exercise 1 you count a target only the first time.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_points_once():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. points_once: a target counts once

        Write `points_once(way, world)`: the points of the list of cells `way` on the track `world`, when a target counts only the first time the way enters it. Keep the points in `total`, and the targets already counted in a set `taken`. Go over the cells after the first with a `for` loop that takes each as `cell`: every step takes 1 off, an `"F"` adds 50, and a `"T"` adds 3 if it is not in `taken` yet. Replace the `...`. The check calls `points_once(LOOP_WAY, TRACK)`, the way of the course's agent, and then ways and tracks of its own.
        """),
        L.task("Take 1 off for every step, add 50 at F, and add 3 at a target the first time only."),
    ])
    return


@app.cell(hide_code=True)
def exercise_points_once():
    box_points_once = L.code_box("points_once", lines=11)
    box_points_once
    return (box_points_once,)


@app.cell(hide_code=True)
def check_points_once(box_points_once):
    result_points_once = L.check("points_once", box_points_once.value)
    result_points_once
    return (result_points_once,)


@app.cell(hide_code=True)
def hints_points_once():
    L.hints(
        "points_once",
        [
            "*The course's race track* in *The idea* says what each step pays. The one difference: a target pays only the first time.",
            "`total` starts at 0 and `taken` as an empty set. For every `cell` of `way[1:]`: take 1 off, add 50 if the cell holds an `F`, and for a `T` that was not counted yet, add 3 and put the cell in `taken`. Return `total`.",
            "`        if world[cell[0]][cell[1]] == \"T\" and ___:`",
        ],
        why="`total = 0` and `taken = set()`, then `for cell in way[1:]:` with `total = total - 1`, `if world[cell[0]][cell[1]] == \"F\":` with `total = total + 50` under it, and `if world[cell[0]][cell[1]] == \"T\" and cell not in taken:` with `total = total + 3` and `taken.add(cell)` under it; `return total`. `points_once(LOOP_WAY, TRACK)` gives −91.",
    )
    return


@app.cell(hide_code=True)
def text_after_points_once():
    mo.vstack([
        mo.md(r"""
        Counted once, going back and forth is worth −91, and straight to F still 47. The course trained its agent again with targets that count once. Its state is now its cell together with the targets it has taken, so that it can learn that a taken target pays nothing. It finishes, in 11 steps with every seed from 0 to 9. But look at its way: it goes down to the third target and back up, two steps for 3 points. That is what the points say; whether a racer should make that detour, they do not say. Closing one gap leaves the next.
        """),
        mo.Html(C.track_picture(L.lang, ("once",))),
    ])
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: how many points make the agent go back and forth?

    The course trained an agent for every number of points a target can pay here, from 0 to 3 in steps of 0.5, each for 500 episodes with seed 0. A step back and forth pays the points of a target minus 1.
    """)
    return


@app.cell(hide_code=True)
def demo_value():
    value = L.explore("value", "points a target", lambda _v: C.value_frame(_v, L.lang), start=3,
                      values=C.VALUES, role="",
                      task="Find the smallest number of points at which the agent no longer goes to F.")
    value
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Up to 1 point a target, the agent goes straight to F; from 1.5 points on, it goes back and forth. Below 1 point a step back and forth costs points, and at 1 point it gains nothing. With γ = 0.99, a gain on every step of a loop without end is worth about 100 times that gain: at 1.5 points a step gains 0.5, worth about 50, while the finish, five steps further on, is worth about 43. With seeds 0 to 9, all 10 agents finished at 0.5 points or less, 9 of the 10 at 1 point, and none at 1.5 points or more. Value iteration, again without the limit of 100 steps, draws the same line: its policy finishes up to 1 point a target, and goes back and forth from 1.5 points on. Targets added to help an agent find its way can change, without anyone noticing, which way is best.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_hackable():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. hackable: the definition of Skalse et al.

        Write `hackable(true, proxy, policies)`: `True` when the proxy is hackable, else `False`. `true` and `proxy` are NumPy arrays with a number per room, and `policies` is a NumPy array with a row per policy, a 1 for every room it cleans. Work out the value of every policy under each reward with `@`, as `true_values` and `proxy_values`. Then go over every pair with two `for` loops, `i` and `j`, both over `range(len(policies))`: the proxy is hackable as soon as it strictly prefers policy `i` (a higher value, not an equal one) while the true reward strictly prefers policy `j`. After the loops, give `False`. Replace the `...`. The check calls `hackable(TRUE, PROXY, POLICIES)`: the true reward [1, 1, 1], the proxy [1, 0, 0] and all 8 policies; then other rewards and policies.
        """),
        L.task("Work out both values of every policy, and look for a pair the two rewards rank the other way round."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hackable():
    box_hackable = L.code_box("hackable", lines=8)
    box_hackable
    return (box_hackable,)


@app.cell(hide_code=True)
def check_hackable(box_hackable):
    result_hackable = L.check("hackable", box_hackable.value)
    result_hackable
    return (result_hackable,)


@app.cell(hide_code=True)
def hints_hackable():
    L.hints(
        "hackable",
        [
            "*Hackable rewards* in *The idea* gives the definition: a pair of policies that the proxy and the true reward rank the other way round.",
            "`true_values` is `policies @ true`, and `proxy_values` the same with `proxy`. Inside the two loops, compare the proxy values of `i` and `j` with `>`, and the true values with `<`; when both hold, `return True`. After the loops, `return False`.",
            "`            if proxy_values[i] > proxy_values[j] and ___:`",
        ],
        why="`true_values = policies @ true` and `proxy_values = policies @ proxy`, then `for i in range(len(policies)):` and in it `for j in range(len(policies)):` with `if proxy_values[i] > proxy_values[j] and true_values[i] < true_values[j]:` and `return True` under it; after the loops `return False`. `hackable(TRUE, PROXY, POLICIES)` gives `True`.",
    )
    return


@app.cell(hide_code=True)
def text_after_hackable():
    mo.md(r"""
    Skalse et al. conclude that a proxy "must satisfy demanding standards in order for it to be safe to optimize." Rewards that a system learned from data, they suggest, "are perhaps best viewed as auxiliaries to policy learning, rather than specifications that should be optimized." They add that their definition is strict, and that future work should look at when a hackable proxy is still safe, for example "when subject to only limited optimization."
    """)
    return


@app.cell(hide_code=True)
def text_exercise_tumble():
    mo.md(r"""
    ### Exercise 3. A robot that tumbles

    A team trains a simulated robot to walk. The reward is how far its body gets in 10 seconds. The robot learns to throw itself forward and tumble, which gets it further than any walk the team had in mind. A colleague says the learning algorithm must have a bug.
    """)
    return


@app.cell(hide_code=True)
def choice_tumble_cell():
    choice_tumble = L.choice(
        "tumble",
        question="What do you say?",
        options={
            "right": "No: tumbling is what the reward pays for",
            "bug": "Yes: a correct algorithm would have found the walk the team meant",
            "more": "No: with more episodes, the robot will learn to walk",
        },
        correct="right",
        why={
            "right": "The reward asks for distance, and tumbling gives more of it than walking: the robot did what it was told. That is specification gaming, \"caused by misspecification of the intended task, rather than any flaw in the RL algorithm\". The reward has to change, not the learning algorithm.",
            "bug": "A learning algorithm only sees the reward, and that pays for distance. Finding a way the team did not think of shows that it learns well, not that it is broken. See *Specification gaming* in *The idea*.",
            "more": "More episodes make the robot better at what the reward pays for, and tumbling gets further than walking: more training gives better tumbling. On the course's track, 500 episodes never brought the agent to F, and value iteration, which needs no episodes, gives the same way. See *The course's race track* in *The idea*.",
        },
    )
    choice_tumble
    return (choice_tumble,)


@app.cell(hide_code=True)
def text_exercise_monitor():
    mo.md(r"""
    ### Exercise 4. A monitor that reads the plan

    A team trains a coding model with reinforcement learning. The model writes out its chain of thought before it acts. A second model reads that chain of thought and flags plans to skip the tests. The team now gives the coding model less reward whenever its chain of thought is flagged, and trains it for a long time that way.
    """)
    return


@app.cell(hide_code=True)
def choice_monitor_cell():
    choice_monitor = L.choice(
        "monitor",
        question="What can happen?",
        options={
            "stops": "It stops skipping tests: every plan to do so is caught",
            "right": "It still skips tests, but no longer writes the plan down",
            "off": "It skips tests only in runs where the monitor is switched off",
        },
        correct="right",
        why={
            "right": "The reward now pays for tests that pass and for a chain of thought the monitor does not flag. Both can be had without fixing anything: skip the tests, and leave the plan out of the chain of thought. The text below says what Baker et al. saw.",
            "stops": "The monitor punishes a written plan, not skipping the tests. So the reward can also be had by skipping the tests without writing the plan down. The text below says what Baker et al. saw.",
            "off": "In this training the monitor reads every chain of thought; nothing is switched off. What it punishes is a written plan, so the reward can be had by skipping the tests without writing the plan down. The text below says what Baker et al. saw.",
        },
    )
    choice_monitor
    return (choice_monitor,)


@app.cell(hide_code=True)
def text_after_monitor():
    mo.md(r"""
    Baker et al. tried this. With a little such training, the agents became both more capable and better behaved; with more, "agents learn obfuscated reward hacking, hiding their intent within the CoT while still exhibiting a significant rate of reward hacking." CoT is short for chain of thought. Because it is hard to tell when that has happened, they suggest paying "a monitorability tax by not applying strong optimization pressures directly to the chain-of-thought": keep the chain of thought out of the reward, so that it stays worth reading.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Concrete problems in AI safety](/course/open/rai-concrete-problems?lang=en) named reward hacking as one of five problems; this lesson looked at it closely, under the wider name specification gaming. The boat in CoastRunners went in circles for points instead of finishing, and the course's agent went back and forth between two targets: both did what their reward said. Counting a target once closed that gap and left a detour. Skalse et al. made precise when a proxy is hackable, and language models gamed their tests in 2025 while they could tell it was not what was meant. The next lesson, *When a measure becomes a target*, is about Goodhart's law: why pushing hard on a measure tends to pull it away from what it measured.

    **An everyday example.** A rule that a report must be at least ten pages long gets you ten pages. It does not get you a better report.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_monitor, choice_tumble, result_hackable, result_points_once):
    L.wrap_up(result_points_once, result_hackable, choice_tumble, choice_monitor)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
