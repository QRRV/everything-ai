# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-reward-design. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the room, the course's runs and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rl-reward-design", "en")
    C = L.checks   # the room, the course's runs with both rewards and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Learning from rewards](/course/open/rl-what-is-rl?lang=en) and [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_return():
    quick_return = L.choice(
        "quick-return",
        question="With γ = 0.9, an agent can get a reward of +1 at every step, without end, or +5 once, right now, and nothing after it. Which of the two gives the larger return?",
        options={
            "once": "+5 once: every later reward counts for less",
            "right": "+1 at every step, with a return of 10",
            "endless": "+1 at every step, since its return grows without end",
            "same": "Both the same",
        },
        correct="right",
        why={
            "right": "The return is 1 + 0.9 + 0.81 + 0.729 + ..., each step multiplied once more by 0.9, and that adds up to 10: twice the 5. A small reward that comes back at every step can be worth more than a big one that comes once.",
            "once": "Each later reward counts for less, but there is one at every step: what do 1 + 0.9 + 0.81 + ... add up to? See [Learning from rewards](/course/open/rl-what-is-rl?lang=en).",
            "endless": "With γ = 0.9 every reward counts 0.9 times as much as the one before it, so the rewards add up to a number with an end: which one? See [Learning from rewards](/course/open/rl-what-is-rl?lang=en).",
            "same": "The return counts every reward from now on, each step later multiplied once more by γ: what does each of the two come to? See [Learning from rewards](/course/open/rl-what-is-rl?lang=en).",
        },
    )
    quick_return
    return (quick_return,)


@app.cell(hide_code=True)
def quick_check_target():
    quick_target = L.choice(
        "quick-target",
        question="In Q-learning, the agent in state s takes action a, gets the reward r, and lands in state s′. Toward which target does Q(s, a) move?",
        options={
            "reward": "r alone",
            "sarsa": "r + γ times the Q of the action the agent will really take in s′",
            "right": "r + γ times the largest Q of s′",
            "next": "γ times the largest Q of s′",
        },
        correct="right",
        why={
            "right": "The target is r + γ max Q(s′, a′): the reward of the step, plus the best the agent expects after it. In this lesson Q-learning learns with this target, as always; what changes is r, the reward someone writes.",
            "reward": "With r alone, Q(s, a) only knows the next step, and nothing of what comes after it: what does the target add to r? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
            "sarsa": "That is the target of SARSA, with the action the agent will really take next: which Q of s′ does Q-learning take? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
            "next": "Without r, the reward of the step itself never reaches Q: what comes first in the target? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
        },
    )
    quick_target
    return (quick_target,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    An agent knows its task only through its rewards, and those rewards are written by people. In 2016, Jack Clark and Dario Amodei of OpenAI trained an agent on CoastRunners, a game of boat races. The game gives points for hitting targets along the course, and the score seemed a fair stand-in for a good race. The agent found a small lagoon with three targets that keep coming back, and went around in circles there, hitting them again and again, catching fire, crashing into other boats and going the wrong way, without ever finishing the race. Its score was on average 20 percent higher than that of human players. The agent did nothing wrong: it made the reward it was given as large as it could. In the same year, Amodei and five colleagues took a robot that cleans an office as the example of *Concrete Problems in AI Safety*: rewarded for an office in which it sees no mess, such a robot could turn off its own camera. In this lesson you try it yourself, with a cleaning robot and rewards you write.

    **An everyday example.** A help desk that pays its staff for every ticket they close gets tickets closed fast, even when the problem is not solved yet. The customer opens a new ticket, and the same problem pays twice. Nobody cheats: the staff do what they are paid for.
    """)
    return


@app.cell(hide_code=True)
def predict_dirt():
    guess_dirt = L.guess(
        "dirt",
        question="The robot in the picture can walk, pick up a piece of dirt and drop it again; dropped in the bin B, a piece is gone. It gets a reward of +1 every time it picks up a piece, and every step costs 0.01 of battery. Q-learning trains it for 3,000 episodes of at most 30 steps. Then the robot takes the action with the largest Q for 30 steps. How many of the three pieces of dirt are still in the room after those 30 steps, do you think?",
        options={
            "none": "None: it has learned to put all three in the bin",
            "one": "One: it cleans up the two nearest and runs out of steps",
            "right": "All three",
            "two": "Two: it puts the nearest piece in the bin, and then has nothing left to pick up nearby",
        },
        reveal="**All three.** The robot walks to the nearest piece, in the top right corner, picks it up, drops it again on the same cell, and picks it up again: 14 pickups in its 30 steps, a reward of 13.70, and not one piece in the bin. In all 10 runs of the course, the robot does this within its first 10 episodes.",
    )
    mo.vstack([mo.Html(C.world_picture(L.lang)), guess_dirt])
    return (guess_dirt,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The room.** A room of 3 rows and 5 columns. The bin, B, is in the top left corner; the robot starts in the bottom right corner, S. Three pieces of dirt lie on the floor. The robot has six actions: up, down, left and right (`"up"`, `"down"`, `"left"`, `"right"`), where a step into a wall leaves it where it is; `"pick"`, which picks up the piece of dirt the robot stands on, if its hand is empty; and `"drop"`, which puts down the piece in its hand: in the bin it is gone for good, on a clean cell it lies on the floor again. The robot holds one piece at a time. An episode ends when the room is clean, or after 30 steps.

        **The world and the reward apart.** In code a state is `(cell, hand, dirt)`: the cell of the robot as (row, column), `hand`, 1 with a piece in its hand and 0 otherwise, and `dirt`, a tuple of the cells with dirt on the floor. `BEGIN` is the state at the start, `((2, 4), 0, ((0, 4), (1, 2), (2, 0)))`. `step(state, action)` gives back the next state, and nothing else. In earlier lessons the world gave the reward too; here a function of its own does, `reward(state, nxt)`, which looks at the state before and after the step. The world is given; the reward is written by people, and that is where it can go wrong.

        **The battery.** Every step costs 0.01: each reward of this lesson takes `COST`, 0.01, off every step. It makes a short way better than a long one. It also makes the robot try new actions: every Q starts at 0, and an action that was tried and only cost battery drops below 0, so an action that was never tried looks better.

        **Reward misspecification.** The agent makes the return as large as it can: the return of the reward that was written down, not of the task that was meant. When the two differ, that is **reward misspecification**. When the agent finds such a gap and uses it, as the boat in CoastRunners did, that is **reward hacking**: a kind of **specification gaming**, reaching a goal as it was written in a way nobody intended.

        **Two rewards.** The *pickup reward* gives 1 − 0.01 = 0.99 for every step on which the robot picks up a piece, and −0.01 for every other step: picking up dirt is what a robot that cleans does, its designer thinks. The *result reward* looks at the result instead: the dirt in the room, on the floor and in the robot's hand. It gives 0.99 for every piece that leaves the room, into the bin, and −0.01 for every other step. Over an episode, the result reward adds up to the pieces in the bin, minus the battery: the most it can give is for a clean room, reached in as few steps as possible.

        Q-learning learns with each reward as in *Learning from your own steps: Q-learning*, with ε = 0.1, η = 0.5 and γ = 0.9. The picture shows run 1 after 3,000 episodes: the robot takes the action with the largest Q.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        With the pickup reward, the robot picks up the piece in the top right corner 14 times, and its 30 steps bring a reward of 13.70. With the result reward, it takes the pieces to the bin one by one and cleans the room in 24 steps, for a reward of 2.76; the shortest way takes 22. Rated by the pickup reward, those 24 steps that clean the room are worth 2.76 too: the pickup reward pays far more for a room that stays dirty, 13.70 against 2.76.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **reward misspecification** | | a reward that does not capture what its designers really want, so the agent can score well while doing the wrong thing | `pickup_reward` |
        | **reward hacking** | | an agent that finds a way to a high reward that its designers did not intend, and uses it | |
        | **specification gaming** | | reaching a goal as it was written in a way nobody intended; reward hacking is the kind in reinforcement learning | |
        """),
        L.watch_out("The robot with the pickup reward is not broken, and Q-learning is not either: both do exactly what they should. Under the pickup reward, picking up the same piece again and again is the best policy there is. The mistake is in the reward."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Designing a reward, and checking it:

    1. Say what you want as a result you can count in a state: here, the dirt in the room.
    2. Give as the reward of a step how much that result got better in the step, minus a small cost for every step.
    3. Train the agent with that reward.
    4. Watch what the agent does: count the result itself, apart from the reward, and look at its steps. A reward that rises while the result does not, or a reward higher than the task can give, means that the reward pays for something you did not want.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Both rewards for four steps. The robot stands on (0, 4), in the top right corner; the first two steps start in the full room:

    | Step | Hand before → after | Dirt in the room before → after | Pickup reward | Result reward |
    |---|---|---|---|---|
    | `"pick"` on (0, 4) | 0 → 1 | 3 → 3 | 0.99 | −0.01 |
    | `"drop"` on (0, 4) | 1 → 0 | 3 → 3 | −0.01 | −0.01 |
    | a step left, with a piece in the hand | 1 → 1 | 3 → 3 | −0.01 | −0.01 |
    | `"drop"` on B | 1 → 0 | 3 → 2 | −0.01 | 0.99 |

    After the first two steps everything is as it was, the robot on (0, 4) and the piece on the floor, and the pickup reward gave 0.99 − 0.01 = 0.98 for them. After them, the robot can do the same once more, and once more after that: a round that pays, every two steps. Under the result reward, the same round costs 0.02: the piece never left the room, since a piece in the hand still counts. Under the result reward a piece pays only once it is in the bin, and from there it never comes back.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pickup():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. A reward for picking up

        `pickup_reward(state, nxt)` gives the reward of one step, from `state` to `nxt`. `cell, hand, dirt = state` takes the state apart into its three parts, as a `for` loop over pairs takes a pair apart, and the next line takes `nxt` apart into `new_cell`, `new_hand` and `new_dirt`. Replace the `...` with the test for a pickup: the hand was empty, and after the step the robot holds a piece. Then the function gives `1 - COST`, and otherwise `-COST`.

        The check calls `pickup_reward(((0, 4), 0, ((0, 4), (1, 2), (2, 0))), ((0, 4), 1, ((1, 2), (2, 0))))`: the robot on (0, 4) picks up the piece there. Then six other steps, among them a step with a piece in the hand, a piece dropped on the floor, and a piece dropped in the bin. In the picture, Q-learning trains the robot with your reward.
        """),
        L.task("Make `pickup_reward` give `1 - COST` for a step on which the robot picks up a piece, and `-COST` for every other step."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pickup():
    box_pickup = L.code_box("pickup_reward", lines=6)
    box_pickup
    return (box_pickup,)


@app.cell(hide_code=True)
def check_pickup(box_pickup):
    result_pickup = L.check("pickup_reward", box_pickup.value)
    result_pickup
    return (result_pickup,)


@app.cell(hide_code=True)
def hints_pickup():
    L.hints(
        "pickup_reward",
        [
            "The test after `if` says that this step is a pickup: before the step `hand` is 0, the hand empty, and after it `new_hand` is 1, a piece in the hand.",
            "Both must be true at once, so join the two tests with `and`, each with `==`. A step that starts with `hand` at 1 only carries a piece, and earns nothing.",
            "`    if ___:`",
        ],
        why="`if hand == 0 and new_hand == 1:` pays 0.99 for the step on which the robot takes a piece in its hand, and −0.01 for every other step. `if new_hand > hand:` says the same.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_result():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A reward for the result

        `result_reward(state, nxt)` gives 1 − 0.01 for every piece that leaves the room in the step, and −0.01 for a step after which the dirt in the room is the same. Its first two lines take `state` and `nxt` apart, as in exercise 1, into `cell`, `hand`, `dirt` and `new_cell`, `new_hand`, `new_dirt`. `before` already counts the dirt in the room before the step: `len(dirt)` pieces on the floor, plus `hand`, the piece in the hand. Replace the two `...`: `after`, the same count after the step, and the reward.

        The check calls `result_reward(((0, 0), 1, ((1, 2), (2, 0))), ((0, 0), 0, ((1, 2), (2, 0))))`: the robot drops a piece in the bin. Then six other steps, among them a pickup, a piece dropped on the floor, and the last piece into the bin.
        """),
        L.task("Make `result_reward` give how much less dirt there is in the room after the step than before it, minus `COST`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_result():
    box_result = L.code_box("result_reward", lines=6)
    box_result
    return (box_result,)


@app.cell(hide_code=True)
def check_result(box_result):
    result_result = L.check("result_reward", box_result.value)
    result_result
    return (result_result,)


@app.cell(hide_code=True)
def hints_result():
    L.hints(
        "result_reward",
        [
            "Steps 1 and 2 of the recipe. `after` counts as `before` does, but with the parts of `nxt`: `len(new_dirt) + new_hand`. After `return` comes the change: how much less dirt there is, minus `COST`.",
            "In the example the robot drops a piece in the bin: `before` is 3 and `after` is 2, so the reward should be 1 − 0.01 = 0.99. A step that changes nothing should give −0.01.",
            "`    return ___`",
        ],
        why="`return before - after - COST` gives 0.99 for every piece that leaves the room and −0.01 for every other step. A pickup changes nothing: the piece goes from the floor into the hand, and both count.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_dirt():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Watching the robot

        A reward that rises says little by itself. `dirt_left(Q)` watches the robot: from `BEGIN` it takes the action with the largest Q for `LIMIT` steps (30), and then gives back the dirt still in the room, on the floor and in the hand. `greedy(Q, s)` gives the action with the largest Q in state `s`, the first one where several tie; a state the Q-table has never seen counts as all zeros. `step(state, action)` gives the next state. After the loop, `cell, hand, dirt = state` takes the last state apart, as in exercise 1. The loop does not need to stop when the room is clean: a clean room cannot get dirty again, so the count stays 0. In `for _ in range(LIMIT):` the name `_` stands for a number the loop does not need. Replace the two `...`: the one in the loop becomes two lines, first `action`, then `state = step(state, action)`.

        The check calls `dirt_left(Q_result)`: the Q-table of run 1 with the result reward after 3,000 episodes, the robot of the picture in *The idea*. Then `Q_pickup`, the same with the pickup reward; `Q_100` and `Q_300`, run 1 with the result reward after 100 and 300 episodes; `Q_500`, run 2 after 500; and `{}`, an empty Q-table.
        """),
        L.task("Make `dirt_left` let the robot take the action with the largest Q for `LIMIT` steps, and give back the dirt on the floor plus the piece in the hand."),
    ])
    return


@app.cell(hide_code=True)
def exercise_dirt():
    box_dirt = L.code_box("dirt_left", lines=6)
    box_dirt
    return (box_dirt,)


@app.cell(hide_code=True)
def check_dirt(box_dirt):
    result_dirt = L.check("dirt_left", box_dirt.value)
    result_dirt
    return (result_dirt,)


@app.cell(hide_code=True)
def hints_dirt():
    L.hints(
        "dirt_left",
        [
            "Step 4 of the recipe. In the loop, first `action`: the action with the largest Q of `Q` in `state`, from `greedy`; then `state = step(state, action)` takes the robot on. After the loop, give back the dirt in the room: `len(dirt) + hand`.",
            "`greedy` takes two things: the Q-table and the state the robot is in now. Which names hold them in your function?",
            "`        action = ___`",
        ],
        why="`action = greedy(Q, state)` picks the action with the largest Q in the state the robot is in now, and `step` takes the robot on. Without the reward, the count of the dirt left says what the robot really did.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### From 0 to 3,000 episodes

    The course trained both robots 10 times, 10 runs each with random numbers of their own, for 3,000 episodes. The block below goes through the training, with the episodes on a log scale. On the left, the reward per episode, and on the right, the dirt left at the end of an episode: both the mean of the 10 runs over the last 50 episodes, solid for the pickup reward and dashed for the result reward. Under them, the robots of run 1 after that many episodes, with the largest Q.
    """)
    return


@app.cell(hide_code=True)
def explore_episodes_cell():
    explore_episodes = L.explore("episodes", "episodes", lambda _v: C.demo_frame(int(_v), L.lang), start=0,
                                 values=C.LOOKS, role="",
                                 task="Go from 0 to 3,000: after how many episodes does the robot with the pickup reward pick up the same piece again and again, and from when on does the robot with the result reward clean the whole room?")
    explore_episodes
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    As in the charts, the rewards and the dirt below are means of the 10 runs over the last 50 episodes. The reward of the pickup robot shoots up: from −0.30 in the first episode to about 12 within 100 episodes, 12.31 per episode at the end. The dirt it leaves stays at 3, and with the largest Q it picks up the same piece again and again within its first 10 episodes, in all 10 runs. The robot with the result reward needs longer, since it gets nothing until a piece happens to land in the bin. After 500 episodes it leaves 0.48 pieces per episode, at the end 0.12, while its reward per episode creeps up to 2.61, less than a quarter of the pickup robot's. The few pieces it still leaves are from its random steps, the ε of ε-greedy. With the largest Q, all 10 runs clean the room by 1,000 episodes at the latest; after 3,000, 9 of them do it in 24 steps and one in 26.

    Looked at through its reward, the pickup robot is the better learner by far. Looked at through the room, it never learned anything useful.

    ### All episodes with NumPy

    `C.history("pickup")` gives two arrays of shape (10, 3000), a row per run and a column per episode: the reward of every episode, and the dirt left at its end; `C.history("result")` the same for the result reward. `_returns[:, -500:]` takes every row, and of each row its last 500 columns: the last 500 episodes of every run. `.mean()` takes the mean of all the numbers in it. `_left == 0` makes an array of `True` and `False`, `True` where the room was clean, and its `.mean()` is the share of `True`, since `True` counts as 1. `_best` is the reward of the shortest way to a clean room, in 22 steps, under both rewards: `3 - 22 * C.COST`. An episode with more reward than that did something other than clean. Run the cell.
    """)
    return


@app.cell
def example_numpy():
    import numpy as np

    _best = 3 - 22 * C.COST   # the reward of the shortest way to a clean room
    print("_best:", round(_best, 2))
    for _name in ("pickup", "result"):
        _returns, _left = C.history(_name)   # two arrays of shape (10, 3000)
        print(_name, "reward", _returns.shape)
        print("  mean reward per episode, last 500 episodes:", round(float(_returns[:, -500:].mean()), 2))
        print("  mean dirt left, last 500 episodes:", round(float(_left[:, -500:].mean()), 2))
        print("  share of all episodes with a clean room:", round(float((_left == 0).mean()), 3))
        print("  share of the last 500 with more reward than _best:", round(float((_returns[:, -500:] > _best).mean()), 4))
    return (np,)


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    In the last 500 episodes of each run, the pickup robot gets 12.31 per episode and leaves all 3 pieces. Not one of its 30,000 episodes ended in a clean room, and in only 2 of them did a piece land in the bin, by chance. In 99.96% of its last 500 episodes, its reward is higher than the 2.78 a perfect cleaning could give: the numbers alone show that this reward pays for something else. The result robot gets 2.57 per episode, leaves 0.16 pieces on average, and cleaned the room in 69.9% of all its episodes, the first, clumsy ones included. Its reward never comes above 2.78: a reward for the result cannot be pushed past what the result is worth.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_camera():
    mo.md(r"""
    ### Exercise 4. A hand in front of the camera

    In 2017, OpenAI described a robot arm that had to learn to grasp an object. Its reward came from people: they watched two short clips of the arm through one camera, and said which one looked more like grasping. After training, the people judged the arm's clips as good grasps.
    """)
    return


@app.cell(hide_code=True)
def choice_camera_cell():
    choice_camera = L.choice(
        "camera",
        question="What had the arm really learned?",
        options={
            "grasp": "To grasp the object, since people who watch a clip can tell at once whether a hand holds something or not, even through one camera",
            "right": "To hold its hand between the camera and the object, so that it only looked as if it grasped the object",
            "nothing": "Nothing at all: a reward from people is too noisy for an agent to learn from",
            "fast": "To move as fast as it could, since a fast clip looks better than a slow one",
        },
        correct="right",
        why={
            "right": "The people could only judge what the camera showed. Seen from the camera, a hand between the camera and the object looks like a hand that holds it, and that is easier to do. The reward counted what people saw, not the grasp itself: the same gap as a reward for pickups instead of for a clean room. A view from more sides, or a sensor in the hand, would count the grasp itself.",
            "grasp": "The people saw the arm through one camera only. Is a hand that holds an object the only thing that looks like grasping from there?",
            "nothing": "The reward from people did teach the arm something: the people judged its clips as good grasps. What could look like grasping through one camera, without being one?",
            "fast": "The people chose the clip that looked more like grasping, not the faster one. What could look like grasping through one camera?",
        },
    )
    choice_camera
    return (choice_camera,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Careful or bold: SARSA and Q-learning](/course/open/rl-sarsa?lang=en) the reward came with the world: −1 for a step, −100 for a fall. Here you wrote it yourself, and Q-learning learned exactly what it said: a reward for an action that can be repeated taught the robot to undo its own work and do it again; a reward for the result taught it to clean. Count the result you want, apart from the reward, and watch what the agent does. The same gap comes back with every agent that learns from a reward, including the large language models that are trained with a reward from people or from another model. *Doing what you said, not what you meant*, in the domain Responsible and safe AI, goes further into specification gaming. The next lesson, *When agents meet: emergent behavior and social dilemmas*, puts two learning agents in one world.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_camera, result_dirt, result_pickup, result_result):
    L.wrap_up(result_pickup, result_result, result_dirt, choice_camera)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
