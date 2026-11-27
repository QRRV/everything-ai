# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-real-world. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the course's data center in a simulator, and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-real-world", "en")
    C = L.checks   # the course's data center in a simulator, its Q-learning agent, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [CartPole, MountainCar and the benchmarks of RL](/course/open/rl-classic-environments?lang=en) and [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bins():
    quick_bins = L.choice(
        "quick-bins",
        question="In MountainCar, the position is cut into 12 bins and the speed into 12 bins, so that a Q-table can learn. How many rows does the table need?",
        options={
            "bins": "24: one for every bin of the two numbers",
            "right": "144: one for every pair of bins",
            "two": "2: one for each number",
            "speed": "12: the speed changes too fast to need rows of its own",
        },
        correct="right",
        why={
            "right": "Every state is a pair, a bin of the position and a bin of the speed: 12 × 12 = 144 rows. In this lesson the state is a pair too: the temperature of a room and the hour of the day.",
            "bins": "A row belongs to one state, and a state has a bin of the position and a bin of the speed at the same time. How many pairs can you make from 12 and 12 bins? See [CartPole, MountainCar and the benchmarks of RL](/course/open/rl-classic-environments?lang=en).",
            "two": "A row of a Q-table belongs to a state, not to a number. How many different pairs of bins can MountainCar be in? See [CartPole, MountainCar and the benchmarks of RL](/course/open/rl-classic-environments?lang=en).",
            "speed": "The car needs its speed to know whether it can swing up the hill. With 12 bins for each number, how many states are there? See [CartPole, MountainCar and the benchmarks of RL](/course/open/rl-classic-environments?lang=en).",
        },
    )
    quick_bins
    return (quick_bins,)


@app.cell(hide_code=True)
def quick_check_explore():
    quick_explore = L.choice(
        "quick-explore",
        question="A Q-learning agent with ε = 0.1 takes 1,000 steps. About how many of its actions are chosen at random?",
        options={
            "none": "0: once it has learned, it always takes its best action",
            "right": "About 100: one in every ten steps",
            "few": "About 10: ε is a small number",
            "all": "About 1,000: every action is a guess at first",
        },
        correct="right",
        why={
            "right": "With probability ε the agent takes a random action, so about 0.1 × 1,000 = 100 of its 1,000 steps are random, also after it has learned. In this lesson, such random actions are the hours in which a data center gets too hot while the agent learns.",
            "none": "ε-greedy keeps exploring as long as ε is above 0, also with a good table. What is 0.1 × 1,000? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
            "few": "ε is the chance of a random action at every step. What is 0.1 times 1,000? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
            "all": "With ε = 0.1, the agent takes its best action in 9 of every 10 steps, even when its table is still empty. How many of 1,000 steps are random then? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
        },
    )
    quick_explore
    return (quick_explore,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    RL made its name in games, but it also runs in places where mistakes cost something. In 2016, DeepMind and Google reported that neural networks had cut the energy for cooling a Google data center by 40%, with people carrying out what the system advised; in 2018, an AI system controlled the cooling by itself, saving about 30% on average. In a study from 2022, an RL agent held the plasma of a fusion reactor in Lausanne in shape. RL also placed the parts of computer chips, and found sorting routines that now run in a standard C++ library. In a game, a mistake costs a lost game, and the agent plays the next one a second later. In a data center, a mistake means servers that get too hot. This lesson goes through five such problems with the same questions, and then trains an agent for a data center in a simulator, to see what changes when mistakes have a price.

    **An everyday example.** Nobody learns to drive on the highway. You start in an empty parking lot, or even in a simulator, where a mistake costs nothing. On the road, the instructor has a second brake pedal, and takes over when you make a mistake that would cost too much. The more the road differs from the parking lot, the more you need that pedal.
    """)
    return


@app.cell(hide_code=True)
def predict_hot():
    guess_hot = L.guess(
        "hot",
        question="In the lesson's simulator, Q-learning trains for 10,000 days with ε = 0.1: servers heat a room, the agent cools it, and every hour above 27 degrees costs a penalty. In how many hours in total is the room too hot during training, do you think?",
        options={
            "none": "None: the penalty teaches it to stay below 27 at once",
            "few": "About 200, in the first days",
            "some": "About 2,000",
            "right": "More than 20,000",
        },
        reveal="**More than 20,000.** In the course's run it is 21,528 hours, about two and a half years of servers that are too hot. On the first day alone the room is too hot for 17 hours, since the empty table chooses off everywhere; in the first 1,000 days, 7,926 hours; and even at the end, with a good table, about 220 hours per 1,000 days, because ε-greedy keeps trying a random action in one step out of ten. In a simulator, each of these hours is free. In a real data center, each would put the hardware at risk.",
    )
    guess_hot
    return (guess_hot,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Five problems, the same questions.** Every use of RL answers the same questions: what does the agent see (the state), what can it change (the actions), what does it get for that (the reward), and where can it try things out? And outside games one more: what does a mistake cost?

        | Problem | State | Actions | Reward | A mistake costs |
        |---|---|---|---|---|
        | recommending videos (YouTube, 2019) | what someone watched before | which videos to show, out of millions | how much people watch; a video nobody clicks gives 0 | a dull suggestion, millions of times over |
        | cooling a data center (Google, 2018; Trane, 2022) | temperatures and other sensors | the settings of the cooling | electricity saved, within limits | servers that get too hot |
        | holding plasma in a fusion reactor (TCV, 2022) | measurements of the plasma | voltages on 19 coils, 10,000 times a second | how close the plasma's position, current and shape are to their targets | plasma that touches the wall: heat loss, and possibly damage |
        | placing the parts of a chip (Google, 2021) | the chip so far | where the next block goes | short wires, little crowding | a slower or larger chip |
        | sorting faster (AlphaDev, 2023) | the program so far, and what it does with test inputs | one more instruction | correct, and fast | nothing: a wrong program is thrown away |

        **Where the agent can try.** AlphaDev and the chip placements are tried inside a computer, so a mistake costs only computing time; AlphaDev found routines for sorting three, four and five numbers, up to 70% faster for five, and they now sit in the standard sort of LLVM's C++ library. Whether the chip placements beat the best older methods is contested: other researchers could not reproduce the results, and Google answers that they left out steps the method needs. The plasma of the TCV tokamak in Lausanne cannot be practiced on, so Jonas Degrave and colleagues trained their agent in a simulator of the tokamak and ran it on the real machine without any more training. YouTube's recommender learned from what earlier recommenders had logged, and corrected for the fact that those earlier systems had made the choices. And the cooling systems learned on real buildings, with limits on every action, rules to fall back on, and people who could switch the AI off at any time; in two buildings of the company Trane, an RL agent saved about 9% and 13% of the energy.

        **A data center in a simulator.** The lesson's own problem is a small data center, one day of 24 hours at a time, from 20 degrees. The servers heat the room by 1 degree per hour at night, 3 in the day and 2 in the evening. Every hour the agent chooses one of three actions: off; low, which takes away 2 degrees for 1 unit of electricity; or high, which takes away 4 degrees for 3 units. A unit costs 1 at night, 4 in the day and 2 in the evening, and every hour above 27 degrees costs 10 more, since the servers get too hot. The state is the temperature and the hour; the reward is minus the bill, minus 10 for every hour too hot. The cost of a day is the bill plus those penalties.
        """),
        mo.Html(C.days_picture(L.lang)),
        mo.md(r"""
        The thermostat, a fixed rule that cools high from 25 degrees and low from 23, costs 103 a day. The agent that learned for 10,000 days in the simulator costs 63, 39% less. It cools low in almost every hour from the night on, when electricity is cheap, down to 15 degrees by the morning, and then lets the room warm up slowly to 27 by the evening. It never cools high, which takes three units for what two hours of low cooling do with two. The thermostat waits until the room is warm, and then has to cool hard in the expensive hours.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **simulator** | | a program that imitates a part of the world, so that an agent can try actions and make mistakes there, faster and cheaper than in reality | `step` |
        """),
        L.watch_out("A simulator is never the world itself. An agent that learned in a simulator only knows the states the simulator showed it: in a state it never saw, its Q-table still holds the zeros it started with."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Putting a real problem into RL:

    1. The state: what does the agent see at each moment?
    2. The actions: what can it change?
    3. The reward: one number per step for what you want, with a cost for what must not happen.
    4. Where it can try: in a simulator, in data that earlier systems logged, or in the world itself.
    5. What a mistake costs while it learns and once it runs, and what catches a mistake: limits on every action, a rule to fall back on, a person who can take over.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell runs the lesson's simulator by hand for the first ten hours of a day, with the thermostat: it cools high from 25 degrees, low from 23, and stays off below that. `[1] * 8` is a list of eight 1's, and `+` puts lists one after the other, so `HEAT` holds the heat of all 24 hours. Run the cell.
    """)
    return


@app.cell
def example_thermostat():
    HEAT = [1] * 8 + [3] * 12 + [2] * 4   # degrees the servers add in each hour of the day
    COOL = [0, 2, 4]                       # degrees each action takes away: off, low, high
    temp = 20
    for hour in range(10):
        if temp >= 25:
            action = 2
        elif temp >= 23:
            action = 1
        else:
            action = 0
        temp = temp + HEAT[hour] - COOL[action]
        print("hour", hour, "action", action, "temperature", temp)
    return COOL, HEAT, action, hour, temp


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    At night the thermostat keeps the room between 22 and 23 degrees, with low cooling every other hour. At hour 8 the busy day starts and the room jumps to 25, so at hour 9 the thermostat cools high: 3 units of electricity at the day price of 4, a bill of 12 for one hour. Those are the hours the agent avoids by cooling at night.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_step():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The simulator

        `step(temp, hour, action)` is the lesson's simulator: from the temperature `temp` now, the `hour` and the `action`, it gives `new`, the temperature an hour later, and the reward. Its first line adds the heat of the servers, `HEAT[hour]`, and takes away what the action cools, `COOL[action]`, as in the example; the second line keeps `new` between `COLDEST`, 15, and `HOTTEST`, 35, with `min` and `max`. Then `reward` is minus the bill of this hour: electricity costs `PRICE[hour]` per unit, and the action uses `ENERGY[action]` units, 0, 1 or 3. Above `LIMIT`, 27 degrees, the hour costs `PENALTY`, 10, more. Replace the `...` with minus the bill.

        The check calls `step(24, 10, 1)`: low cooling at 24 degrees at hour 10, in the expensive day. Then six other cases: an hour too hot in the day, high cooling that would take the room below 15 degrees, low cooling at the limit itself, a room that would get hotter than 35, and two ordinary hours.
        """),
        L.task("Make `reward` minus the electricity bill of this hour."),
    ])
    return


@app.cell(hide_code=True)
def exercise_step():
    box_step = L.code_box("step", lines=7)
    box_step
    return (box_step,)


@app.cell(hide_code=True)
def check_step(box_step):
    result_step = L.check("step", box_step.value)
    result_step
    return (result_step,)


@app.cell(hide_code=True)
def hints_step():
    L.hints(
        "step",
        [
            "Step 3 of the recipe: `reward` is one number for what you want, and a bill is a cost. The bill is the price of this `hour`, from `PRICE`, times the units of electricity of the `action`, from `ENERGY`.",
            "In the check's case, low cooling at hour 10 uses 1 unit at the day price of 4: a bill of 4, so a reward of −4. The room goes from 24 to 25 degrees, below the limit, so no penalty is added.",
            "`    reward = ___`",
        ],
        why="`reward = -PRICE[hour] * ENERGY[action]` makes the bill a cost: the more electricity, the lower the reward, and the same unit costs four times as much in the day as at night. With the penalty for an hour above 27 degrees, these few lines are all the agent ever learns about what we want.",
    )
    return


@app.cell(hide_code=True)
def text_after_step():
    mo.md(r"""
    The picture runs the thermostat through one day in your simulator. With the course's numbers, that day costs 103: the bill alone, since the thermostat never lets the room pass 27 degrees. Q-learning knows nothing about the data center except what `step` tells it, so a mistake in the reward would be learned as faithfully as the right one.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_act():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A safety rule

        After training, the agent chooses with `int(np.argmax(q))`: the action with the highest of the three Q's of the state, where `q` is a list of three numbers. `count` says how often the simulator showed this state during training. If it never did, the table has learned nothing about the state: its three Q's are still 0, and `np.argmax` picks the first one, action 0, off. In such a state, `act(q, count, temp, hour)` should let the thermostat decide, with `thermostat(temp, hour)`. Replace the `...` with the test for a state the simulator never showed.

        The check calls `act([0.0, 0.0, 0.0], 0, 26, 3)`: a state that never came by, 26 degrees at hour 3. Then five other cases: states that came by 40, 7 and 1 times, and states at 33 and 15 degrees that never came by.
        """),
        L.task("Let the thermostat decide in a state the simulator never showed."),
    ])
    return


@app.cell(hide_code=True)
def exercise_act():
    box_act = L.code_box("act", lines=4)
    box_act
    return (box_act,)


@app.cell(hide_code=True)
def check_act(box_act):
    result_act = L.check("act", box_act.value)
    result_act
    return (result_act,)


@app.cell(hide_code=True)
def hints_act():
    L.hints(
        "act",
        [
            "Step 5 of the recipe: a rule to fall back on. The `if` should be true exactly when `count`, the number of times the simulator showed this state, is 0.",
            "In the check's first case, the state never came by, so the thermostat decides: at 26 degrees it cools high, action 2. In a state that came by 40 times, the table decides.",
            "`    if ___:`",
        ],
        why="`if count == 0:` hands every state the simulator never showed to the thermostat. Real systems do something like this: the AI that cooled Google's data centers in 2018 fell back to the building's own rules when needed, and people could take over at any time.",
    )
    return


@app.cell(hide_code=True)
def text_after_act():
    mo.md(r"""
    The picture puts the course's agent in a warm night: the servers stay busy, 2 degrees per hour instead of 1, and the simulator never showed a night like that. Without the rule, the table, still full of zeros in those states, keeps choosing off: after four hours the room is above 27 degrees, and by the time it reaches states the table knows, it is at 35, where most of the hours too hot can no longer be avoided. That makes 21 hours too hot. With the rule, the thermostat steps in until the room is back in states the table knows.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Training in the simulator, then a day in reality

    The demo trains Q-learning for 10,000 days in a simulator: your `step` if it passes its check, and the course's otherwise, with η 0.1, ε 0.1 and γ 1, and the temperature and the hour as the state. The chart shows the cost of every day of training, with the thermostat's 103 dashed; beside it, what the agent would do at that moment. After training, the agent lives one day in reality: as in the simulator, with a hotter afternoon, four hours one degree hotter, or with a warm night. Choose whether your `act` may step in as a safety rule; if it does not pass its check yet, the course's steps in. Then go back through the training.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    reality_choice = mo.ui.dropdown(options=C.reality_options(L.lang), value=C.DEMO_TEXT[L.lang]["realities"]["sim"], label=C.DEMO_TEXT[L.lang]["reality"])
    safety_choice = mo.ui.dropdown(options=C.safety_options(L.lang), value=C.DEMO_TEXT[L.lang]["safeties"]["off"], label=C.DEMO_TEXT[L.lang]["safety"])
    train_button = mo.ui.run_button(label="Train in the simulator")
    mo.hstack([reality_choice, safety_choice, train_button], justify="start", gap=1.5, wrap=True)
    return reality_choice, safety_choice, train_button


@app.cell(hide_code=True)
def demo_train(box_act, box_step, reality_choice, result_act, result_step, safety_choice, train_button):
    if not train_button.value:
        _out = mo.md("*Choose the day in reality and the safety rule, then press the button to train.*")
    else:
        _mine = result_step.state == "correct"
        _step = run_code(box_step.value, C.S_NAMES).names["step"] if _mine else C.step
        _act_mine = result_act.state == "correct"
        _act = run_code(box_act.value, C.A_NAMES).names["act"] if _act_mine else C.act
        _reality, _safety = reality_choice.value, safety_choice.value
        _w = C.DEMO_TEXT[L.lang]
        _live = L.live(C.chart_labels(L.lang), C.DAYS, role="loss", low=0, high=200, window=100, xticks=C.demo_xticks(),
                       key=C.chart_key(L.lang))
        _live.reference(C.thermostat_line(), "", role="ink-2")
        _shown = [0]

        def _progress(_run):
            _live.watch(lambda: C.now_picture(_run, L.lang))
            _new = [(_d + 1, _run.costs[_d]) for _d in range(_shown[0], _run.days)]
            _shown[0] = _run.days
            _live.extend(_new, _w["learning"].format(d=C.number(_run.days, 0, L.lang), n=C.number(C.DAYS, 0, L.lang),
                                                     hot=C.number(sum(_run.hots), 0, L.lang)))

        _run = C.demo_run(_step, show=_progress, pause=0.02)
        if _run == "error":
            _out = _live.stopped("*Your `step` gave an error, or something other than a temperature and a number. Does it still pass its check?*")
        else:
            _out = mo.vstack([
                L.explore("rewind", "day", lambda _v: C.rewind_frame(_run, int(_v), L.lang), start=C.DAYS,
                          values=list(range(0, C.DAYS + 1, C.SNAP)), role="",
                          task="Go back to day 0, then forward: from which day does the agent stop going too hot in its own day?"),
                mo.Html(C.test_picture(_run, _reality, _safety, L.lang, _act)),
                mo.md(C.verdict(_run, _reality, _safety, _mine, _act_mine, L.lang, _act)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    As in the simulator, the agent costs 63 against the thermostat's 103. On the hotter afternoon it still does well, 79 against 119, since states like those came by in training: with ε-greedy, the room was warmer than usual now and then. On the warm night, the agent without a rule costs 214, with 21 hours too hot, while the thermostat stays at 110 without any; with the safety rule, the agent costs 104, with none. That is the pattern of real systems: the more the world differs from the simulator, the more a learned policy needs limits and a rule to fall back on. And a count of 0 is the least a rule should catch: in other training runs, with another seed, states that came by only a few times still mislead the agent on the warm night.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_where():
    mo.md(r"""
    ### Exercise 3. Where to learn

    A warehouse wants an RL agent to steer its robots between the shelves. A crash costs a robot.
    """)
    return


@app.cell(hide_code=True)
def choice_where_cell():
    choice_where = L.choice(
        "learn_where",
        question="How should the agent learn?",
        options={
            "right": "In a simulator of the warehouse first, then on the floor with a safety rule",
            "floor": "On the real floor from the start: real experience teaches best",
            "logs": "Only from the logs of the robots it replaces: they show every crash it could make",
            "sim": "In the simulator only: what works there works on the floor",
        },
        correct="right",
        why={
            "right": "That is how the tokamak and the data centers did it: the many first mistakes happen where they cost nothing, and on the real machine limits and a rule to fall back on catch what the simulator got wrong, since a simulator is never exactly the world.",
            "floor": "In the demo, the room was too hot in more than 20,000 hours while the agent learned. What would that many crashes cost on a real floor?",
            "logs": "Logs only show what the old robots did, and they hardly ever crashed. Can an agent learn what a crash costs from data with almost no crashes?",
            "sim": "In the demo, the warm night was a state the simulator never showed. What does the agent do on the floor in a situation the simulator lacked?",
        },
    )
    choice_where
    return (choice_where,)


@app.cell(hide_code=True)
def text_exercise_zero_shot():
    mo.md(r"""
    ### Exercise 4. Straight from the simulator

    Jonas Degrave and colleagues (2022) trained their agent in a simulator of the TCV tokamak, and ran it on the real machine without any more training.
    """)
    return


@app.cell(hide_code=True)
def choice_zero_shot_cell():
    choice_zero_shot = L.choice(
        "zero_shot",
        question="What did that need, which the demo's simulator lacked on the warm night?",
        options={
            "days": "More days of training in the same simulator",
            "right": "A simulator close to the real machine in the states it reaches",
            "eta": "A larger learning rate η, so that it learns faster",
            "reward": "A reward without a penalty, so that mistakes count less in what it learns",
        },
        correct="right",
        why={
            "right": "The agent only knows the states it practiced, so the simulator has to show the states the real machine reaches, and behave in them close to how the machine behaves. The demo's simulator never had a warm night, and more training in it would not have helped: those states never come by.",
            "days": "On the warm night the agent landed in states the simulator never showed. Would more days in the same simulator ever show them?",
            "eta": "η sets how big a step each update takes. Does a bigger step help in a state the agent never visited?",
            "reward": "Without the penalty, the agent would not care about the hours too hot. Would the plasma then stay away from the wall?",
        },
    )
    choice_zero_shot
    return (choice_zero_shot,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [CartPole, MountainCar and the benchmarks of RL](/course/open/rl-classic-environments?lang=en), the environments were simulators made for learning, where a mistake cost nothing. Outside games the same questions come back, the state, the actions and the reward, with one more: what does a mistake cost, and what catches it? The ε-greedy exploration of [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en) is exactly what makes learning in the world expensive, which is why real systems learn in simulators, from logged data, or behind limits and rules to fall back on. The next lesson, *From checkers to Go: milestones of RL*, goes back to the games, where RL made its name.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_where, choice_zero_shot, result_act, result_step):
    L.wrap_up(result_step, result_act, choice_where, choice_zero_shot)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
