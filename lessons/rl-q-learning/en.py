# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-q-learning. Route: course/route/lessen/overige-takken.yaml. Checks, the environment, the run and the replay: checks.py.
# termen:allow epsilon-greedy

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-q-learning", "en")
    C = L.checks   # GRID, the environment that rolls its dice, the course's Q-learning and its recorded run


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [When steps can slip: value iteration](/course/open/rl-value-iteration?lang=en) and [Random numbers and seeds](/course/open/np-random?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_table():
    quick_table = L.choice(
        "quick-table",
        question="What does value iteration need to work out $Q(s, a)$?",
        options={
            "rewards": "Only the rewards the agent got so far",
            "right": "For every state and action, its outcomes: probability, next state and reward",
            "nothing": "Nothing: it tries actions until it finds G",
        },
        correct="right",
        why={
            "right": "The table `T`: the Bellman equation adds up every outcome of an action, each with its probability.",
            "rewards": "Value iteration tries nothing: it adds up every outcome from the table `T`. See [When steps can slip: value iteration](/course/open/rl-value-iteration?lang=en).",
            "nothing": "Value iteration never takes a step: it needs the table `T`. See [When steps can slip: value iteration](/course/open/rl-value-iteration?lang=en).",
        },
    )
    quick_table
    return (quick_table,)


@app.cell(hide_code=True)
def quick_check_random():
    quick_random = L.choice(
        "quick-random",
        question="`rng = np.random.default_rng(0)`, and then `rng.random(size=1000) < 0.1`. About how many of the 1000 are `True`?",
        options={
            "right": "About 100",
            "most": "About 900",
            "exact": "Exactly 100",
        },
        correct="right",
        why={
            "right": "`rng.random` gives floats from 0 up to 1, spread evenly, so about one in ten lies below 0.1.",
            "most": "`< 0.1` is `True` only for the numbers below 0.1: about one in ten. See [Random numbers and seeds](/course/open/np-random?lang=en).",
            "exact": "About one in ten lies below 0.1, but the numbers are random: with this seed, 89 are. See [Random numbers and seeds](/course/open/np-random?lang=en).",
        },
    )
    quick_random
    return (quick_random,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Value iteration needed the table `T`: every probability and every reward. A robot in a new building, or a game whose rules nobody wrote down, has no such table. It can only try: take an action, and see where it lands and what it gets.

    **An everyday example.** In a new town you look for the quickest way to cycle to work. With no map of travel times, you try routes, remember how long each took, and mostly take the best so far; now and then you try another, in case it is quicker.
    """)
    return


@app.cell(hide_code=True)
def predict_same():
    guess_same = L.guess(
        "same",
        question="An agent in GRID never sees `T`: after each step it only learns where it landed and its reward. Can it still learn to go along the top row, as value iteration did?",
        options={
            "yes": "Yes: the slips it lands in teach it the risk",
            "straight": "No: it takes the straight way, since it does not know the slip",
            "never": "No: without `T` it cannot learn a policy",
        },
        reveal="**Yes.** In the course's run, its best actions come to lead from S along the top row. A slip into a pit costs it −1 each time, so the risk counts as often as it happens.",
    )
    guess_same
    return (guess_same,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        The agent learns $Q(s, a)$ from its own steps, and keeps it in a **Q-table**: a number for every state and action, all 0 at the start. After a step it knows its state $s$, its action $a$, the reward $r$ and the state $s'$ where it landed. Value iteration added up all outcomes, each with its probability; the agent has only the one that happened. So it moves $Q(s, a)$ a step towards that outcome, the **target** $r + \gamma \max_{a'} Q(s', a')$: the reward plus the discounted largest Q where it landed. That is **Q-learning**:
        """),
        L.formula(r"[[param:Q(s, a)]] \leftarrow [[param:Q(s, a)]] + [[hyper:\eta]] \left( [[data:r]] + [[hyper:\gamma]] \max_{a'} [[param:Q(s', a')]] - [[param:Q(s, a)]] \right)"),
        mo.md(r"""
        $\eta$ is the learning rate, as in gradient descent; books on Q-learning write α. An outcome that comes in 8 of 10 steps pulls 8 times as often as one in 1 of 10, so over many steps $Q(s, a)$ moves towards the expected value: the $Q$ of value iteration.

        Which action does the agent take? Always taking the one with the largest $Q$, **exploitation**, is not enough: it would never find out what the others do. **ε-greedy** takes a random action with probability $\varepsilon$, **exploration**, and otherwise the best one. One run from S until G, a pit, or 100 steps is an **episode**; then the agent starts at S again, with the Q-table it has.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **Q-learning** | | learning $Q(s, a)$ from steps, moving it towards the target | `update` |
        | **Q-table** | | a number $Q(s, a)$ for every state and action | `Q[s][a]` |
        | **exploitation** | | taking the action that looks best now | |
        | **exploration** | | trying another action, to find out what it does | |
        | **ε-greedy** | epsilon-greedy | with probability $\varepsilon$ a random action, else the one with the largest $Q$ | `choose` |
        | **episode** | | one run from the start until the task ends, or until a limit of steps | |
        | $\varepsilon$ | epsilon | the probability of a random action | `epsilon` |
        | $\leftarrow$ | becomes | the left side gets the value of the right side | `=` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_replay():
    mo.md(rf"""
    ### The run, episode by episode

    The course ran Q-learning on GRID for {C.EPISODES} episodes, with $\varepsilon$ = {C.number(C.EPSILON, 1, 'en')}; after you train your own agent in *Your turn*, this shows yours. A line is the way of that episode, a dot a step into the edge; a number is the largest <span class="c-param">Q</span> of a state after that episode, and a triangle its best action. The *greedy way* is where the agent would go from S if it always took its best action and nothing slipped.
    """)
    return


@app.cell(hide_code=True)
def replay_cell(trained):
    _run = trained if trained is not None else C.RUN
    _labels = {"title": "Return of every episode", "x": "episode", "y": "return",
               "alt": "The return of every episode up to the chosen one, thin, and its mean over 20 episodes, thick, with a dot at the chosen one.",
               "grid": "After episode {k}",
               "grid_alt": "GRID after episode {k}: the way of that episode as a line, the largest Q of every state, and a triangle for its best action.",
               "start": "Episode 0: every Q is 0, and no action is better than another yet.",
               "episode": "Episode {k}: {end} after {n} steps, with a return of {g}. The greedy way from S {way}.",
               "goal": "ended in G", "pit": "ended in a pit", "limit": "stopped",
               "way_top": "goes up and along the top row, as value iteration's does",
               "way_goal": "reaches G another way", "way_not": "does not reach G yet"}
    _out = L.explore("replay", "episode", lambda _v: C.replay_frame(_run, int(_v), _labels, L.lang), start=C.EPISODES,
                     values=C.stops(), role="",
                     task="Go back to episode 0, then forward. From which episode on does the greedy way from S go along the top row?")
    _out
    return


@app.cell(hide_code=True)
def text_replay_after():
    _last = sum(C.RUN["returns"][-100:]) / 100
    mo.md(rf"""
    In the course's run, from episode {C.top_from(C.RUN)} on. Once the agent has found G, the reward travels back a step at a time. The mean return of the last 100 episodes, {C.number(_last, 2, "en")}, stays below the {C.number(C.Q_STAR[C.START]["up"], 2, "en")} of value iteration: one step in ten is random, and can go into a pit. After {C.EPISODES} episodes it agrees with value iteration in {C.agree(C.RUN["final"])} of the {len(C.STATES)} states. In `(1, 4)`, which it seldom visits, it goes up instead of right; value iteration rates the two almost the same, {C.number(C.Q_STAR[(1, 4)]["up"], 2, "en")} and {C.number(C.Q_STAR[(1, 4)]["right"], 2, "en")}.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Q-learning:

    1. Make a Q-table: 0 for every state and action.
    2. Start an episode at S.
    3. Choose an action, ε-greedy: with probability $\varepsilon$ a random one, else the one with the largest $Q$.
    4. Take it: the environment gives back where you landed and the reward.
    5. Work out the target, the reward plus $\gamma$ times the largest $Q$ where you landed, and move $Q$ of your state and action a step $\eta$ towards it.
    6. Go on from where you landed, at step 3, until the episode ends: at G, in a pit, or after 100 steps. Then start the next episode, at step 2.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    SMALL from the lesson before, with $\eta$ = 0.5 and $\gamma$ = 0.9. Three steps an agent took there go through step 5; the `print` shows each step, its target and the new $Q$. Then step 3, eight times in `(0, 0)`, with $\varepsilon$ = 0.25. Run the cell.
    """)
    return


@app.cell
def example_q_learning():
    import numpy as np

    # 1. the Q-table of SMALL: a Q for every action in every state, all 0 at the start
    _Q = {}
    for _s in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]:
        _Q[_s] = {"up": 0, "down": 0, "left": 0, "right": 0}

    # three steps an agent took: (state, action, reward, where it landed); in the third, up slipped right
    _steps = [((0, 1), "right", 1, (0, 2)), ((0, 0), "right", 0, (0, 1)), ((1, 1), "up", -1, (1, 2))]
    for _s, _a, _r, _nxt in _steps:
        # 5. the target: the reward plus gamma times the largest Q where it landed
        _qs = []
        for _b in _Q[_nxt]:
            _qs.append(_Q[_nxt][_b])
        _target = _r + 0.9 * max(_qs)
        # 5. Q moves a step eta = 0.5 towards the target
        _Q[_s][_a] = _Q[_s][_a] + 0.5 * (_target - _Q[_s][_a])
        print(_s, _a, _r, _nxt, round(_target, 3), round(_Q[_s][_a], 3))

    # 3. epsilon-greedy in (0, 0), eight times: with epsilon = 0.25 a random action, else the one with the largest Q
    _rng = np.random.default_rng(4)
    _actions = ["up", "down", "left", "right"]
    _qs = []
    for _b in _actions:
        _qs.append(_Q[(0, 0)][_b])
    for _i in range(8):
        if _rng.random() < 0.25:
            print("random:", _actions[_rng.integers(0, 4)])
        else:
            print("best:", _actions[np.argmax(_qs)])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The first step lands on G: the target is the reward, 1, since the $Q$ of G stays 0, and $Q$ moves halfway, to 0.5. The second lands on `(0, 1)`, whose largest $Q$ is now 0.5: the target is 0.9 · 0.5 = 0.45, and $Q$ becomes 0.225: the reward of G travels back. In the third, up slipped into the pit: the agent does not know it slipped; its target is the reward, −1, and $Q$ becomes −0.5. Two of the eight choices are random, down and left; the other six take right, the largest $Q$ in `(0, 0)`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_update():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. One step of Q-learning

        `update(Q, state, action, reward, nxt, eta, gamma)` gives back the new $Q$ of `state` and `action` after a step that landed on `nxt`: step 5 of the worked example. Replace the two `...`: the target and the new $Q$. The check calls `update(Q, (1, 0), "up", 0, (0, 0), 0.2, 0.9)`, with `Q` the Q-table at the end of the course's run, and five other steps.
        """),
        L.task("Make `update` work out the target r + γ max Q(s′, a′), and give back Q(s, a) moved a step η towards it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_update():
    box_update = L.code_box("update", lines=7)
    box_update
    return (box_update,)


@app.cell(hide_code=True)
def check_update(box_update):
    result_update = L.check("update", box_update.value)
    result_update
    return (result_update,)


@app.cell(hide_code=True)
def hints_update():
    L.hints(
        "update",
        [
            "Step 5 of the recipe, and the formula of *The idea*: first the target, then the step towards it.",
            "The loop puts every $Q$ of `nxt` in `qs`, so `max(qs)` is the largest. The step is `eta` times the difference between the target and the old $Q$.",
            "`target = reward + gamma * ___(qs)` and `return Q[state][action] + eta * (___ - Q[state][action])`.",
        ],
        why="The target is what this one step says $Q(s, a)$ should be. Moving only a share `eta` of the way there means that one slip does not undo many steps.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_choose():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. ε-greedy

        `choose(Q, state, epsilon, rng)` gives back the action the agent takes in `state`: step 3 of the worked example. Replace the two `...`: when to take a random action, and which action otherwise. The check calls `choose(Q, (1, 0), 0.1, rng)` 4000 times and counts each action: up, the best, should come about 92.5 times in 100, 90 plus a quarter of the 10 random ones.
        """),
        L.task("Make `choose` take a random action with probability ε, and else the action with the largest Q."),
    ])
    return


@app.cell(hide_code=True)
def exercise_choose():
    box_choose = L.code_box("choose", lines=12)
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
            "Step 3 of the recipe, and the worked example: a test with `rng.random()`, then the best action.",
            "`rng.random()` gives a float from 0 up to 1, below `epsilon` in a share `epsilon` of the calls. `np.argmax(qs)` is the place of the largest $Q$; `actions` holds the action at each place.",
            "`if rng.random() < ___:` and `return actions[np.___(qs)]`.",
        ],
        why="In a share `epsilon` of the calls the agent explores, with any of the four actions, the best one too; in the others it exploits.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(rf"""
    ### Train an agent

    The button trains an agent on GRID for {C.EPISODES} episodes with your `update` and `choose`, with $\eta$ = {C.number(C.ETA, 1, 'en')}, $\varepsilon$ = {C.number(C.EPSILON, 1, 'en')} and $\gamma$ = {C.number(C.GAMMA, 1, 'en')}, as in the recipe:

    ```python
    for episode in range({C.EPISODES}):
        state = (1, 0)
        for t in range({C.MAX_STEPS}):
            action = choose(Q, state, epsilon, rng)
            nxt, reward = step(state, action, rng)   # the environment: only it knows T
            Q[state][action] = update(Q, state, action, reward, nxt, eta, gamma)
            state = nxt
            if state in ends:   # G or a pit: the episode is over
                break
    ```

    The chart shows the return of every episode, thin, and its mean over the last 20, thick; dashed, the same, averaged over 20 runs of the course. The demo waits 10 ms after each episode, so that you can watch.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    train_button = mo.ui.run_button(label="Train an agent with my update and choose")
    train_button
    return (train_button,)


@app.cell(hide_code=True)
def demo_train(box_choose, box_update, result_choose, result_update, train_button):
    trained = None
    if not train_button.value:
        _out = mo.md("*Press the button to train an agent.*")
    else:
        _mine = [_n for _n, _r in (("update", result_update), ("choose", result_choose)) if _r.state == "correct"]
        _update = run_code(box_update.value).names["update"] if "update" in _mine else C.update
        _choose = run_code(box_choose.value).names["choose"] if "choose" in _mine else C.choose
        _live = L.live({"title": "Your run: the return of every episode", "x": "episode", "y": "return",
                        "alt": "The return of every episode of your run so far, thin, and its mean over the last 20, thick; dashed, the same, averaged over 20 runs of the course."},
                       C.EPISODES, role="ink", low=-1.05, high=1.05, window=20)
        _live.line("**Training** …")
        _live.reference(list(enumerate(C.mean_line(), 1)), "mean of 20 runs", role="ink")
        _r = C.train(_update, _choose, seed=C.YOUR_SEED, pause=0.01,
                     show=lambda _k, _g: _live.add(_k, _g, f"**Episode {_k} of {C.EPISODES}**"))
        if _r == "update":
            _out = _live.stopped("*Your `update` gave an error or no number during the run. Does it still pass its check?*")
        elif _r == "choose":
            _out = _live.stopped("*Your `choose` gave an error or no action during the run. Does it still pass its check?*")
        else:
            trained = _r
            _theirs = [_n for _n in ("update", "choose") if _n not in _mine]
            if len(_mine) == 2:
                _who = "your `update` and your `choose`"
            elif len(_mine) == 1:
                _who = f"your `{_mine[0]}` and the course's `{_theirs[0]}`"
            else:
                _who = "the course's `update` and `choose`, since yours do not pass their checks yet"
            _off = C.differ(_r["final"])
            _where = "" if not _off else " Where not: " + ", ".join(f'in `{_s}` it takes `"{_a}"`' for _s, _a in _off) + "."
            _out = mo.vstack([
                mo.Html(_live.picture()),
                mo.md(f"Trained with {_who}. The mean return of the last 100 episodes is {C.number(sum(_r['returns'][-100:]) / 100, 2, 'en')}. In {C.agree(_r['final'])} of the {len(C.STATES)} states where the task goes on, your agent takes an action that value iteration takes too.{_where} *The run, episode by episode*, in *The idea*, now shows your run."),
                C.policies(_r["final"], ("Your agent: the largest Q", "Value iteration: the largest Q"),
                           f"GRID twice: first the largest Q your agent learned in every state, with a triangle for its best action, then the same for value iteration. They agree in {C.agree(_r['final'])} of {len(C.STATES)} states.", decimal="."),
            ])
    _out
    return (trained,)


@app.cell(hide_code=True)
def text_exercise_zero():
    mo.md(r"""
    ### Exercise 3. Never explore
    """)
    return


@app.cell(hide_code=True)
def choice_zero_cell():
    choice_zero = L.choice(
        "no-exploring",
        question="With $\\varepsilon$ = 0, the agent always takes the action with the largest $Q$. What happens in GRID, with every $Q$ at 0 at the start?",
        options={
            "faster": "It learns faster: it wastes no steps on random actions",
            "stuck": "It does not get to G: with every Q at 0 it always goes up",
            "later": "It learns the same policy, only later",
        },
        correct="stuck",
        why={
            "stuck": "In 20 runs of 300 episodes it did not reach G once: with equal $Q$, `np.argmax` takes the first action, up, and a slip only moves it sideways. Without exploration it never finds out what the other actions do.",
            "faster": "Saving the random steps does not help: with every $Q$ at 0 it always goes up, and from the top row up never leads to G. See exploration in *The idea*.",
            "later": "It learns nothing at all: it does not reach G, so no reward ever changes a $Q$. See exploration in *The idea*.",
        },
    )
    choice_zero
    return (choice_zero,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [When steps can slip: value iteration](/course/open/rl-value-iteration?lang=en), the Bellman equation added up every outcome from the table `T`. Q-learning uses the same equation without `T`: each step is one outcome, and small steps $\eta$ towards them average the outcomes. Reinforcement learning comes back in a branch still to come, whose project is your own agent. In the next lesson, *A language model predicts the next token*, a model learns from text alone which token comes next.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_zero, result_choose, result_update):
    L.wrap_up(result_update, result_choose, choice_zero)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
