# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-project-grid-agent. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the course's world and agent, the runs, the demo and the pictures: checks.py; the loop the course puts in your project: project/agent/episode.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("rl-project-grid-agent", "en")
    C = L.checks   # the course's world, agent and runs, the check of your world, the demo and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A map of RL methods](/course/open/rl-kinds-of-rl?lang=en) and [Project: your own bandit agent](/course/open/rl-project-bandit?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_kinds():
    quick_kinds = L.choice(
        "quick-kinds",
        question="Monte Carlo control and SARSA both learn the $Q$ of the policy they follow, without a model. Which of the four questions of the map sets them apart?",
        options={
            "model": "Monte Carlo control needs a model of the environment, with the chance of every next state, and SARSA does not",
            "right": "Monte Carlo learns from whole episodes, SARSA from its own estimates",
            "off": "SARSA is off-policy",
        },
        correct="right",
        why={
            "right": "Monte Carlo moves $Q(S, A)$ toward the return that really followed, so it waits for the end of the episode. SARSA moves it after every step, toward $R + \\gamma Q(S', A')$, with $Q(S', A')$ its own estimate. Both are model-free and on-policy.",
            "model": "Both learn only from steps that happened, without the chance of each next state: what does each of them move $Q(S, A)$ toward? See [A map of RL methods](/course/open/rl-kinds-of-rl?lang=en).",
            "off": "SARSA learns from the action $A'$ it really takes next, with the policy it follows: is that on-policy or off-policy? See [A map of RL methods](/course/open/rl-kinds-of-rl?lang=en).",
        },
    )
    quick_kinds
    return (quick_kinds,)


@app.cell(hide_code=True)
def quick_check_fair():
    quick_fair = L.choice(
        "quick-fair",
        question="In your bandit project, every agent played the same 200 bandits, made with the same seeds. Why?",
        options={
            "speed": "So that the run takes less time",
            "right": "So that a difference between the agents comes from the agents, not from easier bandits",
            "luck": "So that every agent gets the same rewards whatever machine it pulls, and every run ends with the same total",
        },
        correct="right",
        why={
            "right": "With the same bandits, the same machine pays the same at the same pull for every agent. A difference that is left comes from what the agents choose.",
            "speed": "Making 200 bandits takes as long with any seeds: what would change if each agent got bandits of its own? See [Project: your own bandit agent](/course/open/rl-project-bandit?lang=en).",
            "luck": "The agents pull different machines, so they get different rewards and different totals: what do the same bandits keep the same? See [Project: your own bandit agent](/course/open/rl-project-bandit?lang=en).",
        },
    )
    quick_fair
    return (quick_fair,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        In [Careful or bold: SARSA and Q-learning](/course/open/rl-sarsa?lang=en) and [Playing better from experience: Monte Carlo control](/course/open/rl-monte-carlo-control?lang=en), the course made the worlds and its code learned them. In this project you make both parts yourself: a grid world in your project, with walls, puddles and pits where you put them, and one agent that can learn it in three ways, with the targets of Monte Carlo control, SARSA and Q-learning. Then you find out which of the three does best on your world, with the fair comparison of your bandit project: the same world, many runs, and a learning curve for each method.

        Which method is best depends on the world. A method that wins on one world can lose on the next, and one run can make the worse method look better. Only a world you can change, and a comparison you can repeat on it, shows you why.

        **An everyday example.** A baker wants to know which of three recipes makes the best bread. One loaf of each, baked on different days, says little: one day the oven runs hot, another day the flour is damp. Baked ten times each, in the same oven with the same flour, the recipes show what they are worth. And a recipe that wins in one oven can lose in another, so the baker tries a second oven too.
        """),
        draw.project_steps(["Your bandit agent", "Your grid agent"], 1,
                           here_label="← today", title="Where your project stands: agent"),
    ])
    return


@app.cell(hide_code=True)
def predict_method():
    guess_method = L.guess(
        "method",
        question="In this lesson your agent learns the course's world, in the picture, with each method: 10 runs of 300 episodes, all with ε = 0.1, $\\eta$ = 0.2 and $\\gamma$ = 1. Every step costs −1, a step onto a puddle −10, and a step into a pit −100, after which the agent starts again on S. The shortest way, 7 steps, runs between the puddles and the pits; a way of 11 steps goes over the top. Which method has the highest mean return over the last 50 episodes, do you think?",
        options={
            "mc": "Monte Carlo",
            "q": "Q-learning",
            "right": "SARSA",
            "same": "All three about the same",
        },
        reveal="**SARSA.** Over the last 50 episodes SARSA has a mean return of −16.7, Q-learning −30.4 and Monte Carlo −42.0. Q-learning learns the shortest way, along the pits, and pays for it each time a random step drops it in. SARSA learns the way over the top. The demo lets your own agent learn the same world.",
    )
    mo.vstack([mo.Html(C.world_picture(L.lang)), guess_method])
    return (guess_method,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A world of your own.** Your world is a list of strings, one string per row and one sign per cell. In every cell the agent can go up, down, left or right. What a step does depends on the sign of the cell it leads to:

        | Sign | Cell | A step onto it |
        |---|---|---|
        | `.` or `S` | free, or the start | −1 |
        | `G` | the goal | −1, and the episode is over |
        | `W` | a puddle | −10 |
        | `X` | a pit | −100, and the agent goes back to S; the episode goes on |
        | `#` | a wall | the agent stays in its cell, and the step costs as much as a step onto its own cell |

        A step into the edge of the world is like a step into a wall. Every step costs, so the agent that reaches G in the fewest steps, without puddles and pits, gets the highest return, as on the cliff of [Careful or bold: SARSA and Q-learning](/course/open/rl-sarsa?lang=en). The agent only sees the cell it is in, as (row, column), and the reward: it does not know the signs or the rules. All three methods are model-free.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        In the picture, a walk of 13 steps through the course's world. From S the agent goes left, into the edge, and stays; then up, right onto a puddle, down, right, and down into a pit, which puts it back on S. Seven steps to the right then take it between the puddles and the pits to G. Its rewards are −1, −1, −10, −1, −1, −100 and seven times −1: a return of −121.

        **One loop.** As in your bandit project, an agent and an environment take turns in a loop. The course puts this loop in your project, as `work/projects/agent/episode.py`:

        ```python
        def episode(agent, world, limit):
            state = world.reset()
            action = agent.act(state)
            total = 0
            for t in range(limit):
                nxt, reward, done = world.step(action)
                nxt_action = agent.act(nxt)
                agent.learn(state, action, reward, nxt, nxt_action)
                total = total + reward
                if done:
                    break
                state = nxt
                action = nxt_action
            agent.end()
            return total
        ```

        The agent chooses its next action before it learns, since SARSA learns from it. After the last step, `agent.end()` lets Monte Carlo learn from the whole episode. An episode ends on G, or stops after `limit` steps, 200 in this lesson: an agent that has learned little could otherwise walk in circles forever. `episode` gives back the return, the total of the rewards, since $\gamma$ = 1.

        **One agent, three targets.** Your `GridAgent` keeps $Q$ in a dict: for every state it has seen, a dict with the $Q$ of each action. It chooses ε-greedy, and moves $Q(S, A)$ a part $\eta$ of the way toward a target, as in [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en):
        """),
        L.formula(r"[[param:Q(S, A)]] \leftarrow [[param:Q(S, A)]] + [[hyper:\eta]] \left( \text{target} - [[param:Q(S, A)]] \right)"),
        mo.md(r"""
        Only the target differs between the methods:

        - SARSA: $R + \gamma Q(S', A')$, with $A'$ the action the agent takes next.
        - Q-learning: $R + \gamma \max_{a'} Q(S', a')$, the largest $Q$ in the next state.
        - Monte Carlo: the return after $S$ and $A$, worked out after the episode, from the last step back to the first: the reward of a step plus $\gamma$ times the return after that step.

        In [Playing better from experience: Monte Carlo control](/course/open/rl-monte-carlo-control?lang=en), $Q$ was the average of the returns, with $\eta = 1/n$. Here Monte Carlo moves with the same constant $\eta$ as the other two methods, like the estimates of the drifting machines in your bandit project: the policy changes while the agent learns, so old returns say less about it. And it learns from every visit of a state and an action in an episode, not only from the first. The agent needs no rule for G: no step starts from G, so every $Q$ of G stays 0, and the target of the last step is its reward.

        **A fair comparison.** Every method gets the same world and the same settings: ε = 0.1, $\eta$ = 0.2 and $\gamma$ = 1. The world has no chance in it: the same action in the same cell always gives the same. So two runs differ only in the random choices of their agents. A run is one new agent on one new world, for 300 episodes; run $k$ gives its agent `np.random.default_rng(k)`, with $k$ from 0 to 9, for every method. Your `learning_curve` averages the returns of the 10 runs at every episode, with your `average` from your bandit project.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **wall** | | a cell the agent cannot enter: a step into it leaves the agent where it is | `"#"` |
        | **puddle** | | a cell that costs −10 to step onto | `"W"` |
        | **pit** | | a cell that costs −100 to step into; the step puts the agent back on S | `"X"` |
        | **limit** | | the most steps an episode gets: after that it stops, also when the agent has not reached G | `limit` |
        | **learning curve** | | the mean return over the runs at every episode | `learning_curve` |
        """),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    An agent of your own on a world of your own:

    1. Draw the world as a list of strings, and make sure that a way leads from S to G.
    2. Write the world as a class: `reset` puts the agent on S, and `step(action)` moves it by the rules and gives back where it is, the reward, and whether the episode is over.
    3. Write the agent as a class: a dict of $Q$ values, an ε-greedy `act`, and a `learn` that moves $Q(S, A)$ toward the target of its method. For Monte Carlo, `learn` keeps the steps, and `end` learns from them after the episode.
    4. Let the agent and the world take turns with `episode`, up to the limit.
    5. For every method and every seed, make a new world and a new agent, play all the episodes, and average the returns over the seeds at every episode.
    6. Compare the learning curves, the mean return over the last episodes, and where the best actions lead from S. Then change the world, and compare again.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    An agent with $\eta$ = 0.5 and $\gamma$ = 0.9 has these $Q$ values in the cell (0, 1): up −4, down −6, left −8 and right −2. Every other $Q$ is still 0. It takes two steps: from (0, 0) right to (0, 1), with reward −1, and there it chooses down; then down onto a puddle in (1, 1), with reward −10, and there it chooses left. Suppose the episode stops there.

    - SARSA, step 1: the target is $-1 + 0.9 \cdot Q$((0, 1), down) $= -1 + 0.9 \cdot (-6) = -6.4$, so $Q$((0, 0), right) $= 0 + 0.5 \cdot (-6.4 - 0) = -3.2$.
    - Q-learning, step 1: the largest $Q$ in (0, 1) is −2, of right, so the target is $-1 + 0.9 \cdot (-2) = -2.8$, and $Q$((0, 0), right) $= 0 + 0.5 \cdot (-2.8 - 0) = -1.4$.
    - Step 2: every $Q$ of (1, 1) is still 0, so both targets are −10, and $Q$((0, 1), down) $= -6 + 0.5 \cdot (-10 - (-6)) = -8$.
    - Monte Carlo changes nothing during the two steps. After the episode it goes back: the return after step 2 is $-10 + 0.9 \cdot 0 = -10$, so $Q$((0, 1), down) becomes −8 as well. The return after step 1 is $-1 + 0.9 \cdot (-10) = -10$, so $Q$((0, 0), right) $= 0 + 0.5 \cdot (-10 - 0) = -5$.

    For the first step, the three targets are −6.4, −2.8 and −10. Q-learning expects the most: it counts on the best action in (0, 1), right. Monte Carlo expects the least: it saw what really came after, the puddle.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_world():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Your GridWorld

        `GridWorld(rows)` is the environment. `__init__` is given: it keeps `rows`, the `height` and the `width` of the world, the cell of S as `self.start`, and puts the agent there, in `self.state`. `reset` puts the agent back on S. In `step(action)` the first lines are given: `MOVES` says how the row and the column change with each action, and `r` and `c` become the row and the column of the cell the action leads to, or of the cell the agent is in when the action leads into the edge or a wall. `sign` is the sign of that cell.

        Replace the `...` with the rules. Into a pit, the agent goes back to S, with −100, and the episode goes on. Otherwise the agent moves to the cell of `r` and `c`, and then gets −1 and the end of the episode on G, −10 on a puddle, and −1 on every other cell. Each rule gives back three values: where the agent is now, the reward, and whether the episode is over.

        The check makes `GridWorld(["S#G", ".W.", "X.."])` and takes eight steps: right, up, down, down, down, right, right and up. The results should be `[((0, 0), -1, False), ((0, 0), -1, False), ((1, 0), -1, False), ((0, 0), -100, False), ((1, 0), -1, False), ((1, 1), -10, False), ((1, 2), -1, False), ((0, 2), -1, True)]`: a wall, the edge, a pit, a puddle and the goal. Then it tries two cases of its own: `reset`, and a world where S is not in a corner. When it is right, the check saves your class in your project, as `work/projects/agent/gridworld.py`.
        """),
        L.task("Make `step` send the agent back to S from a pit, move it otherwise, and give back the reward of the sign and whether the episode is over."),
    ])
    return


@app.cell(hide_code=True)
def exercise_world():
    box_world = L.code_box("world", lines=32)
    box_world
    return (box_world,)


@app.cell(hide_code=True)
def check_world(box_world):
    result_world = L.check("world", box_world.value)
    result_world
    return (result_world,)


@app.cell(hide_code=True)
def hints_world():
    L.hints(
        "world",
        [
            "First the pit: when `sign` is `\"X\"`, `self.state` becomes `self.start`, and `step` gives back `self.state`, `-100` and `False`, before the agent moves. Then `self.state` becomes the cell of `r` and `c`. Then one rule for `\"G\"`, one for `\"W\"`, and a last `return` for every other cell.",
            "In the example, the fourth step goes down from (1, 0) into the pit in (2, 0): `step` gives back `((0, 0), -100, False)`, and the agent stands on S again, so the next step down goes to (1, 0). A step into the wall in (0, 1) leaves `r` and `c` on the cell the agent is in, (0, 0), and the rules then go on as for a free cell. Under the rule for the pit, one line moves the agent to its new cell.",
            "`        self.state = ___`",
        ],
        why="`if sign == \"X\":` with `self.state = self.start` and `return self.state, -100, False` under it; then `self.state = (r, c)`; then `if sign == \"G\":` with `return self.state, -1, True`, `if sign == \"W\":` with `return self.state, -10, False`, and last `return self.state, -1, False`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_agent(result_world):
    result_world  # made again after the check of exercise 1, which saves your world
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Your GridAgent

        `GridAgent(method, epsilon, eta, gamma, rng)` is one agent for all three methods: `method` is `"mc"`, `"sarsa"` or `"q"`, and `ACTIONS` lists the four actions. Given are `__init__`, which keeps the settings, a dict `q` and a list `steps`; `qs(state)`, which gives the dict with the $Q$ of each action in `state`, and first makes one with four zeros when the agent has not seen `state` yet; `act(state)`, ε-greedy; and `move(state, action, target)`, which moves the $Q$ of `state` and `action` a part `eta` of the way toward `target`. `learn(state, action, reward, nxt, nxt_action)` gets one step: for Monte Carlo it only keeps the step in `steps`. `end()` comes after the episode: `reversed(self.steps)` goes through the steps from the last to the first, `g` keeps the return, and each $Q$ moves toward it.

        Replace the three `...` with the targets: in `learn` the target of SARSA and the target of Q-learning, in `end` the new `g`, the return from that step on. A dict's `.values()` gives its values, here the four $Q$ values of a state, in the order of the actions; `act` uses it already.

        The check makes an agent with $\eta$ = 0.5 and $\gamma$ = 0.9 for each method, gives the cell (0, 1) the $Q$ values of the worked example, lets the agent learn the two steps of the worked example, and calls `end()`. $Q$((0, 0), right) and $Q$((0, 1), down) should be `[(-3.2, -8.0), (-1.4, -8.0), (-5.0, -8.0)]`, for SARSA, Q-learning and Monte Carlo. Then it tries three cases of its own: Monte Carlo with a cell that comes back in the episode, Q-learning with $\eta$ = 1, and SARSA after a pit. When it is right, the check saves your class as `work/projects/agent/gridagent.py`.
        """),
        L.task("Write the three targets: SARSA's and Q-learning's in `learn`, and the return `g` of Monte Carlo in `end`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_agent():
    box_agent = L.code_box("agent", lines=46)
    box_agent
    return (box_agent,)


@app.cell(hide_code=True)
def check_agent(box_agent):
    result_agent = L.check("agent", box_agent.value)
    result_agent
    return (result_agent,)


@app.cell(hide_code=True)
def hints_agent():
    L.hints(
        "agent",
        [
            "Each target is `reward` plus `self.gamma` times what comes after the step. SARSA: the $Q$ of `nxt_action` in `nxt`, from `self.qs`. Q-learning: `max` of the four $Q$ values in `nxt`. Monte Carlo: `g`, the return after this step. `self.move(state, action, ...)` then moves the $Q$ of `state` and `action` toward the target.",
            "In the worked example, the target of Q-learning in the first step is −1 + 0.9 · (−2): −2 is the largest of the four $Q$ values in (0, 1). The target of SARSA is −1 + 0.9 · (−6): −6 is the $Q$ of down, the action the agent takes next. The `g` of Monte Carlo starts at 0 and is built up from the last step back: first −10 + 0.9 · 0, then −1 + 0.9 · (−10).",
            "`            self.move(state, action, reward + self.gamma * max(___))`",
        ],
        why="SARSA: `self.move(state, action, reward + self.gamma * self.qs(nxt)[nxt_action])`. Q-learning: `self.move(state, action, reward + self.gamma * max(self.qs(nxt).values()))`. Monte Carlo, in `end`: `g = reward + self.gamma * g`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_curves(result_agent, result_world):
    result_world, result_agent  # made again after the checks of exercises 1 and 2, which save your files
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Learning curves

        `learning_curve(method, rows, seeds, episodes)` measures one method on one world. The start is given: for every seed, a new `GridWorld(rows)` and a new `GridAgent` with ε = 0.1, $\eta$ = 0.2, $\gamma$ = 1 and `np.random.default_rng(seed)`. Replace the `...` with the run of that seed: a new list `returns`; for every episode, one call of the course's `episode`, with at most `LIMIT` steps, whose return goes into `returns`; and then that list into `runs`. The last line is given: it averages the runs at every episode with your `average` from your bandit project.

        `small_world`, the course's `SMALL`, is `["S..", "WX.", "..G"]`. `learning_curve("q", small_world, 3, 4)` should give `[-139.667, -272.667, -48.0, -62.0]`, to three decimals. The first two episodes are long, with pits on the way; from the third on the agents reach G sooner. Then the check tries four cases of its own: other methods, other worlds, and other numbers of seeds and episodes. When it is right, the check saves your function as `work/projects/agent/curves.py`.
        """),
        L.course_versions(),
        L.task("Make `learning_curve` play all the episodes of each seed with one agent, and put the list of returns of each seed into `runs`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_curves():
    box_curves = L.code_box("curves", lines=22)
    box_curves
    return (box_curves,)


@app.cell(hide_code=True)
def check_curves(box_curves, result_agent, result_world):
    result_world, result_agent  # check again when exercises 1 and 2 save your files
    result_curves = L.check("curves", box_curves.value)
    result_curves
    return (result_curves,)


@app.cell(hide_code=True)
def hints_curves():
    L.hints(
        "curves",
        [
            "For one seed: `returns = []`, a `for` loop over `range(episodes)`, and in it the return of one episode of the course's `episode`, added with `returns.append`. After that loop, `runs.append(returns)`.",
            "With 3 seeds and 4 episodes, `runs` ends as three lists of four returns, one list per seed, and `average` gives four numbers. A new list for each seed keeps the runs apart. One agent per seed, made before its episodes, lets the agent learn from one episode to the next.",
            "`            returns.append(___)`",
        ],
        why="`returns = []`, then `for e in range(episodes):` with `returns.append(episode(agent, world, LIMIT))` under it, and after that loop `runs.append(returns)`.",
    )
    return


@app.cell(hide_code=True)
def text_your_world():
    mo.md(r"""
    ### Your own world

    Now draw a world of your own in the cell below, and run the cell. Use the signs of the table in the idea: `.`, `S`, `G`, `#`, `W` and `X`. A world needs 1 to 8 rows of 2 to 12 signs, all rows of the same length, one S, at least one G, and a way from S to G without walls or pits. Some ideas: a maze of walls, a short way through a puddle beside a longer dry one, a pit right next to the shortest way, two goals. The picture under the cell shows your world as the demo will see it. The cell starts with the course's world.
    """)
    return


@app.cell
def your_world():
    # your world: one string per row; . free, S start, G goal, # wall, W puddle, X pit
    my_world = [
        "........",
        ".WWWWWW.",
        "S......G",
        ".XXXXXX.",
    ]
    return (my_world,)


@app.cell(hide_code=True)
def your_world_view(my_world):
    mo.Html(C.world_view(my_world, L.lang))
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Let your agent learn your world

    The button lets your `GridAgent` learn `my_world` with each of the three methods, in your `GridWorld`, with the course's `episode`: 10 runs of 300 episodes per method, as in the comparison of the idea. Your `average` makes the curves. Where a part of yours is not in your project yet, or does not pass its check, the course's stands in. The chart shows the mean return of the last 10 episodes over the 10 runs: SARSA solid, Q-learning dashed, and Monte Carlo lighter and dashed; a return below −150 stays at the bottom edge. Beside it you see the run with seed 0 of each method as it is now: its best actions as small arrows, and in <span class="c-param">orange</span> the way they lead from S, once they lead to G. When the run is done, you can go back through the episodes.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    run_button = mo.ui.run_button(label="Let my agent learn my world")
    run_button
    return (run_button,)


@app.cell(hide_code=True)
def demo_run(my_world, result_agent, result_world, run_button):
    result_world, result_agent   # learn again with your files once a check saves them
    learned = None
    if not run_button.value:
        _out = mo.md("*Press the button to let your agent learn your world.*")
    elif C.problems(my_world):
        _out = mo.md("*The demo cannot use your world yet: the picture under your world says why.*")
    else:
        _w = C.DEMO_TEXT[L.lang]
        _now = {"shown": 0}

        def _progress(_n, _r):
            if "r" not in _now:   # from the first episode on, the best actions of seed 0 beside the chart
                _now["chart"].watch(lambda: C.now_picture(_now["r"], _now["n"], L.lang))
            _now["r"], _now["n"] = _r, _n
            _now["chart"].reference(C.chart_points(_r, "q", 1, _n), "", role="ink", replace=True)
            _now["chart"].reference(C.chart_points(_r, "mc", 1, _n), "", role="ink-3", replace=True)
            _now["chart"].extend(C.chart_points(_r, "sarsa", _now["shown"] + 1, _n),
                                 _w["progress"].format(n=_n, total=C.EPISODES))
            _now["shown"] = _n

        _live = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"], "alt": _w["alt"]}, C.EPISODES, role="ink",
                       low=C.LOW, high=0.0, window=1, key=C.curve_key(L.lang))
        _live.line("**Learning** …")
        _now["chart"] = _live
        _r = C.demo(my_world, show=_progress)
        if _r == "agent":
            _out = _live.stopped("*Your `GridWorld` or your `GridAgent` stopped with an error during the run, or gave a return that is not a number. Do both still pass their checks?*")
        elif _r == "average":
            _out = _live.stopped("*Your `average` stopped with an error during the run, or gave something other than a number for every episode. Does it still pass its check in [Project: your own bandit agent](/course/open/rl-project-bandit?lang=en)?*")
        elif _r == "slow":
            _out = _live.stopped("*The run took more than two minutes. Do your `step` and your agent do more work than they need?*")
        else:
            learned = _r
            _out = mo.vstack([
                L.explore("rewind", "episodes", lambda _v: C.rewind_frame(_r, int(_v), L.lang), start=C.EPISODES,
                          values=C.LOOKS, role="",
                          task="With the course's world: go back to 50, then forward again. When do the arrows of each method first lead from S to G, and which way do the arrows of SARSA take at 100, and at 300?"),
                mo.md(C.summary(_r, L.lang)),
            ])
    _out
    return (learned,)


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With the course's world and the course's parts, over the last 50 episodes, SARSA has a mean return of −16.7, Q-learning −30.4 and Monte Carlo −42.0. At the end, the arrows of SARSA lead over the top, in 11 steps, in all 10 runs; those of Q-learning lead between the puddles and the pits, in 7 steps. Q-learning's arrows show the shortest way, but the agent follows them with ε = 0.1: one step in ten is random, and next to the pits a random step down costs −100. SARSA learns what its own random steps cost, so it learns to keep away from the pits, as on the cliff. Monte Carlo is the slowest to learn: over all 300 episodes its mean return is −179.9, against −29.9 for SARSA and −37.3 for Q-learning. Of its 3,000 episodes, 719 stop after 200 steps without reaching G; SARSA reaches G in 2,997 of its episodes, Q-learning in 2,995. In the rewind, though, the arrows of Monte Carlo with seed 0 are the first to lead to G, at 50 episodes: seed 0 is one of its best runs, as the NumPy cell below shows.
    """)
    return


@app.cell(hide_code=True)
def text_numpy():
    mo.md(r"""
    ### All runs at once with NumPy

    The mean over 10 runs hides how far the runs lie apart. `C.course_run()` gives the course's run on its own world; under `["returns"]` it has, for each method, the returns of every episode of every run. `np.array` turns those of Monte Carlo and of Q-learning into arrays of shape `(10, 300)`: a row per run, a column per episode. `_mc[:, 250:]` takes the last 50 episodes of every run, and `np.mean(..., axis=1)` averages every row: one number per run. `_last_mc > _last_q` is `True` for the runs where Monte Carlo is ahead, and `.sum()` counts them. Run the cell.
    """)
    return


@app.cell
def example_numpy():
    import numpy as np

    _r = C.course_run()                      # the course's run on its own world
    _mc = np.array(_r["returns"]["mc"])      # shape (10, 300): a row per run, a column per episode
    _q = np.array(_r["returns"]["q"])
    _last_mc = np.mean(_mc[:, 250:], axis=1)   # the mean return of the last 50 episodes, one number per run
    _last_q = np.mean(_q[:, 250:], axis=1)
    print("shape:", _mc.shape)
    print("Monte Carlo, per run:", np.round(_last_mc, 1))
    print("Q-learning, per run: ", np.round(_last_q, 1))
    print("runs where Monte Carlo is ahead:", int((_last_mc > _last_q).sum()))
    print("mean over the runs, Monte Carlo:", round(float(np.mean(_last_mc)), 1), "Q-learning:", round(float(np.mean(_last_q)), 1))
    return


@app.cell(hide_code=True)
def text_after_numpy():
    mo.vstack([
        mo.md(r"""
        The runs of Monte Carlo lie far apart: from −20.0 to −89.6 over the last 50 episodes. Those of Q-learning lie between −19.8 and −38.8. Over the 10 runs, Monte Carlo is far behind, −42.0 against −30.4, but in 4 of the 10 runs it is ahead.
        """),
        L.watch_out("One run can tell a different story from ten. With seed 0 alone, Monte Carlo gets −20.9 over the last 50 episodes and Q-learning −29.4: Monte Carlo looks better. Compare methods over many runs, and look at how far the runs lie apart."),
    ])
    return


@app.cell(hide_code=True)
def text_exercise_maze():
    mo.md(r"""
    ### Exercise 4. A maze

    Put this maze in `my_world`, run the cell, and let your agent learn it with the button:

    ```python
    my_world = [
        "S..#.....",
        ".#.#.###.",
        ".#...#...",
        ".####.#X.",
        "......#.G",
    ]
    ```

    The shortest way from S to G takes 16 steps. With the course's parts, over the last 50 episodes, Q-learning has a mean return of −21.3 and SARSA −53.9, and the arrows of Q-learning lead along the shortest way in all 10 runs. Q-learning reaches G in 2,889 of its 3,000 episodes, SARSA in 2,730; at the end, the arrows of SARSA lead to G in 3 of the 10 runs. Monte Carlo reaches G in 9 of its 3,000 episodes: almost every one of its episodes stops after 200 steps, and its mean return over the last 50 is −200.1. In the course's world, Monte Carlo did learn.
    """)
    return


@app.cell(hide_code=True)
def choice_maze_cell():
    choice_maze = L.choice(
        "maze",
        question="Why does Monte Carlo hardly ever reach G in the maze, while SARSA and Q-learning reach G in most of their episodes?",
        options={
            "luck": "Bad luck: with other seeds, Monte Carlo would learn the maze too",
            "model": "Monte Carlo needs a model of the environment to learn where the walls are, and SARSA and Q-learning do not",
            "right": "Monte Carlo changes no $Q$ during an episode, so it can take the same steps in the same cells again and again until the limit; SARSA and Q-learning lower the $Q$ of each step at once, and so move on to other actions",
            "long": "With $\\gamma$ = 1, the return of a step adds up every reward after it, up to 200 of them; totals that long vary too much to learn a maze from, while the targets of SARSA and Q-learning look only one step ahead and vary much less",
        },
        correct="right",
        why={
            "right": "In its first episode every $Q$ is 0, and on a tie the agent takes the first action, up. From S that is the edge, so in its first episode Monte Carlo keeps walking into the edge, with a random step one time in ten, until the limit. Later, too, a cell it comes back to in an episode gets the same action again, so it walks in circles. SARSA and Q-learning lower $Q(S, A)$ after each step, so a step they have just taken looks worse than one they have not tried yet: within one episode they move on, and find G. The course's world has no walls and a short way to G, so there the random steps of Monte Carlo reach G often enough.",
            "luck": "None of the 10 runs learns the maze: together they reach G in 9 of 3,000 episodes. What changes in the $Q$ values of Monte Carlo while an episode is still going on?",
            "model": "Monte Carlo control is model-free, as SARSA and Q-learning are: all three learn only from steps that happened. What does each of them do with a step at the moment it happens?",
            "long": "Returns vary only through the agent's own random steps, and in the course's world Monte Carlo learns from returns of up to 200 rewards as well. What changes in its $Q$ values while an episode is still going on?",
        },
    )
    choice_maze
    return (choice_maze,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Careful or bold: SARSA and Q-learning](/course/open/rl-sarsa?lang=en) and [Playing better from experience: Monte Carlo control](/course/open/rl-monte-carlo-control?lang=en), the course's code learned the course's worlds. Now your project `agent` has a world you design yourself, one agent that learns it with each of the three targets, and learning curves over many runs, next to your bandit agent and your `average`. Your comparison showed that next to pits SARSA does best while it learns, that one run can tell a different story from ten, and that Monte Carlo, which changes nothing during an episode, can get lost in a maze. Later, *One interface for every environment: Gymnasium* gives many environments the same shape as your `GridWorld`, with `reset` and `step`. The next lesson, *Copying an expert: behavior cloning*, starts the part *Other settings*: an agent that learns from the steps of an expert, without rewards.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_maze, result_agent, result_curves, result_world):
    L.wrap_up(result_world, result_agent, result_curves, choice_maze)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
