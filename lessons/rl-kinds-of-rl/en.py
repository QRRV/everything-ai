# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-kinds-of-rl. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the map, the random walk, the three machines and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-kinds-of-rl", "en")
    C = L.checks   # the map, the random walk, the three machines and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Careful or bold: SARSA and Q-learning](/course/open/rl-sarsa?lang=en) and [Playing better from experience: Monte Carlo control](/course/open/rl-monte-carlo-control?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_sarsa():
    quick_sarsa = L.choice(
        "quick-sarsa",
        question="An agent learns with ε-greedy, ε = 0.1. Which method learns the values of always taking the best action, while the agent still takes random steps?",
        options={
            "sarsa": "SARSA, since it chooses $A'$ with ε-greedy",
            "right": "Q-learning, since its target takes the largest $Q$",
            "both": "Both, since after enough episodes they learn the same values",
            "none": "Neither: the random steps always end up in the values",
        },
        correct="right",
        why={
            "right": "Q-learning is off-policy: its target $R + \\gamma \\max_{a'} Q(S', a')$ holds the best action of $S'$, whatever the agent does next. SARSA is on-policy: it learns the values of ε-greedy itself.",
            "sarsa": "SARSA puts the $Q$ of the action it really takes next in its target, random steps included: whose values does it learn then? See [Careful or bold: SARSA and Q-learning](/course/open/rl-sarsa?lang=en).",
            "both": "On the cliff, after 500 episodes, the two still walk different ways: which one learned the way of an agent that always takes its best action? See [Careful or bold: SARSA and Q-learning](/course/open/rl-sarsa?lang=en).",
            "none": "One of the two targets takes the largest $Q$ of the next state, whatever the agent does next: which one? See [Careful or bold: SARSA and Q-learning](/course/open/rl-sarsa?lang=en).",
        },
    )
    quick_sarsa
    return (quick_sarsa,)


@app.cell(hide_code=True)
def quick_check_mc():
    quick_mc = L.choice(
        "quick-mc",
        question="Monte Carlo control learns $Q(s, a)$ from hands of blackjack. When does it update the $Q$ of the steps of a hand?",
        options={
            "step": "After every card, from its estimate of the next state",
            "right": "After the hand, with the return from each step on",
            "start": "Before the hand, from the rules of blackjack",
            "all": "Only after all hands, with the mean over every hand played so far",
        },
        correct="right",
        why={
            "right": "Monte Carlo waits until the episode is over: only then is the return $G$ of every step known, and each $Q(s, a)$ of the hand moves toward its own $G$.",
            "step": "Updating after every step from an estimate of the next state is what TD learning does: what does Monte Carlo wait for? See [Playing better from experience: Monte Carlo control](/course/open/rl-monte-carlo-control?lang=en).",
            "start": "Monte Carlo control never gets the rules: it only sees the hands it plays. What does it need from a hand before it can update? See [Playing better from experience: Monte Carlo control](/course/open/rl-monte-carlo-control?lang=en).",
            "all": "The agent improves its policy while it plays, so it cannot wait for all hands: after which part does it already know every return? See [Playing better from experience: Monte Carlo control](/course/open/rl-monte-carlo-control?lang=en).",
        },
    )
    quick_mc
    return (quick_mc,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    So far you have met eight RL methods, from value iteration to SARSA, and RL has many more. Papers, libraries and job descriptions sort them with a few words: model-free or model-based, value-based or policy-based, on-policy or off-policy. These words say what a method needs and what it can do: whether it must know how its environment works, whether it can learn from steps that an older policy took, whether it has to wait for the end of an episode. With four questions you can place a method you have never seen, and know what to expect from it before you read its details.

    **An everyday example.** Three ways to find your way in a new city. With a map, you can plan the whole route at your table before you take a step. Without a map, you walk and remember which streets got you somewhere, and how good each street was. Or you remember no streets at all, only habits: at a big square, turn left. The first plans with a model of the city, the second learns values, and the third learns a policy directly.
    """)
    return


@app.cell(hide_code=True)
def predict_walks():
    guess_walks = L.guess(
        "walks",
        question="The random walk of *Learning after every step: TD(0)*: five states A to E in a row; every walk starts in C and steps left or right with a chance of 1/2 each, until it leaves the row, and leaving on the right pays 1. Policy evaluation knows these chances: it has the model. Starting from 0.5 for every state, as TD(0) did, how many walks does it need before every value is within 0.01 of the true one?",
        options={
            "hundred": "About 100, as many as TD(0) got in its lesson",
            "thousand": "About 1,000: with the model each walk teaches it more, but 0.01 is close",
            "right": "None: it needs no walks",
            "many": "More than 10,000, since it goes through every walk that can happen",
        },
        reveal="**None: it needs no walks.** Policy evaluation does not walk at all. With the model it knows that from C the walk goes to B or to D, each with a chance of 1/2, so it updates $V(C)$ from $V(B)$ and $V(D)$ directly. After 6 sweeps over the five states, 30 updates, every value is within 0.01 of the true one, from $V(A) = 1/6$ to $V(E) = 5/6$. In its lesson, TD(0), which only sees walks, was after 100 walks still 0.048 from the true values on average. And going through every walk: all walks from C of up to 33 steps, 43,046,720 of them, only pin $V(C)$ between 0.495 and 0.505.",
    )
    guess_walks
    return (guess_walks,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Four questions.** Each of the eight methods answers four questions.

        1. **Does it use a model of the environment?** A model says, for a state and an action, which next states can follow, with what chance, and which reward: the table `T` of policy evaluation is one. A method that uses a model, given or learned, is **model-based RL**. A method that learns only from steps that happened, and never predicts a next state, is **model-free RL**.
        2. **Does it learn values or the policy?** A **value-based method** learns $V$ or $Q$, and its policy follows from them, such as ε-greedy on $Q$. A **policy-based method** learns the policy itself: for every state, the chance of each action, and it chooses with those chances, not from values.
        3. **On-policy or off-policy?** Does it learn about the policy it follows, as SARSA does, or about a different one, as Q-learning does?
        4. **Whole episodes or its own estimates?** Monte Carlo waits for the end of the episode and uses the return. TD learning updates after one step, from its own estimate of the next state: bootstrapping.

        **The map.** Two questions about one update draw a map. Across: how far does an update look, one step and then an estimate, or the whole episode? Down: does it look at all next states, with their chances, which takes a model, or at the one next state that happened? Policy evaluation, policy iteration and value iteration look one step ahead, at all next states; together they are called **dynamic programming**. TD learning looks one step ahead at the next state that happened, and Monte Carlo at the whole episode that happened. The fourth corner, every next state all the way to the end, is every path: it gets close to the exact value, but even the random walk needs millions of walks for that.

        **A policy-based agent.** All eight methods on the map are value-based. Here is a policy-based one, on a task without states. Three machines, 0, 1 and 2, pay 1 coin with a chance of 0.2, 0.5 and 0.8, and otherwise nothing; the agent does not know these chances. It keeps a list `p` with the chance that it pulls each machine, 1/3 each at the start, and every turn it draws a machine with those chances. After a pull that pays a coin, every chance moves $\eta$ of the way toward its target $t_a$: 1 for the machine it pulled, and 0 for the others:
        """),
        L.formula(r"[[param:p_a]] \leftarrow [[param:p_a]] + [[hyper:\eta]] \left( [[data:t_a]] - [[param:p_a]] \right)"),
        mo.md(r"""
        After a pull that pays nothing, nothing changes. The targets add up to 1, so the chances keep adding up to 1. A machine that pays often gets pulled more and more; once its chance is close to 1, the agent pulls almost only that machine. The agent never estimates what a machine pays: it has no values, only its policy.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **model-based RL** | | RL that uses a model of the environment, given or learned, to plan or to update | |
        | **model-free RL** | | RL that learns only from steps that happened, without a model | |
        | **value-based method** | | a method that learns $V$ or $Q$ and chooses its actions from them | |
        | **policy-based method** | | a method that learns the policy itself: the chance of each action | `p` |
        | **dynamic programming** | | policy evaluation, policy iteration and value iteration: updates over all next states, with the model | |
        | $p_a$ | p a | the chance that the agent pulls machine $a$ | `p[a]` |
        | $t_a$ | t a | the target of $p_a$: 1 for the machine pulled, 0 for the others | `target` |
        """),
        L.watch_out("A chance of exactly 1 never changes again: the agent pulls only that machine, and a pull of that machine moves its chance toward 1, where it already is. If that happens early, on a machine that is not the best, the agent never finds out."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    The eight methods you know so far, one at a time. On the left the map, with the chosen method marked in <span class="c-param">orange</span>. On the right the random walk, with C outlined in orange: one update of the chosen method changes the value of C, and what it looks at is drawn in ink, both next states with their chances, the one step that happened, or a whole walk.
    """)
    return


@app.cell(hide_code=True)
def explore_map_cell():
    explore_map = L.explore("map", "method", lambda _v: C.play_frame(int(_v), L.lang), start=1, values=C.STOPS,
                            role="", task="Go through the eight methods. Which ones look at both B and D in one update, and which at only one of them? Which method is the only one that is off-policy?")
    explore_map
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    All eight are value-based: they learn $V$ or $Q$. The three methods of dynamic programming look at both B and D, each with its chance from the model; the other five look only at what happened, here the step to D. Monte Carlo waits until the walk has left the row, and TD learning uses its estimate of D at once. Q-learning is the only one of the eight that is off-policy. The fourth corner stays empty: every path is possible only for tiny tasks.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Placing a method on the map:

    1. Model: does it use knowledge of which next states follow an action, and with what chance, given or learned? Then it is model-based RL; otherwise model-free RL.
    2. Values or the policy: does it keep $V$ or $Q$ and choose from them? Then it is a value-based method. Does it keep a chance for each action and change those chances directly? Then it is a policy-based method.
    3. Whose values: those of the policy it follows? Then it is on-policy. Those of a different policy? Then it is off-policy.
    4. When it updates: after the episode, with the return? Whole episodes. After a step, with its own estimate? Its own estimates.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    **One pull of the policy-based agent.** The chances are `p` = `[0.2, 0.3, 0.5]` and $\eta$ = 0.1, and the agent pulls machine 2, which pays a coin. The targets are 0, 0 and 1:

    - machine 0: $0.2 + 0.1 \cdot (0 - 0.2) = 0.2 - 0.02 = 0.18$
    - machine 1: $0.3 + 0.1 \cdot (0 - 0.3) = 0.3 - 0.03 = 0.27$
    - machine 2: $0.5 + 0.1 \cdot (1 - 0.5) = 0.5 + 0.05 = 0.55$

    Together still 1: 0.18 + 0.27 + 0.55 = 1. Had machine 2 paid nothing, `p` would have stayed `[0.2, 0.3, 0.5]`.

    **The agent on the map.** It uses no model: model-free RL. It learns chances of actions, not values: a policy-based method. It pulls with the chances it updates, so it learns about the policy it follows: on-policy. And every pull is a whole episode, with the reward of that pull as its return: the agent has no estimate to update from.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. With or without a model

    Seven agents. Put each in its group: model-based with a model it was given, model-based with a model it learned from its own tries, or model-free.
    """)
    return


@app.cell(hide_code=True)
def exercise_kinds():
    sort_kinds = L.sort("kinds")
    sort_kinds
    return (sort_kinds,)


@app.cell(hide_code=True)
def check_kinds(sort_kinds):
    result_kinds = L.check("kinds", sort_kinds)
    result_kinds
    return (result_kinds,)


@app.cell(hide_code=True)
def hints_kinds():
    L.hints(
        "kinds",
        [
            "For each agent, ask question 1 of the recipe: does it ever use knowledge of which next states follow an action, and with what chance? If it does, where does that knowledge come from?",
            "A table, a map or the rules of a game that someone wrote down are a model the agent was given. A model can also be learned: the agent first measures what its actions do, then plans with what it measured.",
            "The route planner, the board game program and policy iteration were given their model. Which two of the others learn one, and which two never predict what an action leads to?",
        ],
        why="**Given:** the route planner (the roads), the board game program (the rules), policy iteration on HALL (the table `T`). **Learned:** the heating, the robot arm. **Model-free:** the blackjack agent with Monte Carlo control, and Q-learning on the cliff: they learn only what their tries were worth.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_nudge():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A policy-based agent

        `nudge(p, action, reward, eta)` gives back the new list of chances after one pull. `p` holds the chance of each machine, `action` is the machine the agent pulled, `reward` is 1 or 0, and `eta` is $\eta$. After a reward of 0 the function gives back `p` unchanged. Otherwise it builds the list `new`: for each machine `a`, `target` is 1 if `a` is the machine pulled and 0 otherwise, and `new.append(x)` puts the number `x` at the end of `new`. Replace the `...` with the new chance of machine `a`.

        The check calls `nudge([0.2, 0.3, 0.5], 2, 1, 0.1)`, the pull of the worked example. Then four other cases: a pull that pays nothing, $\eta$ = 0.5, four machines, and $\eta$ = 0.05.
        """),
        L.task("Make `nudge` move the chance of every machine η of the way toward its target, 1 for the machine pulled and 0 for the others."),
    ])
    return


@app.cell(hide_code=True)
def exercise_nudge():
    box_nudge = L.code_box("nudge", lines=11)
    box_nudge
    return (box_nudge,)


@app.cell(hide_code=True)
def check_nudge(box_nudge):
    result_nudge = L.check("nudge", box_nudge.value)
    result_nudge
    return (result_nudge,)


@app.cell(hide_code=True)
def hints_nudge():
    L.hints(
        "nudge",
        [
            "Look at the formula of *The idea* and the table of new words: which symbol is the old chance, and which the target? In the code the chances are in `p`, `a` is the machine in turn, $\\eta$ is `eta`, and `append` puts a number at the end of the list `new`.",
            "In the worked example, machine 0 had chance 0.2 and target 0: $0.2 + 0.1 \\cdot (0 - 0.2)$. Inside the loop, which two names hold the target and the old chance of machine `a`?",
            "`        new.append(p[a] + eta * (___))`",
        ],
        why="`new.append(p[a] + eta * (target - p[a]))` moves the chance of machine `a` $\\eta$ of the way toward its target: up for the machine pulled, down for the others. The targets add up to 1, so the new chances add up to 1 too.",
    )
    return


@app.cell(hide_code=True)
def text_after_nudge():
    mo.md(r"""
    In the picture, your `nudge` plays run 1 of the demo, with $\eta$ = 0.1. After 50 pulls the chance of the machine that pays 0.8 is already 0.95, and after 100 pulls 0.999: from then on the agent pulls only that machine. The demo shows how often that goes right over 200 runs.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Your `nudge` beside a value-based agent

    The demo plays 200 runs of 1,000 pulls with your `nudge` for the policy-based agent; where your function does not pass its check yet, the course's takes its place. Beside it runs a value-based agent: it keeps the mean of what each machine paid so far, and picks with ε-greedy, ε = 0.1. Its estimates all start at 0, and of equal estimates it takes the first machine, the one that pays 0.2: at the start it pulls that machine, until its random picks find a better one. Pick $\eta$ for the policy-based agent: with 0.1 you see the course's run; then try 0.01, 0.2 and 0.5. The chart shows, for every pull, the share of the 200 runs that pulled the best machine, the one that pays 0.8, as a mean over the last 10 pulls: policy-based solid, value-based dashed. Beside it, run 1 as it is now: the chances of the policy-based agent filled, and the estimates of the value-based agent open.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    eta = mo.ui.dropdown(options=C.eta_options(L.lang), value=C.eta_text(0.1, L.lang), label=C.DEMO_TEXT[L.lang]["eta"])
    play_button = mo.ui.run_button(label="Run 200 runs with my nudge")
    mo.hstack([eta, play_button], justify="start", gap=1.5, wrap=True)
    return eta, play_button


@app.cell(hide_code=True)
def demo_play(box_nudge, eta, play_button, result_nudge):
    pulled = None
    if not play_button.value:
        _out = mo.md("*Pick η, then press the button to let both agents play the three machines.*")
    else:
        _mine = result_nudge.state == "correct"
        _fn = run_code(box_nudge.value).names["nudge"] if _mine else C.nudge
        _w = C.DEMO_TEXT[L.lang]
        _now = {"shown": 0}

        def _progress(_n, _run):
            if "run" not in _now:
                _now["chart"].watch(lambda: C.now_picture(_now["run"], _now["n"], L.lang))
            _now["run"], _now["n"] = _run, _n
            if _n % 50 == 0 or _n == 1:
                _now["chart"].reference(C.chart_points(_run, "value", 1, _n), "", role="ink", replace=True)
                _now["chart"].extend(C.chart_points(_run, "policy", _now["shown"] + 1, _n), f"**Pull {C.number(_n, 0, 'en')} of {C.number(C.PULLS, 0, 'en')}**")
                _now["shown"] = _n

        _live = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"], "alt": _w["alt"]}, C.PULLS, role="ink",
                       low=0.0, high=1.0, window=1, key=C.curve_key(L.lang))
        _live.line("**Pulling** …")
        _now["chart"] = _live
        _r = C.run(_fn, eta.value, show=_progress)
        if _r == "nudge":
            _out = _live.stopped("*Your `nudge` gave an error during the run, or gave something other than chances that add up to 1. Does it still pass its check?*")
        elif _r == "slow":
            _out = _live.stopped("*The run took more than a minute. Does your function do more work than it needs?*")
        else:
            pulled = _r
            _who = "your `nudge`" if _mine else "the course's `nudge`, since yours does not pass its check yet"
            _s = C.settled(_r)
            _out = mo.vstack([
                L.explore("rewind", "pulls", lambda _v: C.rewind_frame(_r, int(_v), L.lang), start=C.PULLS,
                          values=C.LOOKS, role="",
                          task="Go back to 10, then forward again: where do the two lines come closest, and when does the solid line pull away for good? What do the bars of run 1 look like then?"),
                mo.md(f"Run with {_who} and η = {C.eta_text(eta.value, 'en')}. After 1,000 pulls, the best machine has a chance above 0.99 in {_s[C.BEST]} of the 200 runs, the one that pays 0.5 in {_s[1]}, the one that pays 0.2 in {_s[0]}, and no machine yet in {_s[None]}. Over all pulls, the policy-based agent got {C.number(_r['reward']['policy'], 3, 'en')} coins per pull, and the value-based agent {C.number(_r['reward']['value'], 3, 'en')}."),
            ])
    _out
    return (pulled,)


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With the course's `nudge` and $\eta$ = 0.1, 198 of the 200 runs end with a chance above 0.99 for the best machine, and 2 for the machine that pays 0.5. Over the last 100 pulls, the policy-based agent pulls the best machine 99.0% of the time and the value-based agent 93.1%: with ε = 0.1, it keeps picking a random machine in one of every ten pulls. Over all 1,000 pulls, the policy-based agent gets 0.782 coins per pull and the value-based agent 0.752. A larger $\eta$ settles faster, and more often on a machine that is not the best: with $\eta$ = 0.5, 50 runs end on the machine that pays 0.5 and 7 on the one that pays 0.2, at 0.701 coins per pull. With $\eta$ = 0.01, 196 of the 200 runs have not settled after 1,000 pulls. The value-based agent explores with ε as long as it plays; the policy-based agent explores less and less by itself, as its chances move toward 0 and 1. Policy-based methods for tasks with states, which learn with gradients, come later in the course.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_paper():
    mo.md(r"""
    ### Exercise 3. A method from a paper

    A paper describes an agent that learns to play video games from the pixels on the screen. Nobody gives it the rules of a game, and it never learns to predict what a step will do. It keeps its last million steps in a memory. Every few moves it picks steps from that memory at random, many of them from long ago, when it played with an older policy, and moves their $Q(s, a)$ toward $r + \gamma \max_{a'} Q(s', a')$.
    """)
    return


@app.cell(hide_code=True)
def choice_paper_cell():
    choice_paper = L.choice(
        "paper",
        question="Where does this agent sit on the map?",
        options={
            "model": "Model-based, value-based, off-policy, from its own estimates: its memory of old steps is a model of the game",
            "right": "Model-free, value-based, off-policy, from its own estimates",
            "on": "Model-free, value-based, on-policy, from whole episodes",
            "policy": "Model-free, policy-based, off-policy, from its own estimates",
        },
        correct="right",
        why={
            "right": "It never predicts a next state: model-free RL. It learns $Q$ and picks from it: a value-based method. Its target takes the largest $Q$ of $s'$, and it learns from steps of older policies: off-policy, as Q-learning. And it updates while it plays, from its own estimate of $s'$, not from whole episodes. This is the agent of Mnih and others (2015), which learned 49 Atari games: Q-learning with a network in place of the table.",
            "model": "The memory holds steps that happened. Can it say what would follow an action the agent never took in that state?",
            "on": "Its target takes the largest $Q$ of $s'$, whatever the older policy did there, and it does not wait for the end of a game: which method on the map does the same?",
            "policy": "What does it learn: a chance for each action, or $Q$?",
        },
    )
    choice_paper
    return (choice_paper,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In this part of the course you built dynamic programming in [What is a policy worth? Policy evaluation](/course/open/rl-policy-evaluation?lang=en) and [Better and better: policy iteration](/course/open/rl-policy-iteration?lang=en), Monte Carlo in [Playing better from experience: Monte Carlo control](/course/open/rl-monte-carlo-control?lang=en), and TD learning in [Learning after every step: TD(0)](/course/open/rl-td-prediction?lang=en) and [Careful or bold: SARSA and Q-learning](/course/open/rl-sarsa?lang=en). This lesson puts them on one map, with four questions, and adds a policy-based agent beside them. The next lesson, *Project: an agent for your own grid world*, lets Monte Carlo control, SARSA and Q-learning learn a grid world you design yourself, side by side.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_paper, result_kinds, result_nudge):
    L.wrap_up(result_kinds, result_nudge, choice_paper)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
