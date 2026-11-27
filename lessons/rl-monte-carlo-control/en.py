# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-monte-carlo-control. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the course's blackjack, its exact values and runs, and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-monte-carlo-control", "en")
    C = L.checks   # the course's blackjack, its exact values and runs, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Learning from whole games: Monte Carlo prediction](/course/open/rl-monte-carlo-prediction?lang=en) and [Better and better: policy iteration](/course/open/rl-policy-iteration?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_first():
    quick_first = L.choice(
        "quick-first",
        question="With first-visit Monte Carlo, an agent estimates $V(s)$ of a policy from episodes played with it. What is its estimate of a state $s$ after 50 episodes that each visited $s$?",
        options={
            "last": "The return of the last of the 50 episodes, the most recent one",
            "right": "The average of the 50 returns that followed the first visit of $s$ in each episode",
            "best": "The largest of the 50 returns, the best the policy did from $s$",
            "reward": "The average of the rewards right after each first visit of $s$, one per episode, over all 50 episodes",
        },
        correct="right",
        why={
            "right": "Each first visit gives one return that really happened, and their average comes close to the return to expect from $s$. This lesson does the same for $Q(s, a)$: the average of the returns after action $a$ in state $s$.",
            "last": "One return can be far from the return to expect: the returns of the robot from its corner spread about 32 steps. What does Monte Carlo do with all 50? See [Learning from whole games: Monte Carlo prediction](/course/open/rl-monte-carlo-prediction?lang=en).",
            "best": "The value of a state is the return to expect when you follow the policy, not the best return it ever got: which number of the 50 comes close to what you can expect? See [Learning from whole games: Monte Carlo prediction](/course/open/rl-monte-carlo-prediction?lang=en).",
            "reward": "The value of a state counts every reward to the end of the episode, not only the first: which number of each episode holds them all? See [Learning from whole games: Monte Carlo prediction](/course/open/rl-monte-carlo-prediction?lang=en).",
        },
    )
    quick_first
    return (quick_first,)


@app.cell(hide_code=True)
def quick_check_improve():
    quick_improve = L.choice(
        "quick-improve",
        question="In policy iteration, the improvement step makes a new policy from the values $V$ of the old one. How?",
        options={
            "copy": "It keeps the old policy and only makes $V$ more exact",
            "random": "It picks a random action in every state where $V$ is below the average of all states, to try something new",
            "right": "In every state it takes the action with the largest $Q$, worked out from $V$ and the table `T`",
            "highest": "It moves to the neighboring state with the largest $V$, without looking at `T`",
        },
        correct="right",
        why={
            "right": "$Q(s, a)$ is what it is worth to take $a$ once and follow the old policy from then on, and working it out needs `T`: where each action can lead, and with what probability. Without `T`, this lesson estimates $Q(s, a)$ itself, from hands it plays.",
            "copy": "Policy iteration takes turns: evaluate, then improve. What changes in the improvement step, $V$ or the policy? See [Better and better: policy iteration](/course/open/rl-policy-iteration?lang=en).",
            "random": "The policy improvement theorem says that the new policy is at least as good as the old one in every state: can a random action promise that? See [Better and better: policy iteration](/course/open/rl-policy-iteration?lang=en).",
            "highest": "An action can lead to several states, each with its probability: how does the improvement step weigh them? See [Better and better: policy iteration](/course/open/rl-policy-iteration?lang=en).",
        },
    )
    quick_improve
    return (quick_improve,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Policy iteration made a policy better, but it needed the table `T`. Monte Carlo prediction needed no table, but it only said what a fixed policy is worth. Together they give a way to learn to play well from games alone: estimate what each action brings from the games you play, and play the action that brought more. That is how an agent can learn a card game, a way through traffic or a strategy against people, where nobody can write down the probabilities. Blackjack is the classic example: the course can work out its probabilities, which the agent never sees, and its best policy has been known since the 1950s, so you can see how close learning from hands gets.

    **An everyday example.** When do you stop looking for a parking spot close to the door and take one further away? Nobody works out the odds of every street. You try both now and then, remember how long each choice took in the end, and do more of what worked. If you only ever did what looks best so far, you would never find out whether the other choice was better after all.
    """)
    return


@app.cell(hide_code=True)
def predict_worth():
    guess_worth = L.guess(
        "worth",
        question="In blackjack you take cards to come closer to 21 than the dealer, and over 21 you lose at once. In the course's blackjack, a hand pays +1 for a win, 0 for a draw and −1 for a loss. Monte Carlo control can find the best policy. What does a hand bring on average with that best policy?",
        options={
            "win": "About +0.05: with the best play you win a little in the long run",
            "zero": "About 0: the best play wins as often as it loses",
            "right": "About −0.05",
            "start": "About −0.35, as much as a policy that keeps hitting until 20, since the dealer always has the edge",
        },
        reveal="**About −0.05.** The best policy loses 0.047 per hand: about 5 of every 100 bets go to the dealer. The edge comes from one rule: you play first, and when you go over 21 you lose, even if the dealer would have gone over too. The policy the agent starts with, take a card below 20, loses 0.354 per hand.",
    )
    guess_worth
    return (guess_worth,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The course's blackjack.** You play against the dealer. Cards come from a deck that never runs out: an ace counts 1 or 11, the cards 2 to 9 their number, and 10, jack, queen and king count 10. You get two cards, and the dealer one card that you see and one hidden. You **hit**, take another card, as often as you like; over 21 and you lose at once. When you **stick**, the dealer turns the hidden card and takes cards until the hand counts 17 or more. Closer to 21 than the dealer wins, +1, the same total is a draw, 0, and when the dealer goes over 21 you win. An ace that can count as 11 without taking the hand over 21 is a **usable ace**: with it you can take a card without going over 21, since the ace can go back to 1. The course plays without the extra rules of a casino, such as doubling and splitting: an ace and a 10 as your first two cards are an ordinary 21, which the dealer can tie.

        A state is what you know when you decide: your total from 12 to 21, the dealer's card, and whether you have a usable ace, so 10 · 10 · 2 = 200 states. Below 12 you always take a card, since no card can take you over 21. The reward comes at the end, and with $\gamma$ = 1 the return of every decision in a hand is that one reward.

        **Monte Carlo control.** Prediction estimated the values of a fixed policy. To make a policy better, policy iteration took in each state the action with the largest $Q$, worked out from $V$ with the table `T`. Without `T`, $V$ alone cannot say which action is better: what comes after hitting depends on cards you cannot look up. So **Monte Carlo control** estimates $Q(s, a)$ itself, as the average of the returns that followed action $a$ in state $s$. With $n$ the number of returns of $(s, a)$ so far, the new return $G$ moves it:
        """),
        L.formula(r"[[param:Q(s, a)]] \leftarrow [[param:Q(s, a)]] + \frac{1}{n} \left( [[loss:G]] - [[param:Q(s, a)]] \right)"),
        mo.md(r"""
        It has the shape of the update of Q-learning, with the whole return $G$ as the target in place of $r + \gamma \max_{a'} Q(s', a')$, and with $\eta = 1/n$ as the learning rate. With it $Q$ stays exactly the average of the $n$ returns: after the first return $Q = G_1$, after the second $G_1 + \frac{1}{2}(G_2 - G_1) = \frac{1}{2}(G_1 + G_2)$, and so on. After every hand, the policy takes in each state of the hand the action with the larger $Q$, and on a tie it sticks. Evaluation and improvement take turns hand by hand: generalized policy iteration, without the table. In a hand of blackjack each state comes up at most once, since the total only grows or the ace stops being usable, so first visit and every visit are the same here.

        **Keep trying both actions.** A policy that always takes the action with the larger $Q$ never finds out what the other action brings: the $Q$ of an action it does not take stays where it is. There are two ways out. With **exploring starts**, every episode starts in a random state with a random first action, so every state and action comes up again and again; after that first action, the policy plays. An **ε-soft policy** gives every action a chance of at least $\varepsilon$ divided by the number of actions. ε-greedy from [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en) is one: with probability $\varepsilon$ the move is random, and a random move can also pick the policy's own action. With two actions, each action gets at least $\varepsilon/2$: the half of the random moves that picks it. Monte Carlo control with ε-greedy learns the best among the ε-soft policies: the best way to play when you keep making random moves.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **Monte Carlo control** | | estimating $Q(s, a)$ from episodes and making the policy take the action with the larger $Q$ after every episode | `update` |
        | **exploring starts** | | every episode starts in a random state with a random first action | |
        | **ε-soft policy** | epsilon-soft policy | a policy that gives every action a chance of at least $\varepsilon$ divided by the number of actions | `soft` |
        | **usable ace** | | an ace that counts as 11 without taking the hand over 21 | `s[2]` |
        | **hit**, **stick** | | take another card; stop and let the dealer play | `"hit"`, `"stick"` |
        | $n$ | n | how many returns of $(s, a)$ there have been so far | `count[(s, a)]` |
        """),
        L.watch_out("A casino's blackjack has more rules, such as doubling, splitting and a payout of 3 to 2 for a natural, an ace and a 10 as your first two cards. The best policy of this lesson is the best policy for the course's rules, not for a real table."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Monte Carlo control with exploring starts on the course's blackjack, 500,000 hands. The agent starts with the policy *take a card below 20, stick on 20 and 21*. The two maps show the policy, for hands without and with a usable ace: your total from 21 at the top down to 12, against the dealer's card from the ace to 10. An <span class="c-param">orange</span> cell with H means that the policy hits there, a ring that it differs from the best policy. The chart under them shows what a hand is worth with the policy, worked out by the course from the probabilities of the cards: the agent itself never sees these numbers, only its hands.
    """)
    return


@app.cell(hide_code=True)
def explore_hands_cell():
    explore_hands = L.explore("play", "hands", lambda _v: C.play_frame(int(_v), L.lang), start=C.N_HANDS,
                              values=C.LOOKS, role="",
                              task="Go to 0, then step by step to 500,000: in how many states does the policy differ from the best one after 1,000, 100,000 and 500,000 hands? And why is a hand worth so much more after 1,000 hands, while the maps still have almost as many rings as at the start?")
    explore_hands
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    Before the first hand, the policy differs from the best one in 70 of the 200 states, and a hand is worth −0.354: you lose about a third of every bet. After 1,000 hands it still differs in 66 states, but a hand is worth −0.118: the agent now sticks on 18 and 19 in 32 of their 40 states, where hitting lost the most, and most of the states still wrong are close calls, where the wrong action costs little, or states that real deals rarely reach, so they count little in what a hand is worth. Between 1,000 and 2,000 hands the value even drops a little, to −0.129: an average of a few hands can point the wrong way. After 100,000 hands 4 states differ, and after 500,000 hands none: the policy is the best policy, worth −0.047 per hand. The value climbs fast and then slowly, since the last states to come right are the ones where the two actions are worth almost the same.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Monte Carlo control with exploring starts:

    1. Set $Q(s, a)$ and the count $n$ to 0 for every state and action, and take a policy to start with.
    2. Start a hand in a random state with a random first action; then play with the policy until you stick or go over 21.
    3. The reward at the end is the return of every decision in the hand, since $\gamma$ = 1.
    4. For every state and action of the hand: add 1 to its count $n$, and move $Q(s, a)$ toward the return by $1/n$ of the distance.
    5. In every state of the hand, let the policy take the action with the larger $Q$, and stick on a tie. Then go back to step 2 for the next hand.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    You have 16 without a usable ace, and the dealer shows a 10: hit or stick? The cell estimates $Q$ of both, steps 2 to 4 of the recipe for this one state: it plays 100,000 hands in which you stick on 16, and 100,000 in which you take one card and then stick, and shows the average reward after 100, 1,000, 10,000 and 100,000 hands. `_card()` draws a card, with 11, 12 and 13 as the jack, queen and king, worth 10, and `_dealer(up)` plays the dealer's hand from the card `up` and gives back what it counts at the end. Run the cell.
    """)
    return


@app.cell
def example_sixteen():
    import numpy as np

    _rng = np.random.default_rng(0)


    def _card():
        # 1 is an ace; 11, 12 and 13 are the jack, queen and king, worth 10
        return min(int(_rng.integers(1, 14)), 10)


    def _dealer(up):
        # the hidden card, then cards until the hand counts 17 or more; an ace counts 11 while that stays at 21 or below
        cards = [up, _card()]
        while True:
            total = sum(cards)
            if 1 in cards and total + 10 <= 21:
                total = total + 10
            if total >= 17:
                return total
            cards.append(_card())


    def _reward(player, dealer):
        if player > 21:
            return -1
        if dealer > 21 or player > dealer:
            return 1
        if player == dealer:
            return 0
        return -1


    _stick = []
    _hit = []
    for _i in range(100000):
        _stick.append(_reward(16, _dealer(10)))          # stick on 16
        _hit.append(_reward(16 + _card(), _dealer(10)))  # one card, then stick: an ace counts 1 here, since 11 goes over 21
        if _i + 1 in [100, 1000, 10000, 100000]:
            print("hands:", _i + 1, "stick:", round(sum(_stick) / len(_stick), 3), "hit:", round(sum(_hit) / len(_hit), 3))
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    After 100 hands sticking looks better, −0.54 against −0.58; after 1,000 hitting does, −0.565 against −0.574; after 100,000 sticking is at −0.574 and hitting at −0.573. Worked out from the probabilities, sticking is worth −0.576 and hitting once −0.569: hitting is better by 0.0065, while the difference of two averages of 100,000 hands still spreads about 0.004. Both lose more than half a bet: 16 against a 10 is a bad hand, whatever you do. A close call like this costs little when you get it wrong, and takes many hands to get right.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_update():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Learn from one hand

        `update(Q, count, policy, hand, reward)` does steps 4 and 5 of the recipe after one hand. `Q` and `count` are dictionaries with a key `(s, a)` for each of the 400 pairs of a state and an action, and `policy` is a dictionary from each state to its action, `"stick"` or `"hit"`. A state is a tuple `(total, dealer's card, usable ace)`, such as `(13, 2, False)`. `hand` is the list of the decisions of the hand, each a pair `(s, a)`, and `reward` is the reward at the end: the return of every decision. The function changes `Q`, `count` and `policy` and gives nothing back: a dictionary that a function changes is changed for the code that called it too. Replace the two `...`: the new `Q[(s, a)]`, step 4, and when the policy hits, step 5; on a tie it sticks.

        The check starts from `Q` and `count` at 0 and the start policy, calls `update` for each hand of `HANDS`, and looks at `Q` and `policy` after the last hand. `HANDS` holds five hands against a dealer's 2: hit on 13 and stick on 19, a win; hit on 13 and over 21; hit on 13 and stick on 17, a draw; stick on 19, a draw; stick on 13, a loss. Then four other cases: `ACE_HAND`, one hand against a 6, hit on 15 with a usable ace, a 10 makes it 15 without one, hit again to 21 and stick, a win; `HANDS + ACE_HAND`; `FIRST_HANDS`, the first 1,000 hands of the play block; and `[]`, no hands at all.
        """),
        L.task("Make `update` keep each `Q[(s, a)]` the average of its returns, and let the policy take the action with the larger `Q`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_update():
    box_update = L.code_box("update", lines=10)
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
            "Step 4 of the recipe and the formula of *The idea*: `Q[(s, a)]` moves toward `reward` by 1/`count[(s, a)]` of the distance, and `count[(s, a)]` has already gone up by 1 for this hand. Step 5: which of `Q[(s, \"hit\")]` and `Q[(s, \"stick\")]` must be larger for the policy to hit, and what does it do on a tie?",
            "In `HANDS`, hitting on 13 gets the returns 1, −1 and 0: its `Q` is 1 after the first, 1 + (−1 − 1)/2 = 0 after the second, and 0 + (0 − 0)/3 = 0 after the third. Sticking on 19 gets 1 and then 0: its `Q` is 1, then 1 + (0 − 1)/2 = 0.5.",
            "`        Q[(s, a)] = ___`",
        ],
        why="`Q[(s, a)] = Q[(s, a)] + (reward - Q[(s, a)]) / count[(s, a)]` keeps `Q` the average of all returns after `(s, a)`, and `if Q[(s, \"hit\")] > Q[(s, \"stick\")]:` lets the policy hit only where hitting brought more on average.",
    )
    return


@app.cell(hide_code=True)
def text_after_update():
    mo.md(r"""
    With the course's `update`, the 500,000 hands of the play block end on the best policy. In the picture, your `update` plays the first 100,000 hands, the same hands as the play block as long as it takes the same actions; with the course's, 4 states still differ after them.

    ### The whole policy at once with NumPy

    The dictionaries of `update` are handy for one hand at a time; for all 200 states at once, an array is shorter. `C.play_q()` gives $Q$ after the 500,000 hands of the play block as an array of shape `(2, 10, 10, 2)`: usable ace or not, your total from 12 to 21, the dealer's card from the ace to 10, and the action, 0 for stick and 1 for hit. `np.argmax(_q, axis=3)` gives for every state at once the place of the largest number along the last axis: the action with the larger $Q$, and on a tie the first, stick. `C.best_array()` is the best policy in the shape of `_policy`, `(2, 10, 10)`; `_policy != _best` is `True` where the two differ, and `.sum()` counts the `True`s. Run the cell.
    """)
    return


@app.cell
def example_numpy(np):
    _q = C.play_q()                      # Q after the play block, shape (2, 10, 10, 2)
    _policy = np.argmax(_q, axis=3)      # 0 stick, 1 hit, for all 200 states at once
    _best = C.best_array()
    print("shape of Q:", _q.shape)
    print("states that differ from the best policy:", int((_policy != _best).sum()))
    print("without a usable ace, total 12 at the top, the dealer's ace on the left:")
    print(_policy[0])
    return


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    No state differs. Read the printed array from the top, total 12, against the dealer's card from the ace to 10: on 12, hit against an ace, a 2 and a 3, stick against a 4 to 6, and hit against a 7 or more; on 13 to 16, stick against a 2 to 6 and hit against the rest; from 17 on, always stick. The dealer's 2 to 6 are the weak cards: with them the dealer goes over 21 most often, so you stick early and let the dealer take the risk.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_soft():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. An ε-soft policy

        `soft(policy, s, epsilon)` gives back the chances of the two actions in state `s` under ε-greedy, an ε-soft policy, as a dictionary `chances` with a chance for each action `a`: `{"stick": ..., "hit": ...}`. A random move comes with probability `epsilon`, and picks each action with probability 1/2. The other action already has its chance, `epsilon / 2`. Replace the `...` with the chance of the policy's own action, `policy[s]`. The check calls `soft(START, (20, 10, False), 0.1)` and five other cases; `START` is the start policy, which sticks on 20.
        """),
        L.task("Make `soft` give the policy's own action its chance under ε-greedy."),
    ])
    return


@app.cell(hide_code=True)
def exercise_soft():
    box_soft = L.code_box("soft", lines=8)
    box_soft
    return (box_soft,)


@app.cell(hide_code=True)
def check_soft(box_soft):
    result_soft = L.check("soft", box_soft.value)
    result_soft
    return (result_soft,)


@app.cell(hide_code=True)
def hints_soft():
    L.hints(
        "soft",
        [
            "*The idea*: with probability `epsilon` the move is random, and a random move picks the policy's own action half the time; every other move takes `policy[s]`. The two chances in `chances` add up to 1, and `chances[a]` is the one of the policy's own action.",
            "With ε = 0.1, 9 moves in 10 follow the policy, and 1 in 10 is random, half of which picks the policy's action again: 0.9 + 0.05 = 0.95.",
            "`            chances[a] = ___`",
        ],
        why="`chances[a] = 1 - epsilon + epsilon / 2`: the moves that are not random, $1 - \\varepsilon$, plus the policy's half of the random ones, $\\varepsilon/2$. Written shorter: `1 - epsilon / 2`.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Learning from real deals

    The demo plays 500,000 hands without exploring starts: every hand starts from a real deal, and every decision follows your `soft` with $\varepsilon$ = 0.1. After every hand, your `update` updates $Q$ and the policy. Where your function does not pass its check yet, the course's takes its place. The chart shows what a hand is worth with the policy, solid, and as the agent plays it, with its random moves, dashed; beside it the maps of the policy as it is now.
    """)
    return


@app.cell(hide_code=True)
def demo_button_cell():
    play_button = mo.ui.run_button(label="Play 500,000 hands with my update and soft")
    play_button
    return (play_button,)


@app.cell(hide_code=True)
def demo_play(box_soft, box_update, play_button, result_soft, result_update):
    played = None
    if not play_button.value:
        _out = mo.md("*Press the button to let an agent learn from real deals.*")
    else:
        _mine = [_n for _n, _r in (("update", result_update), ("soft", result_soft)) if _r.state == "correct"]
        _update = run_code(box_update.value).names["update"] if "update" in _mine else C.update
        _soft = run_code(box_soft.value).names["soft"] if "soft" in _mine else C.soft
        _w = C.DEMO_TEXT[L.lang]
        _now = {}

        def _progress(_n, _run):
            if "policy" not in _now:
                _now["chart"].watch(lambda: C.now_picture(_now["n"], _now["policy"], L.lang))
            _now["n"], _now["policy"] = _n, _run["policies"][_n]
            _now["chart"].reference([(_k, _run["played"][_k]) for _k in sorted(_run["played"]) if _k > 0], "", role="ink", replace=True)
            _now["chart"].extend([(_n, _run["worth"][_n])], f"**Hand {C.number(_n, 0, 'en')} of {C.number(C.N_HANDS, 0, 'en')}**")

        _live = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"], "alt": _w["alt"]}, C.N_HANDS, role="ink",
                       low=-0.4, high=0.0, window=1, xlog=True,
                       key=[("ink", False, _w["key_policy"]), ("ink", True, _w["key_played"]), ("ink-3", True, _w["best"])])
        _live.line("**Playing** …")
        _live.reference([(100, C.BEST_VALUE), (C.N_HANDS, C.BEST_VALUE)], "", role="ink-3")
        _now["chart"] = _live
        _r = C.soft_run(_update, _soft, show=_progress)
        if _r == "update":
            _out = _live.stopped("*Your `update` gave an error during the run, or left a state without `\"stick\"` or `\"hit\"`. Does it still pass its check?*")
        elif _r == "soft":
            _out = _live.stopped("*Your `soft` gave an error during the run, or chances that do not add up to 1. Does it still pass its check?*")
        elif _r == "slow":
            _out = _live.stopped("*The run took more than two minutes. Does your function do more work than it needs?*")
        else:
            played = _r
            _theirs = [_n for _n in ("update", "soft") if _n not in _mine]
            if len(_mine) == 2:
                _who = "your `update` and your `soft`"
            elif len(_mine) == 1:
                _who = f"your `{_mine[0]}` and the course's `{_theirs[0]}`"
            else:
                _who = "the course's `update` and `soft`, since yours do not pass their checks yet"
            _end = _r["policies"][C.N_HANDS]
            _out = mo.vstack([
                L.explore("rewind", "hands", lambda _v: C.rewind_frame(_r, int(_v), L.lang), start=C.N_HANDS,
                          values=C.LOOKS, role="",
                          task="Go back to 0, then forward again: how far apart are the solid and the dashed line at the end, and which states still have a ring?"),
                mo.md(f"Played with {_who}. After 500,000 hands the policy differs from the best policy in {len(C.differ(_end))} of 200 states; a hand is worth {C.number(_r['worth'][C.N_HANDS], 3, 'en')} with it, and {C.number(_r['played'][C.N_HANDS], 3, 'en')} as played, with the random moves."),
            ])
    _out
    return (played,)


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With the course's functions, the policy differs from the best policy in 9 of the 200 states after 500,000 hands, and a hand is worth −0.048 with it; as played, with the random moves, −0.079. The random moves cost about 0.03 per hand: the price of trying both actions all the time. Of the 9 states, 6 have a usable ace: real deals give a usable ace in about 15 of 100 decisions, and some of those states hardly ever, while exploring starts began a hand in every state equally often. Two of the other three are states where the best ε-soft policy sticks: 12 against a 3 and 16 against a 10. When one move in twenty goes the other way, a hand that goes on brings less, so sticking is better there. Monte Carlo control with ε-greedy learns the best way to play with random moves, not without.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_casino():
    mo.md(r"""
    ### Exercise 3. Exploring starts at a real table

    In the play block, every hand started in a state that the course picked at random, with a random first action. An agent that wants to learn at a real table, from real deals, cannot do that.
    """)
    return


@app.cell(hide_code=True)
def choice_casino_cell():
    choice_casino = L.choice(
        "casino",
        question="Why can an agent that plays blackjack at a real table not use exploring starts?",
        options={
            "rules": "A real table has other rules, so the states of the course do not exist there",
            "slow": "A real table deals too slowly for the 500,000 hands that exploring starts need, while an ε-soft policy needs far fewer",
            "right": "The cards decide where a hand starts: you cannot choose to begin at 13 against a 2",
            "hidden": "At a real table you cannot see the dealer's hidden card, and exploring starts need it",
        },
        correct="right",
        why={
            "right": "Exploring starts need a game you can start anywhere, such as the course's own blackjack. At a real table the deal decides the state, and the hand starts where the cards put it. There an ε-soft policy keeps both actions in play, from real deals, as in the demo, at the price of the random moves.",
            "rules": "A state is what you know when you decide: your total, the dealer's card and whether you have a usable ace. A real table has those too; what can a player there not choose?",
            "slow": "The demo needed 500,000 hands with an ε-soft policy too, all from real deals: is the number of hands the problem, or where each hand starts?",
            "hidden": "The states of the course do not hold the hidden card either: the agent decides without it in the play block too. What does an exploring start choose that a real deal does not let you choose?",
        },
    )
    choice_casino
    return (choice_casino,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Learning from whole games: Monte Carlo prediction](/course/open/rl-monte-carlo-prediction?lang=en), you estimated $V$ of a fixed policy by averaging returns. Here the same averages estimate $Q(s, a)$, and after every hand the policy takes the action with the larger $Q$: the turns of evaluation and improvement from [Better and better: policy iteration](/course/open/rl-policy-iteration?lang=en), without the table `T`. Exploring starts and ε-soft policies keep both actions in play, as ε-greedy did in [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en). The next lesson, *Learning after every step: TD(0)*, updates values after every step instead of waiting for the end of the episode. In the project at the end of this part, Monte Carlo control learns in a grid world of your own.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_casino, result_soft, result_update):
    L.wrap_up(result_update, result_soft, choice_casino)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
