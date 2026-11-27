# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-many-agents. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the game, the course's runs and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rl-many-agents", "en")
    C = L.checks   # the game, the course's runs of 20 pairs and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [When steps can slip: value iteration](/course/open/rl-value-iteration?lang=en) and [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_transition():
    quick_transition = L.choice(
        "quick-transition",
        question="In a Markov decision process, what does the transition function say for a state s and an action a?",
        options={
            "best": "Which action is best in s",
            "right": "What can happen after a in s, each outcome with its probability and reward",
            "value": "How much reward the agent can expect from s on, if it acts well from there on",
            "path": "The way from s to the goal",
        },
        correct="right",
        why={
            "right": "The transition function is the rule of the world: what follows an action, and how likely each outcome is. In this lesson, the world of an agent holds a second agent: what follows an action of A depends on what B does, and B keeps learning. So the transition function of the world of A changes as B learns.",
            "best": "Which action is best is what the agent works out, with value iteration or Q-learning; the transition function comes with the world. What does the world say about an action? See [When steps can slip: value iteration](/course/open/rl-value-iteration?lang=en).",
            "value": "That is the value of a state, which value iteration computes from the transition function. What does the transition function itself say? See [When steps can slip: value iteration](/course/open/rl-value-iteration?lang=en).",
            "path": "The transition function looks one step ahead, not all the way to a goal: what does it say about that one step? See [When steps can slip: value iteration](/course/open/rl-value-iteration?lang=en).",
        },
    )
    quick_transition
    return (quick_transition,)


@app.cell(hide_code=True)
def quick_check_epsilon():
    quick_epsilon = L.choice(
        "quick-epsilon",
        question="An agent with two actions chooses ε-greedy, with ε = 0.05: with probability ε a random action of the two, and otherwise the one with the largest Q. In what share of its steps does it take the action with the smaller Q?",
        options={
            "never": "Never: ε-greedy always takes the largest Q",
            "right": "In 2.5% of the steps, about 1 in 40",
            "five": "In 5% of the steps",
            "half": "In half of the steps, since there are two actions",
        },
        correct="right",
        why={
            "right": "In 5% of the steps the agent takes a random action, and half of those happen to be the action with the largest Q anyway: 0.05 × ½ = 0.025. In this lesson two agents choose like this, so each of them betrays the other now and then by chance, and the other has to deal with it.",
            "never": "With probability ε the agent takes a random action of the two, whatever its Q's say: how often is that the one with the smaller Q? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
            "five": "In 5% of the steps the action is random, but a random action of the two can also be the one with the largest Q: in what share of the steps is it the other one? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
            "half": "Only the random steps can take the smaller Q, and they are 5% of all steps: what share of those is the action with the smaller Q? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
        },
    )
    quick_epsilon
    return (quick_epsilon,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In 2019, Bowen Baker and colleagues at OpenAI let agents play hide-and-seek in a small world with walls, boxes and ramps. The hiders got +1 at every step when all of them were out of sight of the seekers, and −1 when any of them was seen; the seekers got the opposite. Nobody rewarded using a box or a ramp. At first the agents only ran and chased. After about 25 million episodes, the hiders learned to push boxes together into a shelter. About 75 million episodes later, the seekers learned to bring a ramp to the shelter and climb over its wall, and soon the hiders learned to take the ramps away first. Later still, the seekers learned to climb onto a box and ride it to the shelter, and the hiders learned to lock every box in place before the seekers came. Nobody programmed any of these strategies: each one came up because the other team had learned something, and every time one team learned, the world of the other team changed. In this lesson you watch the same thing at a size you can follow: two agents, two actions, and a game in which both would gain by working together.

    **An everyday example.** In a shared kitchen at work, rinsing your own cup costs a minute. Whatever the others do, leaving your cup in the sink saves you that minute. When everyone leaves their cup, though, the kitchen is a mess for everyone, and that costs each of them more than the minute they saved.
    """)
    return


@app.cell(hide_code=True)
def predict_pairs():
    guess_pairs = L.guess(
        "pairs",
        question="Two agents, A and B, play the game of the picture: in every round both choose at the same time, cooperate or betray. A game is 100 rounds, and Q-learning trains both agents for 300 games, each with a Q-table of its own. Each agent remembers the last round: its state is what both did in it. After training, both take the action with the largest Q. In how many of 20 such pairs, each trained with random numbers of its own, do both agents cooperate in every round of a game, do you think?",
        options={
            "none": "None: betraying pays more, whatever the other does",
            "five": "About 5 of the 20",
            "right": "14 of the 20",
            "all": "All 20: both learn that cooperating pays more",
        },
        reveal="**14 of the 20.** Without memory, not one pair does: each agent then has a single state, and both learn to betray. With memory, 14 pairs cooperate in every round, and in 13 of them both agents learned the same rule, without anyone writing it: cooperate after a round in which both did the same, and betray after a round in which they differed.",
    )
    mo.vstack([mo.Html(C.world_picture(L.lang)), guess_pairs])
    return (guess_pairs,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The game.** In the **prisoner's dilemma**, two players choose at the same time, without knowing what the other chooses: cooperate or betray. The name comes from a story of two suspects who are questioned apart, and who can each keep quiet or betray the other. In the table of this lesson, `PAYOFF`, both get 3 when both cooperate, and 1 when both betray; when one betrays and the other cooperates, the one who betrays gets 5 and the other 0. `PAYOFF` has a pair of moves as its key and gives the reward of the player whose move comes first: `PAYOFF[("betray", "cooperate")]` is 5, and `PAYOFF[("cooperate", "betray")]` is 0.

        **Betraying pays, whatever the other does.** If B cooperates, A gets 5 for betraying and 3 for cooperating. If B betrays, A gets 1 for betraying and 0 for cooperating. Either way betraying gives A more, and the same holds for B. Yet when both betray, both get 1, while both cooperating would give each of them 3. When what is best for each agent leaves all of them worse off, that is a **social dilemma**; the prisoner's dilemma is the best-known one, for two players.

        **Two learners in one world.** **Multi-agent reinforcement learning**, MARL for short, is reinforcement learning with several agents in one environment, each learning while the others learn too. The simplest way, and the one this lesson uses, makes each agent a Q-learner of its own, as in *Learning from your own steps: Q-learning*: its own Q-table, its own rewards, and the other agent as just a part of its world. That part does not stay the same. For A, what a betrayal brings depends on what B does, and B changes as it learns: the same action in the same state can bring 5 today and 1 a hundred games later. In every earlier lesson, the world stayed the same while the agent learned; here it changes as the other agent learns.

        **Rounds and games.** In a round, both agents choose ε-greedy, and each gets its reward from `PAYOFF`. A game is `ROUNDS` rounds, 100. After every round, each agent moves its own Q toward the target r + γ times the largest Q of its next state, as in Q-learning. As far as the agents know, the game goes on, so the target always has a next state. The course trains every pair for 300 games, with η = 0.1, γ = 0.9 and ε = 0.05.

        **Without memory and with memory.** Without memory, an agent is in the same state, `START`, in every round. It cannot tell what happened before, so it can only learn which move pays more on its own, and that is betraying. With memory, the state of an agent is the last round as it saw it: a pair of moves, its own move first and the move of the other second. For A that is `(a, b)`, for B `(b, a)`; in the first round of a game both are in `START`. Now an agent can learn that what it does in this round changes what the other does in the next one, and with γ = 0.9 the rewards of later rounds count too.

        **Emergent behavior.** Behavior of a group of agents that nobody programmed, and that comes from the agents learning together, is **emergent behavior**: the shelters and ramps of hide-and-seek, and the rule in the picture below.

        The picture shows run 1 after 300 games: what A and B do with the largest Q, in each state.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        Without memory, both agents of run 1 betray. With memory, both cooperate in the first round, and after that both follow the same rule: cooperate after a round in which both did the same, and betray after a round in which they differed. Game theory knows this rule as win-stay, lose-shift: keep your move after a round that paid well, 3 or 5, and switch after a round that paid badly, 0 or 1. Nobody gave the rule to the agents; each learned it while the other learned too.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **multi-agent reinforcement learning**, MARL | | reinforcement learning with several agents in one environment, each learning while the others learn too | `QA`, `QB` |
        | **prisoner's dilemma** | | a game for two in which each does better by betraying, whatever the other does, yet both do worse than if both had cooperated | `PAYOFF`, `is_dilemma` |
        | **social dilemma** | | a situation in which each agent gains by acting for itself, but all end up worse off than if all had cooperated | |
        | **emergent behavior** | | behavior of a group of agents that nobody programmed, coming from the agents learning together | |
        """),
        L.watch_out("Q-learning was made for a world that stays the same. With a second learner in it, nothing promises that the two agents settle down: they may, as most pairs with memory do here, or they may keep changing the world of the other."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Training two agents in one world:

    1. Give each agent a Q-table of its own and a reward of its own: here both from `PAYOFF`, each seen from its own side.
    2. In every round, let both agents choose at the same time, ε-greedy, each from its own state.
    3. Give each agent its reward and its next state: the round as it saw it, its own move first.
    4. Move the Q of each agent as in Q-learning; for each agent, the other is part of the world.
    5. After training, let both take the action with the largest Q, and count what the pair does: here, the rounds in which both cooperate.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Two agents that both follow the rule of the picture cooperate in every round. Then, in round 10, A betrays by chance, in one of its random steps:

    | Round | A does | B does | Reward of A | Reward of B |
    |---|---|---|---|---|
    | 9 | cooperate | cooperate | 3 | 3 |
    | 10 | betray, by chance | cooperate | 5 | 0 |
    | 11 | betray | betray | 1 | 1 |
    | 12 | cooperate | cooperate | 3 | 3 |

    In round 11, A is in state `("betray", "cooperate")` and B in `("cooperate", "betray")`: the two did different things, so both betray. In round 12 both are in `("betray", "betray")`: both did the same, so both cooperate, and from there on they cooperate again. Over rounds 10 and 11, A got 5 + 1 = 6, the same as 3 + 3 for cooperating twice, while B got 0 + 1 = 1 instead of 6. With γ = 0.9 the betrayal is worth a little more to A: 5 + 0.9 × 1 = 5.9, against 3 + 0.9 × 3 = 5.7. The slip cost the pair two rounds and gained A next to nothing: the rule answers a betrayal once, and then goes back to cooperating.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rewards():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The rewards of a round

        `rewards(a, b)` gives the rewards of one round in which A plays `a` and B plays `b`, each `"cooperate"` or `"betray"`. `reward_a = PAYOFF[(a, b)]` already looks up the reward of A: the move of A comes first in the key. Replace the `...` with the reward of B, from the same table. `return reward_a, reward_b` gives both back as a pair.

        The check calls `rewards("betray", "cooperate")`: A betrays and B cooperates. Then the other three pairs of moves. The picture shows the table of the game as your function gives it.
        """),
        L.task("Make `rewards` give the reward of A and the reward of B for one round, both from `PAYOFF`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rewards():
    box_rewards = L.code_box("rewards", lines=4)
    box_rewards
    return (box_rewards,)


@app.cell(hide_code=True)
def check_rewards(box_rewards):
    result_rewards = L.check("rewards", box_rewards.value)
    result_rewards
    return (result_rewards,)


@app.cell(hide_code=True)
def hints_rewards():
    L.hints(
        "rewards",
        [
            "Step 1 of the recipe: each agent gets its reward from its own side. `reward_b` looks up the same table, `PAYOFF`, with a key in which the move of B comes first.",
            "In the example, B cooperates while A betrays, so B should get 0, and `PAYOFF[(\"cooperate\", \"betray\")]` is 0. Which of the two names, `a` or `b`, holds the move of B?",
            "`    reward_b = ___`",
        ],
        why="`reward_b = PAYOFF[(b, a)]` looks up the round from the side of B, with its own move first: 0 for B when A betrays it, and 5 for B when B betrays A.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_dilemma():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Is it a dilemma?

        `is_dilemma(payoff)` says whether a game is a prisoner's dilemma: betraying pays more whatever the other does, while both cooperating gives more than both betraying. `payoff` is a table like `PAYOFF`. The first four lines take its rewards out: `both_cooperate` and `both_betray`; `tempted`, what I get when I betray while the other cooperates; and `fooled`, what I get when I cooperate while the other betrays. Replace the two `...`: `betray_pays`, `True` when betraying pays more in both cases, and after `return` the whole test.

        The check calls `is_dilemma(PAYOFF)`, the game of this lesson. Then five other games, `GAME_2` to `GAME_6`, each with rewards of its own; the picture shows all six.
        """),
        L.task("Make `is_dilemma` give `True` for a game in which betraying pays more whatever the other does, while both cooperating gives more than both betraying, and `False` otherwise."),
    ])
    return


@app.cell(hide_code=True)
def exercise_dilemma():
    box_dilemma = L.code_box("is_dilemma", lines=7)
    box_dilemma
    return (box_dilemma,)


@app.cell(hide_code=True)
def check_dilemma(box_dilemma):
    result_dilemma = L.check("is_dilemma", box_dilemma.value)
    result_dilemma
    return (result_dilemma,)


@app.cell(hide_code=True)
def hints_dilemma():
    L.hints(
        "is_dilemma",
        [
            "`betray_pays` compares betraying with cooperating twice: once for when the other cooperates, and once for when the other betrays. Which two of the four names are my rewards when the other cooperates, and which two when the other betrays?",
            "Both comparisons must be true, so join them with `and`, each with `>`; for `PAYOFF` that is 5 > 3 and 1 > 0. After `return` comes `betray_pays`, and also that both cooperating gives more than both betraying.",
            "`    betray_pays = ___`",
        ],
        why="`betray_pays = tempted > both_cooperate and both_betray > fooled` is true when betraying pays more whatever the other does. `return betray_pays and both_cooperate > both_betray` adds that both betraying leaves both worse off than both cooperating.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_together():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Watching a pair

        `together(QA, QB)` lets a trained pair play one game of `ROUNDS` rounds (100), with the Q-table `QA` of A and `QB` of B, and counts the rounds in which both cooperate. `sa` and `sb` are the states of A and B, both `START` in the first round, and `count` starts at 0; `count += 1` adds 1 to it. `greedy(Q, s)` gives the action with the largest Q in state `s`, `"cooperate"` where the two tie or the state is new, so `a` and `b` are the moves of A and B in this round. In `for _ in range(ROUNDS):` the name `_` stands for a number the loop does not need. Replace the two `...`: the test for a round in which both cooperate, and the states for the next round, each with the agent's own move first.

        The check calls `together(QA_1, QB_1)`: A and B of run 1 after 300 games, the pair of the picture in *The idea*. Then runs 4, 7 and 12 after 300 games, run 14 after 100 games, and run 16 after 50. In the picture, your function counts all 20 pairs after 300 games.
        """),
        L.task("Make `together` count the rounds of a game in which both agents cooperate, with the largest Q and memory of the last round."),
    ])
    return


@app.cell(hide_code=True)
def exercise_together():
    box_together = L.code_box("together", lines=10)
    box_together
    return (box_together,)


@app.cell(hide_code=True)
def check_together(box_together):
    result_together = L.check("together", box_together.value)
    result_together
    return (result_together,)


@app.cell(hide_code=True)
def hints_together():
    L.hints(
        "together",
        [
            "Steps 3 and 5 of the recipe. The test after `if`: `a` is `\"cooperate\"` and `b` is too, two tests with `==` joined with `and`. Then `sa, sb` get the round as each agent saw it: `sa` the state of A, `sb` the state of B.",
            "A state is a pair of moves between parentheses, with the agent's own move first. For A, which of `a` and `b` comes first, and for B? Two pairs with a comma between them give `sa` and `sb` at once.",
            "`        sa, sb = ___`",
        ],
        why="`sa, sb = (a, b), (b, a)` gives each agent the round as it saw it: A its own move `a` first, B its own move `b` first. With the same pair for both, B would remember a betrayal by A as one of its own.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### From 0 to 300 games

    The course trained 20 pairs with memory and 20 pairs without memory, each with random numbers of its own, for 300 games. The block below goes through the training, with the games on a log scale. The chart shows the rounds of 100 in which both cooperate: the mean of the 20 runs over the last 10 games, solid with memory and dashed without. Beside it, what A and B of run 1 do after that many games, with the largest Q.
    """)
    return


@app.cell(hide_code=True)
def explore_games_cell():
    explore_games = L.explore("games", "games", lambda _v: C.demo_frame(int(_v), L.lang), start=0,
                              values=C.LOOKS, role="",
                              task="Go from 0 to 300: when do the pairs without memory stop cooperating, and what do the pairs with memory do between 20 and 300 games?")
    explore_games
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    Both kinds of pairs start out cooperating, and they have not learned that: every Q starts at 0, and where the two Q's of a state tie, an agent takes the first action, `"cooperate"`. Then they find out that a betrayal pays. Without memory, the rounds in which both cooperate fall from about 90 to almost 0 within 80 games, and after 200 games not one pair cooperates in every round any more, with the largest Q. With memory they fall too, to about 23 of 100 near game 80, when only a few pairs still cooperate in every round. Then they climb back, as the agents learn that a betrayal gets answered in the next round. After 300 games both cooperate in about 70 of 100 rounds, and 14 of the 20 pairs cooperate in every round. Trained on to 1,000 games, 14 pairs still do.

    The cooperation of the pairs with memory depends on where they started. With `"betray"` as the first action, so that a tie gives a betrayal, the same training ends with not one of the 20 pairs cooperating in every round. Memory makes cooperation possible; it does not make it certain.

    ### All games with NumPy

    `C.history("memory")` gives an array of shape (20, 300), a row per run and a column per game: the rounds of 100 in which both cooperated. `C.history("none")` gives the same without memory, and `C.history("betray_first")` with memory and `"betray"` as the first action; with `"reward_a"` as a second argument, `C.history` gives the reward of A in every game instead. `_both[:, -50:]` takes the last 50 games of every run. `.mean(axis=1)` takes the mean of every row, one number per run, where `.mean()` without `axis` takes the mean of all the numbers. `np.round` rounds every number, and `.astype(int)` turns the rounded numbers into whole numbers, so they print without a decimal point. Run the cell.
    """)
    return


@app.cell
def example_numpy():
    import numpy as np

    for _kind in ("memory", "none", "betray_first"):
        _both = C.history(_kind)   # shape (20, 300): the rounds of 100 in which both cooperated
        _reward = C.history(_kind, "reward_a")   # the same shape: the reward of A in every game
        _runs = _both[:, -50:].mean(axis=1)   # one number per run: its mean over the last 50 games
        print(_kind, _both.shape)
        print("  both cooperate, rounds of 100, last 50 games:", round(float(_both[:, -50:].mean()), 2))
        print("  reward of A per round, last 50 games:", round(float(_reward[:, -50:].mean()) / C.ROUNDS, 2))
        print("  per run:", np.round(_runs).astype(int))
    return (np,)


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    Over the last 50 games, the pairs with memory cooperate in 66.78 of 100 rounds, and A gets 2.46 per round. Without memory that is 2.6 rounds, and 1.14 per round: little more than the 1 of two agents that always betray. With memory, 15 of the 20 runs are above 70, not at 100: with ε = 0.05 each agent still takes a random step now and then, and every slip costs the pair a round or two, as in the worked example. The other five runs stay below 40. A mean over the last 50 games is not the same as where the training ends, though: runs 9 and 17 only start to cooperate in their last few games, and run 5 keeps betraying after a round in which both betrayed, so a single slip can cost it the rest of the game. With the largest Q, all three cooperate in every round after 300 games, while runs 6 and 12, both above 70 here, do not, as the picture of exercise 3 shows. Without memory, 19 runs are at 0; in one run the agents cooperate for a stretch of games, betray, and cooperate again, several times, and after 300 games they betray with the largest Q. With `"betray"` as the first action, 19 runs are at 0 too, and A gets 1.16 per round; the one other run cooperates in 83 of 100 rounds, and in 99 of 100 with the largest Q after 300 games.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pricing():
    mo.md(r"""
    ### Exercise 4. Prices that learn

    In 2020, Emilio Calvano and three colleagues let two Q-learners set the prices of two competing shops, round after round. Each agent got the profit of its own shop as its reward, and remembered the prices of both shops in the last round, like the agents with memory in this lesson. Nobody built them to work together, and they had no way to talk to each other.
    """)
    return


@app.cell(hide_code=True)
def choice_pricing_cell():
    choice_pricing = L.choice(
        "pricing",
        question="What did the two agents learn?",
        options={
            "low": "To undercut each other until both prices were as low as they could go, since the cheaper shop sells more",
            "right": "To keep both prices high, and to answer a price cut with a few rounds of low prices",
            "chaos": "Nothing stable: each agent kept changing its price because the other did",
            "one": "One shop learned to be cheap and won all the customers; the other gave up",
        },
        correct="right",
        why={
            "right": "The agents learned prices well above the level that competition pushes prices to, without talking and without being built for it. When one agent cut its price, the other answered with low prices for a few rounds, and then both went back up, step by step. It is what the agents with memory in this lesson do: a betrayal gets answered, so betraying gains next to nothing. For the buyers it is bad news. Competition law forbids shops to agree on prices, but these agents never agreed on anything, and the paper sees a challenge for the people who enforce that law.",
            "low": "That is what a single round would teach: the cheaper shop sells more. But the agents met again and again, and remembered the last round. What did a price cut bring in the next round?",
            "chaos": "Two learners can keep changing the world of the other, but these agents settled, as most pairs with memory in this lesson did. At what prices?",
            "one": "Both agents learned with the same method and the same kind of reward, and both remembered the last round. What did a price cut by one of them bring in the next round?",
        },
    )
    choice_pricing
    return (choice_pricing,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Careful or bold: SARSA and Q-learning](/course/open/rl-sarsa?lang=en) one agent learned in a world that stayed the same. Here each agent was part of the world of the other, and that world changed while both learned. Without memory, two Q-learners in the prisoner's dilemma learned to betray each other. With memory of one round, most pairs found a rule that nobody gave them, and cooperated; where they started still mattered a great deal. The same happens at a larger scale: in hide-and-seek, each new strategy of one team led to the next strategy of the other, and two pricing agents learned to keep their prices high without ever talking. The next lesson, *One interface for every environment: Gymnasium*, starts the part on reinforcement learning in practice.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_pricing, result_dilemma, result_rewards, result_together):
    L.wrap_up(result_rewards, result_dilemma, result_together, choice_pricing)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
