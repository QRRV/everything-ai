# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-milestones. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the course's game of 21 sticks with its self-play, and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("rl-milestones", "en")
    C = L.checks   # the course's game of 21 sticks, its self-play, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [RL outside games](/course/open/rl-real-world?lang=en) and [Learning after every step: TD(0)](/course/open/rl-td-prediction?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_unseen():
    quick_unseen = L.choice(
        "quick-unseen",
        question="An agent learned only in a simulator, and in the real world it lands in a state the simulator never showed. What does its Q-table hold for that state?",
        options={
            "similar": "The values of the most similar state it did see",
            "right": "The zeros it started with",
            "ask": "Values it asks the simulator for",
            "none": "No row: the agent stops",
        },
        correct="right",
        why={
            "right": "A row of the table only changes when its own state comes by, so a state that never came by keeps the values it started with. The table of this lesson starts with zeros too, and fills them in by playing against itself.",
            "similar": "A Q-table has one row per state, and each row only changes when its own state comes by. What did the row of a state that never came by start with? See [RL outside games](/course/open/rl-real-world?lang=en).",
            "ask": "After training, the agent only looks in its table, and the simulator plays no part any more. What did that row hold before training? See [RL outside games](/course/open/rl-real-world?lang=en).",
            "none": "Every state has its row from the start, and the agent still chooses an action there. With which values does the row of a state that never came by start? See [RL outside games](/course/open/rl-real-world?lang=en).",
        },
    )
    quick_unseen
    return (quick_unseen,)


@app.cell(hide_code=True)
def quick_check_td():
    quick_td = L.choice(
        "quick-td",
        question="The value of a state is 0.3, the target is 0.4, and the learning rate η is 0.5. What is the value after one TD(0) update?",
        options={
            "target": "0.4: the value jumps all the way to the target",
            "plus": "0.7: the value plus the target",
            "eta": "0.5: the learning rate itself",
            "right": "0.35: halfway from 0.3 to the target",
        },
        correct="right",
        why={
            "right": "The update moves the value a part η of the way toward the target: 0.3 + 0.5 × (0.4 − 0.3) = 0.35. In this lesson, the target of a position comes from the position the opponent gets.",
            "target": "With η = 0.5, the value moves only half of the way. What is half of the gap between 0.3 and 0.4? See [Learning after every step: TD(0)](/course/open/rl-td-prediction?lang=en).",
            "plus": "The update adds η times the gap between the target and the value, not the target itself. How big is that gap here? See [Learning after every step: TD(0)](/course/open/rl-td-prediction?lang=en).",
            "eta": "η says how big a part of the gap the value moves, not what the value becomes. What is 0.3 + 0.5 × (0.4 − 0.3)? See [Learning after every step: TD(0)](/course/open/rl-td-prediction?lang=en).",
        },
    )
    quick_td
    return (quick_td,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    For more than sixty years, games have shown how far RL had come. A game has clear rules and a clear winner, a computer can play as many games as it has time for, and everyone knows how strong the best human players are. In 1959, Arthur Samuel described a program that learned checkers by playing against a copy of itself. In 2016, AlphaGo beat Lee Sedol, one of the strongest Go players in the world, by 4 games to 1. Between and after them came backgammon, Atari, chess, StarCraft, Dota 2, racing and a robot hand, and many of those programs learned from games against themselves. This lesson goes through the milestones, and then lets one table learn a small game by playing both sides, to see how much a program finds out with nobody to teach it.

    **An everyday example.** You can learn a board game by playing both sides yourself: you make a move for one side, turn the board around, and look for the best answer for the other. Every game teaches you from both sides, and your opponent is always exactly as good as you are, so it gets better when you do. What you cannot learn this way is a move that neither side ever tries.
    """)
    return


@app.cell(hide_code=True)
def predict_lost():
    guess_lost = L.guess(
        "lost",
        question="In the lesson's game, 21 sticks lie on the table; two players take 1, 2 or 3 sticks in turn, and whoever takes the last stick wins. One table, with a value for every number of sticks, plays both sides for 200 games. At which numbers of sticks will it end up below 0, a bad position for the player who must move, do you think?",
        options={
            "none": "At none: with the right move, every position can be won",
            "few": "At 1, 2 and 3: the end is close, and the opponent may win",
            "right": "At 4, 8, 12, 16 and 20",
            "far": "At the large numbers: far from the end, anything can happen",
        },
        reveal="**At 4, 8, 12, 16 and 20.** After 200 games, the table holds −0.89 at 4 sticks, −0.56 at 8, −0.52 at 12, −0.48 at 16 and −0.31 at 20, and more than 0 everywhere else. The reason: whoever must move at 4 sticks takes 1, 2 or 3, and the opponent takes the rest of the 4 and wins. At 8, the opponent does the same twice. A multiple of 4 is lost against a player who knows this, and every other number is won by leaving the opponent a multiple of 4. Nobody gave the table this rule: it found the rule by playing against itself.",
    )
    guess_lost
    return (guess_lost,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Sixty years of milestones.** These are the milestones of RL in games, with what each program learned from.

        | Year | Program | Task | Learned from | Scale and result |
        |---|---|---|---|---|
        | 1959 | Arthur Samuel's checkers program | checkers | games against a copy of itself, and looking a few moves ahead | ran on an IBM 704; in 1962 it won a game against a strong player, who won the rematch |
        | 1992 | TD-Gammon (Gerald Tesauro) | backgammon | TD learning in a neural network, from games against itself | 1.5 million games in its best-known version; close to the best human players, not better |
        | 2015 | DQN (DeepMind) | 49 Atari 2600 games | only the pixels of the screen and the score, with the same network for every game | more than 75% of the score of a professional game tester in 29 of the 49 games |
        | 2016 | AlphaGo (DeepMind) | Go | games of human experts first, then games against itself, with search | beat Fan Hui, the European champion, 5–0, and Lee Sedol 4–1 |
        | 2017 | AlphaGo Zero | Go | games against itself only, from random play | after 3 days, beat the version that beat Lee Sedol, 100–0 |
        | 2018 | AlphaZero | chess, shogi and Go | games against itself only, with the same algorithm for all three games | 5,000 TPUs, chips made for neural networks, played its games; in chess it passed the program Stockfish after 4 hours |
        | 2019 | AlphaStar (DeepMind) | StarCraft II | human games first, then a league of agents playing each other | Grandmaster, above 99.8% of ranked human players |
        | 2019 | OpenAI Five | Dota 2, five against five | games against itself | 250 years of games a day; beat OG, the world champions, 2–0 |
        | 2019 | a robot hand (OpenAI) | solving a Rubik's Cube | a simulator that kept changing the size of the cube, friction and more | solved 60% of the scrambles that need 15 turns; a fixed solver chose the turns |
        | 2022 | GT Sophy (Sony AI) | racing in Gran Turismo | races in the game, with a reward that punished unsporting collisions | lost a team event against top drivers in July 2021, 70–86, and won the rematch in October, 104–52 |

        **What they share.** Each milestone joined learning from its own games to something that made those games count: looking ahead at moves before choosing one (Samuel, AlphaGo, AlphaZero), a neural network that judges positions it never saw (from TD-Gammon on), or an enormous number of games (OpenAI Five). Search, trying out moves in advance, has lessons of its own later in the course. And most of them learned from games against themselves: **self-play**. Self-play needs no human games and never runs out of opponents, and the opponent is always exactly as strong as the player, so it gets stronger as the player does.

        **Self-play in miniature.** The lesson's game is as small as a game for two can be: 21 sticks, each player takes 1, 2 or 3 in turn, and whoever takes the last stick wins. One table `V` holds, for every number of sticks from 0 to 21, how good the position is for the player who must move: 1 is a sure win, −1 a sure loss. `V[0]` is −1, since with no sticks left the opponent has just taken the last one. The same table plays both sides:

        - **Choosing.** With `sticks` sticks, each move leaves the opponent a position, and the best move leaves the position with the lowest value, since that position is worst for the opponent. With probability ε, a random move instead.
        - **Learning.** After a move from `sticks` to `left` sticks, `V[sticks]` takes a TD step toward the value of the position the opponent got, with its sign turned around: what is good for the opponent is bad for you.
        """),
        mo.Html(C.tables_picture(L.lang)),
        mo.md(r"""
        Before the first game, every value is 0 except `V[0]`. The table takes all the sticks at 2 and 3, since that wins at once, and 1 stick everywhere else: right at only 8 of the 16 winnable positions. After 200 games against itself, with η 0.1 and ε 0.1, the values below 0 sit at exactly the multiples of 4, and at every other position the table takes what leaves the opponent one.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **self-play** | | training an agent by letting it play against copies of itself, so that its opponent gets stronger as it does | one `V` for both players |
        | **AlphaZero** | | a program from DeepMind (2018) that learned chess, shogi and Go from self-play alone, with search and a neural network that suggests moves and judges positions | |
        """),
        L.watch_out("The table could learn this game because the game is tiny: 22 values, and every position comes by again and again. Chess has more positions than could ever come by, so AlphaZero used a neural network that judges positions it never saw, and looked ahead with search before every move."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Self-play in a game for two:

    1. One table, or one network, that says for every position how good it is for the player who must move.
    2. Choose the move that leaves the opponent the worst position, and with probability ε a random move.
    3. After each move, let the value of the position you were in take a step toward the opponent's value, with its sign turned around.
    4. Let the same table play both sides, game after game.
    5. Test it against a player outside its own games: a perfect player, an older version, or people.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell takes the course's table after 200 games of self-play, for 0 to 7 sticks, and looks at the three moves from 7 sticks: how many sticks each move leaves, and how good that position is for the opponent. Run the cell.
    """)
    return


@app.cell
def example_moves():
    # the course's table after 200 games of self-play, for 0 to 7 sticks
    V = [-1.0, 0.99, 0.95, 0.95, -0.89, 0.5, 0.86, 0.85]
    sticks = 7
    for take in (1, 2, 3):
        left = sticks - take
        print("take", take, "leaves", left, "sticks, worth", V[left], "to the opponent")
    return V, left, sticks, take


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Taking 1 or 2 leaves the opponent a position worth 0.86 or 0.5: both good for the opponent. Taking 3 leaves 4 sticks, worth −0.89 to the opponent, a lost position, so that is the move. And `V[7]` itself, 0.85, is close to what this move is worth to the player who makes it, 0.89: the opponent's −0.89 with its sign turned around.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_choose():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Choosing a move

        `choose(V, sticks)` gives the move for the player who must move with `sticks` sticks, using the table `V`. `best` starts at 1, and the loop tries `take` = 1, 2 and 3; the test `take <= sticks` skips a move that would take more sticks than there are. Then `best` becomes `take` when this move leaves the opponent a position with a lower value than the position `best` leaves. Replace the `...` with that test.

        The check calls `choose` with a table for 0 to 7 sticks, `[-1.0, 0.9, 0.7, 0.5, -0.4, 0.3, 0.1, 0.2]`, and 6 sticks: taking 1, 2 or 3 leaves the opponent 5, 4 or 3 sticks, worth 0.3, −0.4 and 0.5 to the opponent. Then five other cases: 1, 2, 3 and 7 sticks with the same table, and 7 sticks with a table of zeros, where all three moves look the same and `best` should stay 1.
        """),
        L.task("Pick the move that leaves the opponent the lowest value."),
    ])
    return


@app.cell(hide_code=True)
def exercise_choose():
    box_choose = L.code_box("choose", lines=6)
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
            "Step 2 of the recipe. The line starts with `if` and the test that `take` is at most `sticks`; after `and` comes the test that compares two values of `V`: of the position this move leaves the opponent, `sticks - take` sticks, and of the position `best` leaves, `sticks - best` sticks.",
            "In the check's case, with 6 sticks, `best` starts at 1, which leaves 5 sticks, worth 0.3. Taking 2 leaves 4, worth −0.4: lower, so `best` becomes 2. Taking 3 leaves 3, worth 0.5: higher than −0.4, so `best` stays 2.",
            "`        if take <= sticks and ___:`",
        ],
        why="`if take <= sticks and V[sticks - take] < V[sticks - best]:` keeps the move whose position is worst for the opponent. The strict `<` changes `best` only for a move that is really better, so among moves that look the same, the first one stays.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_learn():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Learning from the opponent's position

        `learn(V, sticks, left, eta)` gives the new value of `V[sticks]` after a move from `sticks` to `left` sticks, with learning rate `eta`, η. The `return` line is the TD(0) step: `V[sticks]` moves a part `eta` of the way toward `target`. The target is the value of the position the opponent gets, seen from your side: what is good for the opponent is bad for you. Replace the `...` with the target.

        The check calls `learn` with the same table: from 5 sticks the move took 1 and left the opponent 4, with η 0.5. Then five other cases: taking the last 3 sticks, so that `left` is 0; moves from 7 and from 6 sticks that leave 4; the last stick with η 1; and a move from 4 sticks with a table of zeros.
        """),
        L.task("Make `target` the opponent's value, seen from your side."),
    ])
    return


@app.cell(hide_code=True)
def exercise_learn():
    box_learn = L.code_box("learn", lines=3)
    box_learn
    return (box_learn,)


@app.cell(hide_code=True)
def check_learn(box_learn):
    result_learn = L.check("learn", box_learn.value)
    result_learn
    return (result_learn,)


@app.cell(hide_code=True)
def hints_learn():
    L.hints(
        "learn",
        [
            "Step 3 of the recipe: `target` is the value of the position the opponent gets, at `left` sticks, seen from your side, where a good position for the opponent counts as bad.",
            "In the check's case, 4 sticks are worth −0.4 to the opponent: a bad position for the opponent, so worth 0.4 to you. `V[5]` is 0.3, and with η 0.5 it moves halfway to 0.4: 0.35.",
            "`    target = ___`",
        ],
        why="`target = -V[left]` turns the opponent's value into yours. With that one minus sign, one table can play both sides: every game teaches it from the side of the winner and of the loser at once.",
    )
    return


@app.cell(hide_code=True)
def text_after_learn():
    mo.md(r"""
    At first, only the moves at the end of a game learn something, since `V[0]` is the only value that is not 0: whoever takes the last sticks steps toward 1. Game by game, the values spread back from the end, as they did in [Learning after every step: TD(0)](/course/open/rl-td-prediction?lang=en), and every game teaches the table from both sides at once.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Self-play, then a game against a perfect player

    The demo lets one table play both sides of the game: your `learn` and `choose` if they pass their checks, and the course's otherwise, with η 0.1. Choose how many games it plays against itself, and ε, the chance of a random move. The chart counts after every game how many of the 16 winnable positions the table still plays wrong; beside it, the table at that moment. After the games, the table plays one game against a perfect player, moving first from 21 sticks. Then go back through the games.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    games_choice = mo.ui.dropdown(options=C.games_options(L.lang), value="200", label=C.DEMO_TEXT[L.lang]["games"])
    eps_choice = mo.ui.dropdown(options=C.eps_options(L.lang), value=C.number(C.EPSILON, 1, L.lang),
                                label=C.DEMO_TEXT[L.lang]["eps"])
    play_button = mo.ui.run_button(label="Play against itself")
    mo.hstack([games_choice, eps_choice, play_button], justify="start", gap=1.5, wrap=True)
    return eps_choice, games_choice, play_button


@app.cell(hide_code=True)
def demo_play(box_choose, box_learn, eps_choice, games_choice, play_button, result_choose, result_learn):
    if not play_button.value:
        _out = mo.md("*Choose the games and ε, then press the button to let the table play against itself.*")
    else:
        _mine_learn = result_learn.state == "correct"
        _learn = run_code(box_learn.value, {}).names["learn"] if _mine_learn else C.learn
        _mine_choose = result_choose.state == "correct"
        _choose = run_code(box_choose.value, {}).names["choose"] if _mine_choose else C.choose
        _games, _eps = games_choice.value, eps_choice.value
        _w = C.DEMO_TEXT[L.lang]
        _live = L.live(C.chart_labels(L.lang), _games, role="loss", low=0, high=16, window=1,
                       xticks=C.demo_xticks(_games), key=C.chart_key(L.lang))
        _shown = [-1]

        def _progress(_run):
            _live.watch(lambda: C.now_picture(_run, L.lang, _choose))
            _new = [(_g, _run.wrong[_g]) for _g in range(_shown[0] + 1, _run.games + 1)]
            _shown[0] = _run.games
            _live.extend(_new, _w["playing"].format(g=C.number(_run.games, 0, L.lang), n=C.number(_games, 0, L.lang),
                                                    wrong=_run.wrong[-1]))

        _run = C.demo_run(_learn, _choose, _games, _eps, show=_progress, pause=0.02)
        if _run == "error":
            _out = _live.stopped("*Your `learn` or `choose` gave an error, or something other than a number or a move. Do they still pass their checks?*")
        else:
            _out = mo.vstack([
                L.explore("rewind", "game", lambda _v: C.rewind_frame(_run, int(_v), L.lang, _choose), start=_games,
                          values=list(range(0, _games + 1)), role="",
                          task="Go back to game 0, then forward: after which game does the table play every winnable position right?"),
                mo.Html(C.game_picture(_run, L.lang, _choose)),
                mo.md(C.verdict(_run, _mine_learn, _mine_choose, _eps, L.lang, _choose)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With ε 0.1 and 200 games, the table plays every winnable position right from game 98 on, and beats the perfect player: it takes 1 of the 21 sticks, and after every answer takes what brings the opponent back to a multiple of 4. After 25 or 50 games it still plays 8 or 4 positions wrong, and the perfect player wins. Two things may surprise you. With ε 0, no random moves at all, the table plays everything right from game 16 on: in this tiny game, losing lowers the value of the moves that lost, so the next time the table tries another move by itself, and random moves only add noise. With ε 0.3, it plays everything right at game 92, then slips back, and only from game 118 on stays right, since 3 in every 10 moves are random and keep pushing values around. Runs with another seed and ε 0.1 need between 21 and 66 games, and every one ends with its values below 0 at exactly the multiples of 4. In a large game a program never sees most positions, and can keep playing the same line; AlphaZero added random noise to the move choices of its search while it played against itself, so that it kept trying moves it would otherwise skip.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_from_rules():
    mo.md(r"""
    ### Exercise 3. A new game

    A team invents a new board game for two. Nobody plays it well yet, so a program cannot learn from the games of strong players.
    """)
    return


@app.cell(hide_code=True)
def choice_from_rules_cell():
    choice_from_rules = L.choice(
        "from_rules",
        question="Which milestone shows that a program can still learn to play the game well?",
        options={
            "alphago": "AlphaGo, which first copied human experts and then played itself",
            "dqn": "DQN, which learned 49 games from pixels",
            "right": "AlphaGo Zero and AlphaZero, which played only themselves",
            "alphastar": "AlphaStar, which became a Grandmaster",
        },
        correct="right",
        why={
            "right": "AlphaGo Zero and AlphaZero started from random play and learned from games against themselves only, so a new game needs its rules and a lot of computing, nothing more. The table of this lesson also learned from nothing but the rules.",
            "alphago": "AlphaGo first learned to copy the moves of human experts, and only then played against itself. Where would those expert games come from in a game that nobody plays well?",
            "dqn": "DQN learned games for one player, with a score on the screen that told it how it did. Who would be its opponent in a board game for two?",
            "alphastar": "AlphaStar also started by copying human games of StarCraft II. What would it copy in a game without strong players?",
        },
    )
    choice_from_rules
    return (choice_from_rules,)


@app.cell(hide_code=True)
def text_exercise_league():
    mo.md(r"""
    ### Exercise 4. A league of opponents

    In StarCraft II, every strategy can be beaten by another one, as in rock, paper, scissors. AlphaStar did not only play its latest self: it trained in a league of earlier versions and other agents.
    """)
    return


@app.cell(hide_code=True)
def choice_league_cell():
    choice_league = L.choice(
        "league",
        question="Why the league?",
        options={
            "faster": "Playing many agents at once uses less computing",
            "people": "The league was made of human players, who are harder to beat",
            "right": "So that it keeps beating strategies it no longer meets",
            "rules": "StarCraft II does not allow an agent to play itself",
        },
        correct="right",
        why={
            "right": "Against only its latest self, an agent can switch from rock to paper to scissors and back, and forget along the way how to beat what it no longer meets. A league keeps the old strategies around as opponents, so every new version must still beat them.",
            "faster": "A league means more games, not fewer. What can go wrong when an agent only ever meets its latest self, in a game where strategies beat each other in a circle?",
            "people": "The league held agents, not people: earlier versions of AlphaStar and other agents. What does an agent risk when it only meets its latest self?",
            "rules": "Self-play needs only two copies of the same program, and every game for two allows that. So why keep earlier versions around as opponents?",
        },
    )
    choice_league
    return (choice_league,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [RL outside games](/course/open/rl-real-world?lang=en), a mistake had a price, so the agent learned in a simulator. In a game, a mistake costs a lost game and nothing more, and a program can play millions of games against itself: that is how games carried RL from Samuel's checkers to AlphaZero. The table of this lesson learned as in [Learning after every step: TD(0)](/course/open/rl-td-prediction?lang=en), with one change: the target comes from the opponent's position, with its sign turned around. This is the last lesson of the basis of RL. In March 2025, Andrew Barto and Richard Sutton were named the winners of the 2024 Turing Award, the best-known prize in computer science, "for developing the conceptual and algorithmic foundations of reinforcement learning"; their book *Reinforcement Learning: An Introduction* goes deeper into everything this domain covered. Beyond the basis, RL replaces the table with a neural network, as DQN and AlphaZero did.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_from_rules, choice_league, result_choose, result_learn):
    L.wrap_up(result_choose, result_learn, choice_from_rules, choice_league)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
