# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson search-minimax. Route: course/route/lessen/overige-takken.yaml. Checks, the positions and the play block: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("search-minimax", "en")
    C = L.checks   # the positions, and the course's own minimax that draws the play block


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [A*: search with a sense of direction](/course/open/search-astar?lang=en) and [A function that calls itself](/course/open/py-recursion?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_state():
    quick_state = L.choice(
        "quick-state",
        question="In the maze of *A\\*: search with a sense of direction*, what is a state?",
        options={
            "maze": "The whole maze, with its walls",
            "right": "The cell you stand in, as `(row, column)`",
            "path": "The path you took so far",
        },
        correct="right",
        why={
            "right": "A state is one situation of the task; in the maze, where you stand. The actions lead from one state to the next.",
            "maze": "The maze stays the same during the search; what changes is where you stand. See [A*: search with a sense of direction](/course/open/search-astar?lang=en).",
            "path": "A path is a list of states, one action at a time; a state is one of them. See [A*: search with a sense of direction](/course/open/search-astar?lang=en).",
        },
    )
    quick_state
    return (quick_state,)


@app.cell(hide_code=True)
def quick_check_recursion():
    quick_recursion = L.choice(
        "quick-recursion",
        question="What does this program show?\n\n```python\ndef total(xs):\n    if len(xs) == 0:\n        return 0\n    return xs[0] + total(xs[1:])\n\nprint(total([4, 1, 2]))\n```",
        options={
            "first": "`4`",
            "right": "`7`",
            "base": "`0`",
        },
        correct="right",
        why={
            "right": "`total([4, 1, 2])` is 4 + `total([1, 2])`, so 4 + 1 + 2 + 0: the empty list is the base case.",
            "first": "`total` does not stop at the first number: it adds `total(xs[1:])`, the same question for the rest of the list. See [A function that calls itself](/course/open/py-recursion?lang=en).",
            "base": "0 is what the base case gives back, for the empty list; on the way back up, every call adds its own first number. See [A function that calls itself](/course/open/py-recursion?lang=en).",
        },
    )
    quick_recursion
    return (quick_recursion,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        In a maze, only you move, so a path to the goal is a plan. In a game, an opponent moves too, and wants the opposite of what you want: after each of your moves, they choose. How do you pick a move when the other player decides what happens next?

        **An everyday example.** You play tic-tac-toe with a friend. Before you put down your X, you think: if I go here, where does she go? And then me? You do not count on a mistake: you expect her best answer.

        Here the squares are numbered 0 to 8, row by row, and X is to move.
        """),
        C.squares_and_game(("The squares", "X to move"), "Two boards of 3 by 3 squares. The first shows the number of each square, 0 to 8 row by row: 0, 1 and 2 on top, 6, 7 and 8 at the bottom. The second is a game with X to move: top row X, O, X; middle row O and two empty squares, 4 and 5; bottom row O, X and the empty square 8."),
    ])
    return


@app.cell(hide_code=True)
def predict_end():
    guess_end = L.guess(
        "end",
        question="If both players play their best from this position, how does the game end?",
        options={
            "win": "X wins",
            "draw": "A draw",
            "loss": "O wins",
        },
        reveal="**X wins**, with square 8; with square 4, in the middle, it is a draw. *Minimax*, the search of this lesson, works this out for every position, and the play block below shows how.",
    )
    guess_end
    return (guess_end,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A game where two players take turns is a search problem with an opponent. A state is a position: the marks on the board, and who is to move. An action is a move: here, a mark in an empty square. A position where someone has three in a row, or where the board is full, ends the game.

        All positions you can reach from one position, move by move until each game ends, form the **game tree**: that position at the top, and each level down one turn. Each finished game has a value: +1 when X won, −1 when O won, 0 for a draw. X wants the value as large as possible, O as small as possible.

        **Minimax** gives every position the value the player to move can force, if the other plays their best as well. With X to move, it is the largest value among the positions after X's moves; with O to move, the smallest. Each of those values is the same question one move further, so minimax calls itself, and the finished games are its base case.

        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **game tree** | all positions you can reach from one position, move by move; each level is one player's turn | |
        | **minimax** | a search that gives every position the value the player to move can force: for X the largest value after its moves, for O the smallest | `minimax(board, player)` |
        """),
        L.watch_out("*Value* is a plain word here: the value of a position is how the game ends from there when both play their best, +1, 0 or −1."),
    ])
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    Minimax on the position of the guess, call by call. Under each position is its value, a `?` until minimax knows it, and the square just played is grey. A dashed ring is a call that waits for the values below it, the thick ring the position whose value just came back, and a thick line the move the player there takes. The block opens at the last step.
    """)
    return


@app.cell(hide_code=True)
def explore_tree_cell():
    _rows = ["X to move", "O to move", "X to move", "Game over"]

    def _listed(items):
        return ", ".join(items[:-1]) + " and " + items[-1]

    def _say(k):
        if k == 0:
            return "Step 0: minimax is called on the position at the top. Before it knows any value, it calls itself on the position after each move, down to the games that are over."
        _i = C.BACK[k - 1]
        _n = C.NODES[_i]
        _value = C.shown(_n["value"])
        if not _n["kids"]:
            _w = C.winner(_n["board"])
            if _w is None:
                return f"Step {k}: the board is full without three in a row. A draw: value {_value}."
            return f"Step {k}: {_w} has three in a row. The game is over: value {_value}."
        if len(_n["kids"]) == 1:
            return f"Step {k}: {_n['player']} has one empty square left, so this position is worth what that move is worth: {_value}."
        _values = [C.shown(C.NODES[_c]["value"]) for _c in _n["kids"]]
        _square = C.NODES[C.chosen(_i)]["square"]
        if _n["player"] == "X":
            _said = f"Step {k}: X is to move, and takes the largest of {_listed(_values)}: {_value}, with square {_square}."
        else:
            _said = f"Step {k}: O is to move, and takes the smallest of {_listed(_values)}: {_value}, with square {_square}."
        if _n["parent"] is None:
            _said += " That is the value of the position at the top: X wins."
        return _said

    def _frame(k):
        k = int(k)
        _known = len(C.step(k)["known"])
        _alt = (f"The game tree of the position of the guess, {len(C.NODES)} positions on 4 levels, after step {k}: "
                f"{_known} of their values known.")
        if k == C.STEPS:
            _alt += " The top is worth +1, with X in square 8."
        return C.tree_picture(k, _rows, _alt).text, _say(k)

    explore_tree = L.explore("tree", "step", _frame, start=C.STEPS, low=0, high=C.STEPS, step=1, role="",
                             task="Slide back to step 0, and follow the values up. Why is X in square 4 worth only 0?")
    explore_tree
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    With square 4, X makes one line of two X's, and O blocks it with square 8: O takes the smallest value, 0, and not the +1 of the game where it does not block. With square 8, X makes two such lines, and O can block only one: both of its answers are worth +1. X takes the largest of 0, 0 and +1.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Minimax, the value of a position with `player` to move:

    1. Is the game over? Then give back its value: 1 when X has three in a row, −1 when O has, 0 when the board is full.
    2. Otherwise, play each empty square, and ask minimax for the value of the position after that move, with the other player to move.
    3. With X to move, give back the largest of those values; with O to move, the smallest.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    A board is a text of 9 characters, row by row, with `.` for an empty square: `"XOXO..OX."` is the position of the guess. `_winner` looks at the 8 lines of three squares; `for a, b, c in _lines` gives the three squares of each line their own names, as `x, y = point` does. `board[:i] + player + board[i + 1:]` is the board with `player` in square `i`: the text before it, the mark, and the text after it. `"." not in board` is `True` when no empty square is left. `max(values)` gives the largest number of the list `values`, and `min(values)` the smallest. `_minimax` follows the recipe; the loop at the end shows the value of each move of X. Run the cell.
    """)
    return


@app.cell
def example_minimax():
    # the 8 lines of three squares: the rows, the columns and the two diagonals
    _lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

    def _winner(board):
        # the player with three in a row, or None
        for a, b, c in _lines:
            three = board[a] + board[b] + board[c]
            if three == "XXX":
                return "X"
            if three == "OOO":
                return "O"
        return None

    def _minimax(board, player):
        # 1. a game that is over: its value
        w = _winner(board)
        if w == "X":
            return 1
        if w == "O":
            return -1
        if "." not in board:
            return 0
        # 2. the value after each move, with the other player to move
        if player == "X":
            other = "O"
        else:
            other = "X"
        values = []
        for i in range(9):
            if board[i] == ".":
                after = board[:i] + player + board[i + 1:]
                values.append(_minimax(after, other))
        # 3. X takes the largest value, O the smallest
        if player == "X":
            return max(values)
        return min(values)

    _board = "XOXO..OX."
    for _i in range(9):
        if _board[_i] == ".":
            _after = _board[:_i] + "X" + _board[_i + 1:]
            print("X in square", _i, "is worth", _minimax(_after, "O"))
    print("the position is worth", _minimax(_board, "X"))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    X in square 4 or 5 is worth 0, and X in square 8 is worth 1: the largest, so the position is worth 1, as in the play block. For that last line, `_minimax` called itself 15 times: once for every position below the top of the game tree.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_minimax():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Your own minimax

        `minimax(board, player)` below is `_minimax` from the worked example, with `LINES` and `winner` above it. Replace the three `...`: in step 2 of the recipe the value of the position after the move, and in step 3 the value X takes and the value O takes. The check calls `minimax("XOXO..OX.", "X")`, which should give 1, and then nine other positions, with X or O to move, three of them games that are over. None has more than 6 empty squares, so the check is quick: on the empty board, minimax makes 549,946 calls.
        """),
        L.task("Make `minimax` give every position the value the player to move can force."),
    ])
    return


@app.cell(hide_code=True)
def exercise_minimax():
    box_minimax = L.code_box("minimax", lines=30)
    box_minimax
    return (box_minimax,)


@app.cell(hide_code=True)
def check_minimax(box_minimax):
    result_minimax = L.check("minimax", box_minimax.value)
    result_minimax
    return (result_minimax,)


@app.cell(hide_code=True)
def hints_minimax():
    L.hints(
        "minimax",
        [
            "Steps 2 and 3 of the recipe. Who is to move in the position `after`?",
            "The value of `after` is a question for `minimax` itself, with `other` to move. X wants the largest value, O the smallest.",
            "`values.append(minimax(___, ___))`, then `return max(___)` and `return ___(values)`.",
        ],
        why="`minimax(after, other)` is the value of the position after the move, with the other player to move: the same question one move further, until a game is over. X then takes the largest of those values, `max(values)`, and O the smallest, `min(values)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_small():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. A game tree by hand

        In this game tree, X chooses a, b or c, and then O chooses one of three answers. Each box at the bottom is a game that is over, with its value.
        """),
        C.small_tree(["X to move", "O to move", "Game over"], "A game tree with X to move at the top. Below it three positions, a, b and c, with O to move, and below each three games that are over: after a +1, +1 and −1; after b 0, +1 and 0; after c +1, −1 and 0. The values of a, b, c and the top are question marks."),
    ])
    return


@app.cell(hide_code=True)
def choice_small_cell():
    choice_small = L.choice(
        "small-tree",
        question="What is the value of the position at the top?",
        options={
            "plus": "+1",
            "right": "0",
            "minus": "−1",
        },
        correct="right",
        why={
            "right": "O takes the smallest value below each of its positions: −1 after a, 0 after b, −1 after c. X takes the largest of those: 0, with b. Below a and c lies a +1, but O never lets X get there.",
            "plus": "There is a +1 below a and below c, but there O chooses, and O takes the smallest value. Work out the value of a, b and c first, as in step 3 of the recipe.",
            "minus": "−1 is what O takes after a and after c. But X chooses first, and takes the move with the largest value: b, where O can do no better than 0.",
        },
    )
    choice_small
    return (choice_small,)


@app.cell(hide_code=True)
def text_exercise_chess():
    mo.md(r"""
    ### Exercise 3. And chess?
    """)
    return


@app.cell(hide_code=True)
def choice_chess_cell():
    choice_chess = L.choice(
        "chess",
        question="On the empty board, minimax gives 0 after 549,946 calls: with best play, tic-tac-toe is a draw. Why does a chess program not work out the value of a chess position in the same way?",
        options={
            "outcomes": "A game of chess has more outcomes than a win, a draw and a loss",
            "right": "Its game tree is far too large: it looks only a few moves ahead, and estimates how good the positions there are",
            "board": "Minimax only works on a board of 3 by 3 squares",
        },
        correct="right",
        why={
            "right": "In a chess position, a player has about 35 moves, so 10 moves ahead, the game tree already has $35^{10}$ positions at the bottom, nearly 3 million billion. A chess program stops after a few moves, gives each position there a number from an *evaluation function*, an estimate of who is ahead, and does minimax on those numbers.",
            "outcomes": "A game of chess also ends in a win, a draw or a loss, so +1, 0 and −1 would do. The trouble is the size of its game tree.",
            "board": "Minimax only needs the moves in each position and the value of a game that is over; chess has those too. The trouble is the size of its game tree.",
        },
    )
    choice_chess
    return (choice_chess,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** Minimax searches a game tree as [breadth-first search](/course/open/search-bfs?lang=en) searched a maze, but in another order: it follows each move down to the end of the game before it tries the next, as in the play block. It gives back a value instead of a path, and it calls itself, as in [A function that calls itself](/course/open/py-recursion?lang=en). With a value for every position, the best move is the one whose value is best for the player to move. Search in games such as chess comes back in a branch still to come, *Search, logic, planning and optimization*. In the next lesson, *When steps can slip: value iteration*, no opponent answers your action, but chance does, and every state gets a value again.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_chess, choice_small, result_minimax):
    L.wrap_up(result_minimax, choice_small, choice_chess)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
