# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson search-minimax: the positions of the lesson, the course's own minimax, which draws the play
block, the small tree of exercise 2; and minimax, the recursion the learner completes."""
import re

from everything_ai import draw
from everything_ai.check import FunctionBox, Mistake

# The 8 lines of three squares: the rows, the columns and the two diagonals. A board is a text of 9 signs, row by
# row, X, O or "." for an empty square; square 0 is top left, square 8 bottom right.
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
GAME = "XOXO..OX."   # the position of the guess, the play block and the worked example: X to move
EMPTY_CALLS = 549946   # the calls minimax makes on the empty board, one per position of its game tree, as exercises 1 and 3 say


def winner(board: str):
    """The player with three in a row, or None. As in the lesson."""
    for a, b, c in LINES:
        three = board[a] + board[b] + board[c]
        if three == "XXX":
            return "X"
        if three == "OOO":
            return "O"
    return None


def other(player: str) -> str:
    return "O" if player == "X" else "X"


def moves(board: str) -> list[int]:
    """The empty squares, where the player to move can put a mark."""
    return [i for i in range(9) if board[i] == "."]


def play(board: str, square: int, player: str) -> str:
    return board[:square] + player + board[square + 1:]


def end_value(board: str):
    """The value of a finished game, +1, -1 or 0; None while the game goes on."""
    w = winner(board)
    if w is not None:
        return 1 if w == "X" else -1
    return None if "." in board else 0


def minimax(board: str, player: str) -> int:
    """The course's own minimax, as in the lesson: the value the player to move can force."""
    done = end_value(board)
    if done is not None:
        return done
    values = [minimax(play(board, i, player), other(player)) for i in moves(board)]
    return max(values) if player == "X" else min(values)


def grow(board: str, player: str) -> tuple[list[dict], list[int]]:
    """The game tree below a position, as minimax walks it: the nodes in the order the calls start, each with its
    board, the player to move, the square of the move that led to it, its parent and its value; and the order in
    which the values come back."""
    nodes, back = [], []

    def call(board, player, parent, square):
        i = len(nodes)
        nodes.append({"board": board, "player": player, "parent": parent, "square": square, "kids": []})
        if parent is not None:
            nodes[parent]["kids"].append(i)
        value = end_value(board)
        if value is None:
            for m in moves(board):
                call(play(board, m, player), other(player), i, m)
            values = [nodes[k]["value"] for k in nodes[i]["kids"]]
            value = max(values) if player == "X" else min(values)
        nodes[i]["value"] = value
        back.append(i)

    call(board, player, None, None)
    return nodes, back


NODES, BACK = grow(GAME, "X")
STEPS = len(BACK)   # the play block has a stop for every value that comes back, and step 0


def shown(value: int) -> str:
    """A value as the lesson writes it: +1, 0 or −1."""
    return {1: "+1", 0: "0", -1: "−1"}[value]


def chosen(i: int) -> int | None:
    """The child of node i whose value the player there takes: the first with the largest value for X, the smallest
    for O, as max and min pick it."""
    kids = NODES[i]["kids"]
    for k in kids:
        if NODES[k]["value"] == NODES[i]["value"]:
            return k
    return None


def ancestors(i: int) -> list[int]:
    out = []
    while NODES[i]["parent"] is not None:
        i = NODES[i]["parent"]
        out.append(i)
    return out


def step(k: int) -> dict:
    """The play block after k values came back: the node whose value came back last (None at step 0), what is known,
    and the calls that wait for the values below them."""
    known = set(BACK[:k])
    mark = BACK[k - 1] if k > 0 else None
    wait = [0] if k == 0 else ancestors(mark)
    return {"known": known, "mark": mark, "wait": wait}


def tree_picture(k: int, rows: list[str], alt: str):
    """The game tree of GAME after step k of the play block."""
    s = step(k)
    nodes = []
    for i, n in enumerate(NODES):
        p = n["parent"]
        ring = "mark" if i == s["mark"] else "wait" if i in s["wait"] else None
        nodes.append({"parent": p, "board": n["board"], "new": n["square"],
                      "value": shown(n["value"]) if i in s["known"] else "?", "ring": ring,
                      "chosen": p is not None and p in s["known"] and chosen(p) == i})
    return draw.game_tree(nodes, rows, alt=alt, cell=15, slot=64)


def squares_and_game(titles: tuple[str, str], alt: str):
    """Two boards: the number of every square, and GAME with the numbers of its empty squares."""
    return draw.boards([{"board": "." * 9, "title": titles[0], "index": True},
                        {"board": GAME, "title": titles[1], "index": True}], alt=alt, cell=30)


# --- exercise 2: a small tree, worked out by hand -------------------------------------------------------------------

SMALL = {"a": [1, 1, -1], "b": [0, 1, 0], "c": [1, -1, 0]}   # the values of the finished games after each move of X
SMALL_O = {move: min(values) for move, values in SMALL.items()}   # what O takes after each move
SMALL_TOP = max(SMALL_O.values())


def small_tree(rows: list[str], alt: str):
    """The tree of exercise 2: X chooses a, b or c, then O one of three answers, each a finished game with its value;
    the values above them are still to work out."""
    nodes = [{"parent": None, "text": "?"}]
    for move, values in SMALL.items():
        nodes.append({"parent": 0, "text": "?", "name": move})
        here = len(nodes) - 1
        nodes += [{"parent": here, "text": shown(v)} for v in values]
    return draw.game_tree(nodes, rows, alt=alt, cell=12, slot=42)


# --- exercise 1: minimax --------------------------------------------------------------------------------------------

# The lesson is done when these are correct. "small-tree" and "chess" are the choice questions in the lesson.
REQUIRED = ["minimax", "small-tree", "chess"]

# a `...` of the box, not the dots of a board such as "XOX.O....", which a learner may copy from a check message
_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*?(?<![\w.'\"])\.\.\.(?![\w.'\"])", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_HEAD = ("LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]\n\n\n"
         "def winner(board):\n"
         "    for a, b, c in LINES:\n"
         "        three = board[a] + board[b] + board[c]\n"
         "        if three == \"XXX\":\n"
         "            return \"X\"\n"
         "        if three == \"OOO\":\n"
         "            return \"O\"\n"
         "    return None\n\n\n")


def _m(value: str = "values.append(minimax(after, other))", x: str = "return max(values)",
       o: str = "return min(values)") -> str:
    """minimax as the box holds it, with these three lines: the value after a move in step 2 of the recipe, and the
    value X and O each take in step 3."""
    return (_HEAD +
            "def minimax(board, player):\n"
            "    w = winner(board)\n"
            "    if w == \"X\":\n"
            "        return 1\n"
            "    if w == \"O\":\n"
            "        return -1\n"
            "    if \".\" not in board:\n"
            "        return 0\n"
            "    if player == \"X\":\n"
            "        other = \"O\"\n"
            "    else:\n"
            "        other = \"X\"\n"
            "    values = []\n"
            "    for i in range(9):\n"
            "        if board[i] == \".\":\n"
            "            after = board[:i] + player + board[i + 1:]\n"
            f"            {value}\n"
            "    if player == \"X\":\n"
            f"        {x}\n"
            f"    {o}\n")


_M_START = _m("values.append(...)", "return ...", "return ...")
_M = _m()

EXERCISES = {
    "minimax": FunctionBox(
        name="minimax",
        function="minimax",
        start=_M_START,
        solution=_M,
        visible=[(GAME, "X")],
        hidden=[("XOX.O....", "X"), ("XO..X....", "O"), ("XOXO.X...", "O"), ("XXO..O...", "X"), ("XOXOXXO..", "O"),
                ("XX.OO....", "X"), ("XXXOO....", "O"), ("OOOXX.X..", "X"), ("XOXXOOOXX", "O")],
        first=[_DOTS],
        mistakes=[
            Mistake("same-player", wrong=_m("values.append(minimax(after, player))"),
                    en="`{call}` gives {got}, and it should give {want}. After a move, the other player is to move, but your call passes `player` on, so the same player moves again and again. Pass `other`: `minimax(after, other)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Na een zet is de andere speler aan zet, maar je aanroep geeft `player` door, dus dezelfde speler zet steeds opnieuw. Geef `other` door: `minimax(after, other)`."),
            Mistake("always-o", wrong=_m("values.append(minimax(after, \"O\"))"),
                    en="`{call}` gives {got}, and it should give {want}. Your call always passes `\"O\"` on, but after a move of O, X is to move. Pass `other`, the player who is not to move now: `minimax(after, other)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je aanroep geeft altijd `\"O\"` door, maar na een zet van O is X aan zet. Geef `other` door, de speler die nu niet aan zet is: `minimax(after, other)`."),
            Mistake("always-x", wrong=_m("values.append(minimax(after, \"X\"))"),
                    en="`{call}` gives {got}, and it should give {want}. Your call always passes `\"X\"` on, but after a move of X, O is to move. Pass `other`, the player who is not to move now: `minimax(after, other)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je aanroep geeft altijd `\"X\"` door, maar na een zet van X is O aan zet. Geef `other` door, de speler die nu niet aan zet is: `minimax(after, other)`."),
            Mistake("max-for-both", wrong=_m(o="return max(values)"),
                    en="`{call}` gives {got}, and it should give {want}. Your O takes the largest value too, so it plays as if it wants X to win. O wants the smallest value: `return min(values)` on the last line.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Bij jou neemt O ook de grootste waarde, dus O speelt alsof hij X wil laten winnen. O wil de kleinste waarde: `return min(values)` op de laatste regel."),
            Mistake("min-for-both", wrong=_m(x="return min(values)"),
                    en="`{call}` gives {got}, and it should give {want}. Your X takes the smallest value too, so it plays as if it wants O to win. X wants the largest value: `return max(values)` under `if player == \"X\":`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Bij jou neemt X ook de kleinste waarde, dus X speelt alsof hij O wil laten winnen. X wil de grootste waarde: `return max(values)` onder `if player == \"X\":`."),
            Mistake("swapped", wrong=_m(x="return min(values)", o="return max(values)"),
                    en="`{call}` gives {got}, and it should give {want}. Your X takes the smallest value and your O the largest: each plays for the other. A win of X is +1, so X takes `max(values)` and O `min(values)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Bij jou neemt X de kleinste waarde en O de grootste: elk speelt voor de ander. Een winst van X is +1, dus X neemt `max(values)` en O `min(values)`."),
            Mistake("list-back", matches=lambda got, text: isinstance(got, list),
                    en="`{call}` gives {got}: the whole list of values. Give back one of them: the largest when X is to move, `max(values)`, and the smallest when O is, `min(values)`.",
                    nl="`{call}` geeft {got}: de hele lijst waarden. Geef er één terug: de grootste als X aan zet is, `max(values)`, en de kleinste als O aan zet is, `min(values)`."),
            Mistake("board-not-value", matches=lambda got, text: isinstance(got, str),
                    en="`{call}` gives {got}, a board, where it should give a number, {want}. Your list holds the boards after each move, not their values. Put the value of `after` in it: `values.append(minimax(after, other))`.",
                    nl="`{call}` geeft {got}, een bord, waar het een getal moet geven, {want}. Je lijst bevat de borden na elke zet, niet hun waarden. Zet de waarde van `after` erin: `values.append(minimax(after, other))`."),
            Mistake("never-ends", error=lambda e: isinstance(e, RecursionError),
                    en="With `{call}`, your function keeps calling itself and never gets to a finished game: Python gives up after about a thousand calls inside each other. Each call must get a board with one mark more, `after`: `minimax(after, other)`.",
                    nl="Met `{call}` roept je functie zichzelf steeds weer aan en komt hij nooit bij een afgelopen spel: Python geeft het op na zo'n duizend aanroepen in elkaar. Elke aanroep moet een bord met één teken meer krijgen, `after`: `minimax(after, other)`."),
            Mistake("one-argument", error=lambda e: isinstance(e, TypeError) and "missing 1 required positional argument" in str(e),
                    en="With `{call}`, your function stops with an error: a call of `minimax` gets only the board, and `minimax` needs two things, the board and the player to move. Pass both: `minimax(after, other)`.",
                    nl="Met `{call}` stopt je functie met een fout: een aanroep van `minimax` krijgt alleen het bord, en `minimax` heeft er twee nodig, het bord en de speler die aan zet is. Geef ze allebei door: `minimax(after, other)`."),
            Mistake("nothing-in-values", error=lambda e: isinstance(e, ValueError) and "empty" in str(e),
                    en="With `{call}`, your function stops with an error: `values` is still an empty list when `max` or `min` looks for its largest or smallest number. Does step 2 put the value after each move in it, with `values.append(minimax(after, other))`? And is the base case of a full board still there, `if \".\" not in board:` with `return 0` under it?",
                    nl="Met `{call}` stopt je functie met een fout: `values` is nog een lege lijst als `max` of `min` er het grootste of kleinste getal in zoekt. Zet stap 2 de waarde na elke zet erin, met `values.append(minimax(after, other))`? En staat het basisgeval van een vol bord er nog, `if \".\" not in board:` met `return 0` eronder?"),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "empty": {"en": "Replace the three `...`: in step 2 of the recipe the value of the position after the move, and in step 3 the value X takes and the value O takes.",
                      "nl": "Vervang de drie `...`: in stap 2 van het recept de waarde van de positie na de zet, en in stap 3 de waarde die X neemt en de waarde die O neemt."},
            "arguments": {"en": "The check calls `{call}`, with a board and the player to move, and the `def` line of your function has another number of names between its parentheses. Keep `(board, player)`.",
                          "nl": "De check roept `{call}` aan, met een bord en de speler die aan zet is, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(board, player)`."},
            "slow": {"en": "Your minimax is still not done after 2 seconds, on a board with at most 6 empty squares. Does each call get a board with one mark more, `after`, so that every game ends?",
                     "nl": "Je minimax is na 2 seconden nog niet klaar, op een bord met hooguit 6 lege vakjes. Krijgt elke aanroep een bord met één teken meer, `after`, zodat elk spel afloopt?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Compare your three lines with steps 2 and 3 of the recipe.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Vergelijk je drie regels met stap 2 en 3 van het recept."},
            "almost": {"en": "Right for the position of the guess. But `{call}` gives {got}, and it should give {want}. Does your minimax work for both players, and for a game that is over already?",
                       "nl": "Goed voor de positie van de gok. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je minimax voor beide spelers, en voor een spel dat al afgelopen is?"},
            "correct": {"en": "Your minimax is right in all {n} positions, with X and with O to move, and for games that are over already. Each call asks the same question one move further, and X takes the largest value, O the smallest.",
                        "nl": "Je minimax klopt in alle {n} posities, met X en met O aan zet, en voor spellen die al afgelopen zijn. Elke aanroep stelt dezelfde vraag één zet verder, en X neemt de grootste waarde, O de kleinste."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("minimax", _M_START, "empty", None),
    ("minimax", _m(x="return ..."), "wrong", "dots-left"),
    ("minimax", _m("values.append(minimax(after, player))"), "wrong", "same-player"),
    ("minimax", _m("values.append(minimax(after, \"O\"))"), "wrong", "always-o"),
    ("minimax", _m("values.append(minimax(after, \"X\"))"), "wrong", "always-x"),
    ("minimax", _m(o="return max(values)"), "wrong", "max-for-both"),
    ("minimax", _m(x="return min(values)"), "wrong", "min-for-both"),
    ("minimax", _m(x="return min(values)", o="return max(values)"), "wrong", "swapped"),
    ("minimax", _m(x="return values", o="return values"), "wrong", "list-back"),
    ("minimax", _m("values.append(after)"), "wrong", "board-not-value"),
    ("minimax", _m("values.append(minimax(board, other))"), "error", "never-ends"),
    ("minimax", _m("values.append(minimax(after))"), "error", "one-argument"),
    ("minimax", _m("minimax(after, other)"), "error", "nothing-in-values"),
    ("minimax", _M.replace("    if \".\" not in board:\n        return 0\n", ""), "error", "nothing-in-values"),
    # the value of the first move: right by luck for the position of the guess, not for the others
    ("minimax", _m(x="return values[0]", o="return values[0]"), "almost", None),
    # an error in another line than the three: told in plain words
    ("minimax", _m("values.append(minimax(after, others))"), "error", None),
    # other right ways: the call written out, O first, and the value kept under a name of its own
    ("minimax", _m("values.append(minimax(board[:i] + player + board[i + 1:], other))"), "correct", None),
    ("minimax", _m(x="return max(values)", o="return min(values)").replace(
        "    if player == \"X\":\n        return max(values)\n    return min(values)\n",
        "    if player == \"O\":\n        return min(values)\n    return max(values)\n"), "correct", None),
    ("minimax", _m("value = minimax(after, other)\n            values.append(value)"), "correct", None),
    # a call copied from a check message to try it: the dots of its board are not a `...` left in the box
    ("minimax", _M + "\n\nprint(minimax(\"XOX.O....\", \"X\"))\n", "correct", None),
]
