# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson search-bfs: the mazes of the lesson and the course's own breadth-first search, which draws the
play block one step at a time; bfs, the search the learner completes; and which part of a search problem is which."""
import re
from collections import deque
from dataclasses import dataclass, field

from everything_ai import draw
from everything_ai.check import FunctionBox, Mistake, Sort


class Maze(list):
    """A maze as a list of strings, one per row: '#' is a wall, S the start and G the goal. A check message calls it
    by the name the lesson gives it, such as SMALL, and the learner's code gets a plain copy."""

    def __init__(self, rows, name: str):
        super().__init__(rows)
        self.name = name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return list(self)


MAZE = Maze([
    "###############",
    "#.......#.....#",
    "#.###...#..#..#",
    "#...#......#..#",
    "#S..#..#####..#",
    "#...#.......#G#",
    "#......#......#",
    "###############",
], "MAZE")
SMALL = Maze([        # the maze of the worked example
    "#######",
    "#S..#.#",
    "#.#...#",
    "#...#G#",
    "#######",
], "SMALL")
CLOSED = Maze([       # no way leads to G
    "#######",
    "#S....#",
    "#.##..#",
    "#..####",
    "#..#G.#",
    "#######",
], "CLOSED")


def find(maze, sign: str) -> tuple[int, int]:
    """The cell (row, column) with this sign."""
    for r, row in enumerate(maze):
        if sign in row:
            return (r, row.index(sign))
    raise ValueError(sign)


START, GOAL = find(MAZE, "S"), find(MAZE, "G")


def neighbors(maze, state):
    """The states one action away: a step up, down, left or right, but not into a wall. As in the lesson."""
    r, c = state
    result = []
    for nxt in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
        if maze[nxt[0]][nxt[1]] != "#":
            result.append(nxt)
    return result


def bfs(maze, start, goal):
    """The course's own breadth-first search, as in the lesson: the path and the set of expanded states."""
    frontier = deque([[start]])
    done = set()
    while len(frontier) > 0:
        path = frontier.popleft()
        state = path[-1]
        if state == goal:
            return path, done
        if state not in done:
            done.add(state)
            for nxt in neighbors(maze, state):
                frontier.append(path + [nxt])
    return None, done


def distances(maze, start) -> dict:
    """The number of steps of a shortest path from the start to every state it can reach."""
    far, queue = {start: 0}, deque([start])
    while queue:
        state = queue.popleft()
        for nxt in neighbors(maze, state):
            if nxt not in far:
                far[nxt] = far[state] + 1
                queue.append(nxt)
    return far


def open_cells(maze) -> int:
    return sum(len(row) - row.count("#") for row in maze)


def trace(maze, start, goal) -> list[dict]:
    """What breadth-first search has done after each step: the states it expanded with their number of steps from the
    start, the states on its frontier, the state it expanded last; at the last step, the path it found. For the play
    block, one picture per step."""
    steps = [{"done": {}, "frontier": {start: 0}, "mark": None, "path": None}]
    frontier, done = deque([[start]]), {}
    while frontier:
        path = frontier.popleft()
        state = path[-1]
        if state == goal:
            steps.append({"done": dict(done), "frontier": steps[-1]["frontier"], "mark": state, "path": path})
            return steps
        if state not in done:
            done[state] = len(path) - 1
            for nxt in neighbors(maze, state):
                frontier.append(path + [nxt])
            front = {}
            for p in frontier:
                if p[-1] not in done and p[-1] not in front:
                    front[p[-1]] = len(p) - 1
            steps.append({"done": dict(done), "frontier": front, "mark": state, "path": None})
    return steps


def newest_first(maze, start, goal):
    """The same search, but taking the newest path from the back of the queue: depth-first search, for exercise 3."""
    frontier = deque([[start]])
    done = set()
    while len(frontier) > 0:
        path = frontier.pop()
        state = path[-1]
        if state == goal:
            return path, done
        if state not in done:
            done.add(state)
            for nxt in neighbors(maze, state):
                frontier.append(path + [nxt])
    return None, done


STEPS = trace(MAZE, START, GOAL)
SHORTEST, DONE = bfs(MAZE, START, GOAL)
DEPTH_FIRST, _ = newest_first(MAZE, START, GOAL)


def picture(step: int, alt: str):
    """The maze of the lesson after `step` steps of breadth-first search, for the play block."""
    s = STEPS[step]
    return draw.maze_grids([{"rows": MAZE, "done": s["done"], "frontier": s["frontier"], "numbers": s["done"], "role": "loss",
                             "mark": s["mark"], "path": s["path"], "index": True}], alt=alt, cell=28)


def mazes(alt: str):
    """The two other mazes the check gives bfs, with their names."""
    return draw.maze_grids([{"rows": SMALL, "title": "SMALL", "index": True}, {"rows": CLOSED, "title": "CLOSED", "index": True}],
                           alt=alt, cell=26)


# --- how a search's answer is judged ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Found:
    """What a search gave back, as far as the check is concerned: not the exact path, since every shortest path is
    right, but its kind and number of steps, and whether the search expanded every state at least 2 steps closer to the
    start than the goal. Breadth-first search does so before it finds the goal, both when it tests for the goal as it
    takes a path out, as in the lesson, and when it tests as it puts a path in, as AIMA does."""
    kind: str                 # "path"; "none": None, no path; "bad": not a way from the start to the goal; "shape"
    steps: int | None = None  # the path cost of the path: its number of steps
    ok: bool | None = None    # it expanded what breadth-first search expands; without a way, every state it can reach
    n: int = field(default=0, compare=False)   # how many states it expanded


def same_found(a, b) -> bool:
    """Right when kind, steps and what was expanded agree; the number of expanded states only counts when the search
    is not right, to tell one known mistake from another."""
    if not (isinstance(a, Found) and isinstance(b, Found)):
        return a == b
    return a == b and (a.ok is True or a.n == b.n)


def is_way(maze, start, goal, path) -> bool:
    """The path goes from the start to the goal, one action at a time, and never into a wall."""
    if not isinstance(path, (list, tuple)) or not path or path[0] != start or path[-1] != goal:
        return False
    for a, b in zip(path, path[1:]):
        if not (isinstance(b, tuple) and len(b) == 2 and all(isinstance(x, int) for x in b)):
            return False
        if not (0 <= b[0] < len(maze) and 0 <= b[1] < len(maze[0])) or maze[b[0]][b[1]] == "#":
            return False
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return False
    return True


def judge(maze, start, goal, value):
    """What `bfs(maze, start, goal)` gave back, judged: None stays None, so the check can say that a return is missing."""
    if value is None:
        return None
    if not (isinstance(value, tuple) and len(value) == 2):
        return Found("shape")
    path, done = value
    try:
        done = set(done)
    except TypeError:
        return Found("shape")
    far = distances(maze, start)
    if path is None:
        return Found("none", ok=done == set(far), n=len(done))
    if not is_way(maze, start, goal, path):
        return Found("bad", n=len(done))
    goal_steps = far[goal]
    return Found("path", len(path) - 1, all(s in done for s, d in far.items() if d < goal_steps - 1), len(done))


def _count(n: int, one: str, more: str) -> str:
    return f"{n} {one if n == 1 else more}"


def show(found) -> dict:
    """A judged answer in words, for the check's messages."""
    if not isinstance(found, Found) or found.kind == "shape":
        return {"en": "something other than two values, a path and `done`",
                "nl": "iets anders dan twee waarden, een pad en `done`"}
    if found.kind == "bad":
        return {"en": "a path that is not a way from the start to the goal, one action at a time",
                "nl": "een pad dat geen weg van de start naar het doel is, één action per stap"}
    expanded = {"en": "(" + _count(found.n, "state", "states") + " expanded)",
                "nl": "(" + _count(found.n, "state", "states") + " uitgebreid)"}
    if found.kind == "none":
        return {"en": f"`None`, no path {expanded['en']}", "nl": f"`None`, geen pad {expanded['nl']}"}
    return {"en": f"a path of {_count(found.steps, 'step', 'steps')} {expanded['en']}",
            "nl": f"een pad van {_count(found.steps, 'stap', 'stappen')} {expanded['nl']}"}


@dataclass
class SearchBox(FunctionBox):
    """A search in a code box. The check calls it on each maze and judges what it gives back (judge), so any shortest
    path is right, and a known mistake is known by what it does in every maze."""
    judge: object = None

    def _run(self, fn, case):
        return self.judge(*case, super()._run(fn, case))


# The lesson is done when these are correct. "newest" is the choice question in the lesson.
REQUIRED = ["bfs", "formulation", "newest"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# --- exercise 1: bfs ------------------------------------------------------------------------------------------------

_NEIGHBORS = ("def neighbors(maze, state):\n"
               "    r, c = state\n"
               "    result = []\n"
               "    for nxt in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:\n"
               "        if maze[nxt[0]][nxt[1]] != \"#\":\n"
               "            result.append(nxt)\n"
               "    return result\n")


def _b(take: str = "path = frontier.popleft()", put: str = "frontier.append(path + [nxt])") -> str:
    """bfs as the box holds it, with these two lines: step 2 and step 4 of the recipe."""
    put = put.replace("\n", "\n                ")
    return ("from collections import deque\n\n\n" + _NEIGHBORS + "\n\n"
            "def bfs(maze, start, goal):\n"
            "    frontier = deque([[start]])\n"
            "    done = set()\n"
            "    while len(frontier) > 0:\n"
            f"        {take}\n"
            "        state = path[-1]\n"
            "        if state == goal:\n"
            "            return path, done\n"
            "        if state not in done:\n"
            "            done.add(state)\n"
            "            for nxt in neighbors(maze, state):\n"
            f"                {put}\n"
            "    return None, done\n")


_B_START = _b("path = ...", "...")
_B = _b()

EXERCISES = {
    "bfs": SearchBox(
        name="bfs",
        function="bfs",
        start=_B_START,
        solution=_B,
        judge=judge,
        compare=same_found,
        show=show,
        visible=[(MAZE, START, GOAL)],
        hidden=[(SMALL, (1, 1), (3, 5)), (CLOSED, (1, 1), (4, 4)), (SMALL, (1, 1), (1, 1)), (MAZE, GOAL, START)],
        first=[_DOTS],
        mistakes=[
            Mistake("depth-first", wrong=_b("path = frontier.pop()"),
                    en="`{call}` gives {got}, and it should give {want}. Your search takes the newest path first, so it follows one way as far as it goes. It finds a way, but not always a shortest one. Put every path at the back with `append`, and take the oldest from the front with `popleft()`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je search pakt het nieuwste pad eerst, dus hij volgt één weg zo ver als hij gaat. Hij vindt een weg, maar niet altijd een kortste. Zet elk pad achteraan met `append`, en pak het oudste van voren met `popleft()`."),
            Mistake("only-next", wrong=_b(put="frontier.append([nxt])"),
                    en="`{call}` gives {got}. You put a path with only the next state in the queue, so the way back to the start is lost. Put the whole path in it, one action longer: `path + [nxt]`.",
                    nl="`{call}` geeft {got}. Je zet een pad met alleen de volgende state in de wachtrij, dus de weg terug naar de start raakt kwijt. Zet het hele pad erin, één action langer: `path + [nxt]`."),
            Mistake("shared-path", wrong=_b(put="path.append(nxt)\nfrontier.append(path)"),
                    en="`{call}` gives {got}, and it should give {want}. `path.append(nxt)` changes the path itself, so every path you put in the queue is one and the same list, with all the neighbors on it. Make a new list instead: `path + [nxt]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. `path.append(nxt)` verandert het pad zelf, dus elk pad dat je in de wachtrij zet, is één en dezelfde lijst, met alle buren erop. Maak in plaats daarvan een nieuwe lijst: `path + [nxt]`."),
            Mistake("two-values", matches=lambda got, text: isinstance(got, Found) and got.kind == "shape",
                    en="`{call}` gives {got}. Keep the lines `return path, done` and `return None, done`: the check needs both, the path and the set of states the search expanded.",
                    nl="`{call}` geeft {got}. Houd de regels `return path, done` en `return None, done`: de check heeft ze allebei nodig, het pad en de set states die de search uitbreidde."),
            Mistake("plus-tuple", error=lambda e: isinstance(e, TypeError) and "concatenate list" in str(e) and "tuple" in str(e),
                    en="With `{call}`, your function stops with an error: `path + nxt` glues a tuple to a list, and `+` only glues a list to a list. Put `nxt` in a list of its own: `path + [nxt]`.",
                    nl="Met `{call}` stopt je functie met een fout: `path + nxt` plakt een tuple aan een lijst, en `+` plakt alleen een lijst aan een lijst. Zet `nxt` in een eigen lijst: `path + [nxt]`."),
            Mistake("pop-number", error=lambda e: isinstance(e, TypeError) and "pop()" in str(e) and "argument" in str(e),
                    en="With `{call}`, your function stops with an error: the `pop()` of a `deque` takes no number. A `deque` takes the oldest path, from the front, with `popleft()`.",
                    nl="Met `{call}` stopt je functie met een fout: de `pop()` van een `deque` krijgt geen getal. Een `deque` pakt het oudste pad, van voren, met `popleft()`."),
            Mistake("state-not-path", error=lambda e: isinstance(e, TypeError) and "unpack" in str(e),
                    en="With `{call}`, your function stops with an error: the queue holds a state where it should hold a path, so `path[-1]` is a number instead of a state. Put the whole path in it, one action longer: `path + [nxt]`.",
                    nl="Met `{call}` stopt je functie met een fout: in de wachtrij staat een state waar een pad hoort, dus `path[-1]` is een getal in plaats van een state. Zet het hele pad erin, één action langer: `path + [nxt]`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "empty": {"en": "Replace the two `...`: step 2 of the recipe, which takes a path out of the queue, and step 4, which puts a longer one in.",
                      "nl": "Vervang de twee `...`: stap 2 van het recept, die een pad uit de wachtrij pakt, en stap 4, die er een langer pad in zet."},
            "arguments": {"en": "The check calls `{call}`, with a maze, a start and a goal, and the `def` line of your function has another number of names between its parentheses. Keep `(maze, start, goal)`.",
                          "nl": "De check roept `{call}` aan, met een doolhof, een start en een doel, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(maze, start, goal)`."},
            "slow": {"en": "Your search is still not done after 2 seconds. Does every round of the `while` loop take a path out of the queue? `frontier.popleft()` takes it out; `frontier[0]` only looks at it.",
                     "nl": "Je search is na 2 seconden nog niet klaar. Pakt elke ronde van de `while`-loop een pad uit de wachtrij? `frontier.popleft()` haalt het eruit; `frontier[0]` kijkt er alleen naar."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Compare your two lines with steps 2 and 4 of the recipe.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Vergelijk je twee regels met stap 2 en 4 van het recept."},
            "almost": {"en": "Right for MAZE. But `{call}` gives {got}, and it should give {want}. Does your search work in every maze, even when no way leads to the goal, or when the start is the goal?",
                       "nl": "Goed voor MAZE. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je search in elk doolhof, ook als er geen weg naar het doel is, of als de start het doel is?"},
            "correct": {"en": "Your search is right in all {n} cases, with and without a way to the goal, and with the start as the goal.",
                        "nl": "Je search klopt in alle {n} gevallen, met en zonder weg naar het doel, en met de start als doel."},
        },
    ),
    "formulation": Sort(
        name="formulation",
        groups={
            "state": {"en": "State", "nl": "State"},
            "action": {"en": "Action", "nl": "Action"},
            "goal": {"en": "Goal test", "nl": "Goal test"},
            "cost": {"en": "Path cost", "nl": "Path cost"},
        },
        items={
            "slide": {"en": "Slide a tile into the gap", "nl": "Een stukje in het gat schuiven"},
            "address": {"en": "Is this the address you want to go to?", "nl": "Is dit het adres waar je heen wilt?"},
            "crossroads": {"en": "A crossroads in the town", "nl": "Een kruispunt in de stad"},
            "slides": {"en": "How many times you slid a tile", "nl": "Hoe vaak je een stukje schoof"},
            "street": {"en": "Drive one street, to the next crossroads", "nl": "Eén straat rijden, naar het volgende kruispunt"},
            "order": {"en": "Are the tiles 1 to 8 in order?", "nl": "Liggen de stukjes 1 tot en met 8 op volgorde?"},
            "minutes": {"en": "The minutes you drove", "nl": "De minuten die je reed"},
            "tiles": {"en": "Where each of the eight tiles lies, and the gap", "nl": "Waar elk van de acht stukjes ligt, en het gat"},
        },
        solution={"slide": "action", "address": "goal", "crossroads": "state", "slides": "cost",
                  "street": "action", "order": "goal", "minutes": "cost", "tiles": "state"},
        nudge={
            "crossroads": {"en": "Is this where the car can be, or what it does to get somewhere else?",
                           "nl": "Is dit waar de auto kan zijn, of wat hij doet om ergens anders te komen?"},
            "tiles": {"en": "Is this one situation of the puzzle, or a move?",
                      "nl": "Is dit één situatie van de puzzel, of een zet?"},
            "street": {"en": "Is this where you are, or what takes you from one place to the next?",
                       "nl": "Is dit waar je bent, of wat je van de ene plek naar de volgende brengt?"},
            "slide": {"en": "Is this a situation of the puzzle, or what changes it?",
                      "nl": "Is dit een situatie van de puzzel, of wat hem verandert?"},
            "address": {"en": "Is this a question about one state, or something you add up along the way?",
                        "nl": "Is dit een vraag over één state, of iets wat je onderweg optelt?"},
            "order": {"en": "Is this a question about one state, or a move?",
                      "nl": "Is dit een vraag over één state, of een zet?"},
            "minutes": {"en": "Is this a question about one state, or something you add up along the whole path?",
                        "nl": "Is dit een vraag over één state, of iets wat je over het hele pad optelt?"},
            "slides": {"en": "Is this a question about one state, or something you count along the whole path?",
                       "nl": "Is dit een vraag over één state, of iets wat je over het hele pad telt?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the parts again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de onderdelen opnieuw uit."},
            "empty": {"en": "Put each part of the two tasks in a group: state, action, goal test or path cost.",
                      "nl": "Zet elk onderdeel van de twee taken in een groep: state, action, goal test of path cost."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "{item} belongs in another group.",
                      "nl": "{item} hoort in een andere groep."},
            "correct": {"en": "All {n} are right. Both tasks are search problems: states, the actions between them, a goal test and a path cost, so the same search can solve both.",
                        "nl": "Alle {n} kloppen. Beide taken zijn search problems: states, de actions ertussen, een goal test en een path cost, dus dezelfde search kan ze allebei oplossen."},
        },
    ),
}

_SORT = EXERCISES["formulation"].solution

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("bfs", _B_START, "empty", None),
    ("bfs", _b("path = ..."), "wrong", "dots-left"),
    ("bfs", _b("path = frontier.pop()"), "wrong", "depth-first"),
    ("bfs", _b(put="frontier.append([nxt])"), "wrong", "only-next"),
    ("bfs", _b(put="path.append(nxt)\nfrontier.append(path)"), "wrong", "shared-path"),
    ("bfs", _b(put="frontier.append(nxt)"), "error", "state-not-path"),
    ("bfs", _b(put="frontier.appendleft(path + [nxt])"), "wrong", "depth-first"),
    ("bfs", _b(put="frontier.append(path + nxt)"), "error", "plus-tuple"),
    ("bfs", _b("path = frontier.pop(0)"), "error", "pop-number"),
    ("bfs", _b("path = frontier[0]"), "slow", None),
    ("bfs", _b(put="new = path + [nxt]\nfrontier.append(new)"), "correct", None),
    # another right search: it marks a state when the path to it goes into the queue, as many books do
    ("bfs", "from collections import deque\n\n\n" + _NEIGHBORS + "\n\n"
            "def bfs(maze, start, goal):\n    frontier = deque([[start]])\n    done = set([start])\n"
            "    while len(frontier) > 0:\n        path = frontier.popleft()\n        if path[-1] == goal:\n            return path, done\n"
            "        for nxt in neighbors(maze, path[-1]):\n            if nxt not in done:\n                done.add(nxt)\n"
            "                frontier.append(path + [nxt])\n    return None, done\n", "correct", None),
    # and another: it tests for the goal when it puts a path in the queue, as AIMA does
    ("bfs", "from collections import deque\n\n\n" + _NEIGHBORS + "\n\n"
            "def bfs(maze, start, goal):\n    done = set()\n    if start == goal:\n        return [start], done\n"
            "    frontier = deque([[start]])\n    reached = {start}\n    while len(frontier) > 0:\n        path = frontier.popleft()\n"
            "        done.add(path[-1])\n        for nxt in neighbors(maze, path[-1]):\n            if nxt == goal:\n"
            "                return path + [nxt], done\n            if nxt not in reached:\n                reached.add(nxt)\n"
            "                frontier.append(path + [nxt])\n    return None, done\n", "correct", None),
    # right in MAZE, but it tests for the goal only after it took the start out: wrong when the start is the goal
    ("bfs", _B.replace("        if state == goal:\n            return path, done\n", "").replace(
            "                frontier.append(path + [nxt])\n",
            "                if nxt == goal:\n                    return path + [nxt], done\n                frontier.append(path + [nxt])\n"),
     "almost", None),
    ("bfs", _B.replace("            return path, done\n", "            return path\n"), "wrong", "two-values"),
    ("formulation", {}, "empty", None),
    ("formulation", {"crossroads": "state", "slide": "action"}, "wrong", "unfinished"),
    ("formulation", {**_SORT, "crossroads": "action"}, "wrong", "wrong:crossroads"),
    ("formulation", {**_SORT, "slide": "state"}, "wrong", "wrong:slide"),
    ("formulation", {**_SORT, "minutes": "goal"}, "wrong", "wrong:minutes"),
    ("formulation", _SORT, "correct", None),
]
