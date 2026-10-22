# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson search-astar: the mazes of the lesson, the course's own A* and breadth-first search, which draw
the play block side by side; and astar, the search the learner completes."""
import heapq
import re
import traceback
from collections import deque
from dataclasses import dataclass, field

from everything_ai import draw
from everything_ai.check import FunctionBox, Mistake

INF = float("inf")


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


MAZE = Maze([         # the maze of the lesson before, and of the play block
    "###############",
    "#.......#.....#",
    "#.###...#..#..#",
    "#...#......#..#",
    "#S..#..#####..#",
    "#...#.......#G#",
    "#......#......#",
    "###############",
], "MAZE")
SMALL = Maze([        # the maze of the worked example, as in the lesson before
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
TRAP = Maze([         # heading straight for G runs into a wall; the short way starts along the bottom
    "###########",
    "#......#..#",
    "#......#..#",
    "#......#.G#",
    "#......#..#",
    "#S........#",
    "###########",
], "TRAP")


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


def h(state, goal):
    """The heuristic of the lesson: the steps to the goal if there were no walls, rows apart plus columns apart."""
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def astar(maze, start, goal, weight=1, with_g=True):
    """The course's own A*, as in the lesson: the path and the set of expanded states. With another `weight`, h counts
    that many times, for exercise 3; without g, f is h alone, for a known mistake."""
    frontier = [(weight * h(start, goal), [start])]
    done = set()
    while len(frontier) > 0:
        f, path = heapq.heappop(frontier)
        state = path[-1]
        if state == goal:
            return path, done
        if state not in done:
            done.add(state)
            for nxt in neighbors(maze, state):
                g = len(path)
                heapq.heappush(frontier, (g * with_g + weight * h(nxt, goal), path + [nxt]))
    return None, done


def bfs(maze, start, goal):
    """The breadth-first search of the lesson before, to compare with."""
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


def _step(done: dict, paths, mark) -> dict:
    """One step of a search for the play block: what it expanded, and the states its frontier holds, each with the
    number of steps of the first path to it there."""
    front = {}
    for p in paths:
        if p[-1] not in done and p[-1] not in front:
            front[p[-1]] = len(p) - 1
    return {"done": dict(done), "frontier": front, "mark": mark, "path": None}


def trace(maze, start, goal, informed: bool) -> list[dict]:
    """What a search has done after each step: the states it expanded with their number of steps from the start, the
    states on its frontier, the state it expanded last; at the last step, the path it found. With `informed`, A*,
    with its priority queue as in the lesson; otherwise breadth-first search, with its queue."""
    steps = [{"done": {}, "frontier": {start: 0}, "mark": None, "path": None}]
    queue, heap, done = deque([[start]]), [(h(start, goal), [start])], {}
    while (heap if informed else queue):
        path = heapq.heappop(heap)[1] if informed else queue.popleft()
        state = path[-1]
        if state == goal:
            steps.append({**steps[-1], "mark": state, "path": path})
            return steps
        if state not in done:
            done[state] = len(path) - 1
            for nxt in neighbors(maze, state):
                if informed:
                    heapq.heappush(heap, (len(path) + h(nxt, goal), path + [nxt]))
                else:
                    queue.append(path + [nxt])
            steps.append(_step(done, [p for _, p in sorted(heap)] if informed else queue, state))
    return steps


BFS_STEPS = trace(MAZE, START, GOAL, informed=False)
ASTAR_STEPS = trace(MAZE, START, GOAL, informed=True)
SHORTEST, DONE = astar(MAZE, START, GOAL)
BFS_PATH, BFS_DONE = bfs(MAZE, START, GOAL)
TRAP_START, TRAP_GOAL = find(TRAP, "S"), find(TRAP, "G")
TOO_LARGE, _ = astar(TRAP, TRAP_START, TRAP_GOAL, weight=3)   # h three times too large, for exercise 3
TRAP_PATH, _ = astar(TRAP, TRAP_START, TRAP_GOAL)
ONLY_H, _ = astar(TRAP, TRAP_START, TRAP_GOAL, with_g=False)   # f without g, for a known mistake
FAR_STATE = (6, 3)   # the state of exercise 2: 4 steps from S in MAZE
FAR_G, FAR_H = distances(MAZE, START)[FAR_STATE], h(FAR_STATE, GOAL)


def at_step(steps: list[dict], k: int) -> dict:
    """A search after k steps; a search that already took out the goal stays at its last step."""
    return steps[min(k, len(steps) - 1)]


def heuristic(alt: str):
    """MAZE with h of every state in it."""
    numbers = {(r, c): h((r, c), GOAL) for r, row in enumerate(MAZE) for c, sign in enumerate(row) if sign != "#"}
    return draw.maze_grids([{"rows": MAZE, "numbers": numbers, "role": "pred", "index": True}], alt=alt, cell=28)


def side_by_side(k: int, titles: tuple[str, str], alt: str):
    """Breadth-first search and A* on MAZE after k steps, for the play block."""
    panels = []
    for steps, title in zip((BFS_STEPS, ASTAR_STEPS), titles):
        s = at_step(steps, k)
        panels.append({"rows": MAZE, "title": title, "done": s["done"], "frontier": s["frontier"], "mark": s["mark"],
                       "path": s["path"]})
    return draw.maze_grids(panels, alt=alt, cell=20)


def mazes(alt: str):
    """The other mazes the check gives astar, with their names."""
    return draw.maze_grids([{"rows": maze, "title": maze.name, "index": True} for maze in (SMALL, CLOSED, TRAP)],
                           alt=alt, cell=22)


# --- how a search's answer is judged ------------------------------------------------------------------------------

@dataclass(frozen=True)
class Found:
    """What a search gave back, as far as the check is concerned: not the exact path, since every shortest path is
    right, but its kind and number of steps, and whether the search expanded what A* expands before it takes out the
    goal: every state whose f, its fewest steps from the start plus h, is less than the length of a shortest path,
    and no state whose f is more."""
    kind: str                 # "path"; "none": None, no path; "bad": not a way from the start to the goal; "shape"
    steps: int | None = None  # the path cost of the path: its number of steps
    ok: bool | None = None    # it expanded what A* expands; without a way, every state it can reach
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
    """What `astar(maze, start, goal)` gave back, judged: None stays None, so the check can say that a return is missing."""
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
    shortest = far[goal]
    f = {s: d + h(s, goal) for s, d in far.items()}
    ok = all(s in done for s in far if f[s] < shortest) and all(f.get(s, INF) <= shortest for s in done)
    return Found("path", len(path) - 1, ok, len(done))


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
    expanded = {"en": "after expanding " + _count(found.n, "state", "states"),
                "nl": "na " + _count(found.n, "state", "states") + " uit te breiden"}
    if found.kind == "none":
        return {"en": f"`None`, no path, {expanded['en']}", "nl": f"`None`, geen pad, {expanded['nl']}"}
    return {"en": f"a path of {_count(found.steps, 'step', 'steps')}, {expanded['en']}",
            "nl": f"een pad van {_count(found.steps, 'stap', 'stappen')}, {expanded['nl']}"}


@dataclass
class SearchBox(FunctionBox):
    """A search in a code box. The check calls it on each maze and judges what it gives back (judge), so any shortest
    path is right, and a known mistake is known by what it does in every maze."""
    judge: object = None

    def _run(self, fn, case):
        return self.judge(*case, super()._run(fn, case))


# The lesson is done when these are correct. "f-value" and "too-large" are the choice questions in the lesson.
REQUIRED = ["astar", "f-value", "too-large"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

# --- exercise 1: astar ----------------------------------------------------------------------------------------------

_HEAD = ("import heapq\n\n\n"
         "def neighbors(maze, state):\n"
         "    r, c = state\n"
         "    result = []\n"
         "    for nxt in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:\n"
         "        if maze[nxt[0]][nxt[1]] != \"#\":\n"
         "            result.append(nxt)\n"
         "    return result\n\n\n"
         "def h(state, goal):\n"
         "    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])\n\n\n")


def _a(take: str = "f, path = heapq.heappop(frontier)", f: str = "f = g + h(nxt, goal)") -> str:
    """astar as the box holds it, with these two lines: step 2 of the recipe, and f for step 4."""
    return (_HEAD +
            "def astar(maze, start, goal):\n"
            "    frontier = [(h(start, goal), [start])]\n"
            "    done = set()\n"
            "    while len(frontier) > 0:\n"
            f"        {take}\n"
            "        state = path[-1]\n"
            "        if state == goal:\n"
            "            return path, done\n"
            "        if state not in done:\n"
            "            done.add(state)\n"
            "            for nxt in neighbors(maze, state):\n"
            "                g = len(path)\n"
            f"                {f}\n"
            "                heapq.heappush(frontier, (f, path + [nxt]))\n"
            "    return None, done\n")


_A_START = _a("...", "f = ...")
_A = _a()

EXERCISES = {
    "astar": SearchBox(
        name="astar",
        function="astar",
        start=_A_START,
        solution=_A,
        judge=judge,
        compare=same_found,
        show=show,
        visible=[(MAZE, START, GOAL)],
        hidden=[(SMALL, (1, 1), (3, 5)), (TRAP, TRAP_START, TRAP_GOAL), (CLOSED, (1, 1), (4, 4)), (SMALL, (1, 1), (1, 1))],
        first=[_DOTS],
        mistakes=[
            Mistake("h-wrong-place", matches=lambda got, text: re.search(r"^\s*f\s*=.*\bh\([^)\n]*\bstart\b", text, re.MULTILINE) is not None,
                    en="`{call}` gives {got}, and it should give {want}. Your h does not go from `nxt`, where the longer path ends, to the goal: `h(start, goal)` is the same for every path, and `h(nxt, start)` counts the way back to the start. Use `h(nxt, goal)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je h gaat niet van `nxt`, waar het langere pad eindigt, naar het doel: `h(start, goal)` is voor elk pad hetzelfde, en `h(nxt, start)` telt de weg terug naar de start. Gebruik `h(nxt, goal)`."),
            Mistake("only-g", wrong=_a(f="f = g"),
                    en="`{call}` gives {got}, and it should give {want}. Your f is only g, the steps so far: then the search takes the paths in order of their number of steps, as breadth-first search does, and expands as many states. Add the estimate of the steps still to go: `h(nxt, goal)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je f is alleen g, de stappen tot nu toe: dan pakt de search de paden op volgorde van hun aantal stappen, zoals breadth-first search, en breidt hij evenveel states uit. Tel de schatting van de stappen die nog komen erbij op: `h(nxt, goal)`."),
            Mistake("only-h", wrong=_a(f="f = h(nxt, goal)"),
                    en="`{call}` gives {got}, and it should give {want}. Your f is only h, the estimate of the steps still to go, so the search forgets how many steps a path took already: it heads straight for the goal. In TRAP, that hands back a way of " + f"{len(ONLY_H) - 1} steps, where {len(TRAP_PATH) - 1} is enough." + " Add g: `f = g + h(nxt, goal)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je f is alleen h, de schatting van de stappen die nog komen, dus de search vergeet hoeveel stappen een pad al zette: hij gaat recht op het doel af. In TRAP geeft dat een weg van " + f"{len(ONLY_H) - 1} stappen terug, waar {len(TRAP_PATH) - 1} genoeg is." + " Tel g erbij op: `f = g + h(nxt, goal)`."),
            Mistake("h-of-state", wrong=_a(f="f = g + h(state, goal)"),
                    en="`{call}` gives {got}, and it should give {want}. Your h is that of `state`, where the path ends now, but f belongs to the new path, which ends in `nxt`. Use `h(nxt, goal)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je h is die van `state`, waar het pad nu eindigt, maar f hoort bij het nieuwe pad, dat in `nxt` eindigt. Gebruik `h(nxt, goal)`."),
            Mistake("pop-last", wrong=_a("f, path = frontier.pop()"),
                    en="`{call}` gives {got}, and it should give {want}. `frontier.pop()` takes the last item of the list, and that is not the pair with the smallest f. Let `heapq` take that one out: `heapq.heappop(frontier)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. `frontier.pop()` pakt het laatste element van de lijst, en dat is niet het paar met de kleinste f. Laat `heapq` dat eruit halen: `heapq.heappop(frontier)`."),
            Mistake("two-values", matches=lambda got, text: isinstance(got, Found) and got.kind == "shape",
                    en="`{call}` gives {got}. Keep the lines `return path, done` and `return None, done`: the check needs both, the path and the set of states the search expanded.",
                    nl="`{call}` geeft {got}. Houd de regels `return path, done` en `return None, done`: de check heeft ze allebei nodig, het pad en de set states die de search uitbreidde."),
            Mistake("pair-as-path", error=lambda e: isinstance(e, TypeError) and "unhashable" in str(e),
                    en="With `{call}`, your function stops with an error: `heapq.heappop(frontier)` gives back the whole pair that went in, `(f, path)`, so `path[-1]` is a list instead of a state. Give the pair two names: `f, path = heapq.heappop(frontier)`.",
                    nl="Met `{call}` stopt je functie met een fout: `heapq.heappop(frontier)` geeft het hele paar terug dat erin ging, `(f, path)`, dus `path[-1]` is een lijst in plaats van een state. Geef het paar twee namen: `f, path = heapq.heappop(frontier)`."),
            Mistake("h-not-called", error=lambda e: isinstance(e, TypeError) and "'function'" in str(e),
                    en="With `{call}`, your function stops with an error: `h` is a function, so your f adds a number and a function. Call it for the state where the longer path ends: `f = g + h(nxt, goal)`.",
                    nl="Met `{call}` stopt je functie met een fout: `h` is een functie, dus je f telt een getal en een functie op. Roep hem aan voor de state waar het langere pad eindigt: `f = g + h(nxt, goal)`."),
            Mistake("swapped-pair", error=lambda e: isinstance(e, TypeError) and "'int' object is not subscriptable" in str(e)
                    and traceback.extract_tb(e.__traceback__)[-1].name == "astar",
                    en="With `{call}`, your function stops with an error: `path` is a number. `heapq.heappop(frontier)` gives back the pair as it went in, `(f, path)`, with f first, so the names go in that order: `f, path = heapq.heappop(frontier)`.",
                    nl="Met `{call}` stopt je functie met een fout: `path` is een getal. `heapq.heappop(frontier)` geeft het paar terug zoals het erin ging, `(f, path)`, met f eerst, dus de namen staan in die volgorde: `f, path = heapq.heappop(frontier)`."),
            Mistake("popleft", error=lambda e: isinstance(e, AttributeError) and "popleft" in str(e),
                    en="With `{call}`, your function stops with an error: `frontier` is a list that `heapq` keeps in order now, not a `deque`, so it has no `popleft()`. Take out the pair with the smallest f with `heapq.heappop(frontier)`.",
                    nl="Met `{call}` stopt je functie met een fout: `frontier` is nu een lijst die `heapq` op volgorde houdt, geen `deque`, dus hij heeft geen `popleft()`. Haal het paar met de kleinste f eruit met `heapq.heappop(frontier)`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "empty": {"en": "Replace the two `...`: step 2 of the recipe, which takes out the path with the smallest f, and f itself for the longer path of step 4.",
                      "nl": "Vervang de twee `...`: stap 2 van het recept, die het pad met de kleinste f eruit pakt, en f zelf voor het langere pad van stap 4."},
            "arguments": {"en": "The check calls `{call}`, with a maze, a start and a goal, and the `def` line of your function has another number of names between its parentheses. Keep `(maze, start, goal)`.",
                          "nl": "De check roept `{call}` aan, met een doolhof, een start en een doel, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(maze, start, goal)`."},
            "slow": {"en": "Your search is still not done after 2 seconds. Does every round of the `while` loop take a pair out of the priority queue? `heapq.heappop(frontier)` takes it out; `frontier[0]` only looks at it.",
                     "nl": "Je search is na 2 seconden nog niet klaar. Pakt elke ronde van de `while`-loop een paar uit de prioriteitswachtrij? `heapq.heappop(frontier)` haalt het eruit; `frontier[0]` kijkt er alleen naar."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Compare your two lines with steps 2 and 4 of the recipe.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Vergelijk je twee regels met stap 2 en 4 van het recept."},
            "almost": {"en": "Right for MAZE. But `{call}` gives {got}, and it should give {want}. Does your search work in every maze, also when no way leads to the goal?",
                       "nl": "Goed voor MAZE. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je search in elk doolhof, ook als er geen weg naar het doel is?"},
            "correct": {"en": "Your search is right in all {n} cases, also in TRAP, without a way to the goal and with the start as the goal. It takes out the path with the smallest f = g + h first, so it finds a shortest path and expands only states with an f no larger than its length.",
                        "nl": "Je search klopt in alle {n} gevallen, ook in TRAP, zonder weg naar het doel en met de start als doel. Hij pakt het pad met de kleinste f = g + h eerst, dus hij vindt een kortste pad en breidt alleen states uit met een f die niet groter is dan de lengte ervan."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("astar", _A_START, "empty", None),
    ("astar", _a(f="f = ..."), "wrong", "dots-left"),
    ("astar", _a(f="f = g"), "wrong", "only-g"),
    ("astar", _a(f="f = h(nxt, goal)"), "wrong", "only-h"),
    ("astar", _a(f="f = g + h(state, goal)"), "wrong", "h-of-state"),
    ("astar", _a("f, path = frontier.pop()"), "wrong", "pop-last"),
    ("astar", _a("path = heapq.heappop(frontier)"), "error", "pair-as-path"),
    ("astar", _a("f, path = frontier.popleft()"), "error", "popleft"),
    ("astar", _a(f="f = g + h"), "error", "h-not-called"),
    ("astar", _a("path, f = heapq.heappop(frontier)"), "error", "swapped-pair"),
    ("astar", _a(f="f = g + h(start, goal)"), "wrong", "h-wrong-place"),
    ("astar", _a(f="f = g + h(nxt, start)"), "wrong", "h-wrong-place"),
    # an error in h, not in astar, is not the swapped pair
    ("astar", _a(f="f = g + h(nxt[0], goal)"), "error", None),
    ("astar", _a("f, path = frontier[0]"), "slow", None),
    ("astar", _A.replace("            return path, done\n", "            return path\n"), "wrong", "two-values"),
    # other right searches: h first, and g of the path that ends in state, the same order of the frontier
    ("astar", _a(f="f = h(nxt, goal) + g"), "correct", None),
    ("astar", _a(f="f = len(path) - 1 + h(nxt, goal)"), "correct", None),
]
