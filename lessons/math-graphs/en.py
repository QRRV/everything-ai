# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson math-graphs. Route: course/route/lessen/klassiek-ml.yaml. Checks, the graphs and the play block: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("math-graphs", "en")
    C = L.checks   # the graphs of the lesson, and the course's adjacency list and matrix that draw the play block

    def names_in(names):
        """Names in words: B, E and F."""
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Searching a maze: breadth-first search](/course/open/search-bfs?lang=en) and [From lists to arrays: shape and dtype](/course/open/np-arrays?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_bfs():
    quick_bfs = L.choice(
        "quick-bfs",
        question="Breadth-first search finds a path from S to G in a maze. What do you know about that path?",
        options={
            "first": "It is the first way the maze offers; a shorter one may exist",
            "right": "No path from S to G has fewer steps",
            "least": "No other search expands fewer states to find it",
        },
        correct="right",
        why={
            "right": "Breadth-first search takes the oldest path from its queue first, so every shorter path came out before it, and none of them reached G.",
            "first": "Breadth-first search tries every path of 1 step, then of 2 steps, and so on, so the first path that reaches G is a shortest one. See [Searching a maze: breadth-first search](/course/open/search-bfs?lang=en).",
            "least": "Breadth-first search expands every state closer to S than G, also on the side away from G; its path is a shortest one, but it expands many states to find it. See [Searching a maze: breadth-first search](/course/open/search-bfs?lang=en).",
        },
    )
    quick_bfs
    return (quick_bfs,)


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="`m = np.array([[0, 1, 1], [1, 0, 0]])`. What is `m.shape`?",
        options={
            "flat": "`(6,)`",
            "swapped": "`(3, 2)`",
            "right": "`(2, 3)`",
        },
        correct="right",
        why={
            "right": "Two inner lists make 2 rows, of 3 numbers each: first the rows, then the columns.",
            "swapped": "The shape gives the rows first: two inner lists are 2 rows, of 3 numbers each. See [From lists to arrays: shape and dtype](/course/open/np-arrays?lang=en).",
            "flat": "A nested list makes an array with two directions, rows and columns, so its shape has two numbers. See [From lists to arrays: shape and dtype](/course/open/np-arrays?lang=en).",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Much of the world is made of things and the links between them: people and who knows whom, airports and the flights between them, web pages and their links, towns and roads. Written down, each of them is a **graph**: points, and lines between them. You met two already without the name. The maze of [Searching a maze: breadth-first search](/course/open/search-bfs?lang=en) is one, with a cell as a point and a step as a line, and so is the calculation in [Backpropagation by hand on a tiny graph](/course/open/nn-backward-by-hand?lang=en). Once a network is a graph, one piece of code works for all of them: breadth-first search finds a shortest way between two people just as it did in the maze.

    **An everyday example.** A metro map does not show how long the tracks are or which way they bend. It shows the stations, and which station follows which: a graph. That is all you need to plan a trip.
    """)
    return


@app.cell(hide_code=True)
def predict_pairs():
    guess_pairs = L.guess(
        "pairs",
        question="Seven people each count how many of the others they know: 2, 3, 3, 3, 3, 3 and 1. Knowing goes both ways. How many pairs of people know each other?",
        options={
            "sum": "18, the sum of the counts",
            "half": "9",
            "unknown": "You cannot tell from these counts",
        },
        reveal="**9.** Every pair is counted twice, once by each of the two. So the counts add up to 18, twice the number of pairs. In the words of this lesson: the degrees of a graph add up to twice its number of edges.",
    )
    guess_pairs
    return (guess_pairs,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        A **graph** is a set of **nodes** joined by **edges**. An edge joins two nodes, and the two are each other's neighbors. In this lesson, the nodes are seven people, A to G, and an edge means that two of them know each other. The **degree** of a node is its number of edges: A knows B and C, so A has degree 2. Every edge has two ends, so the degrees add up to twice the number of edges, as in the guess.

        Knowing someone goes both ways: an edge without a direction. In a **directed graph**, every edge has a direction, drawn as an arrow: A follows B on a social network, but B need not follow A. In a **weighted graph**, every edge carries a number, its weight, such as the length of a road in km.

        A **path** is a row of nodes in which each one is joined by an edge to the next, such as A, B, D: from A to D in two steps. Its length is its number of edges; in a weighted graph, you can also add up the weights along it. Breadth-first search finds a path with the fewest edges.

        To compute with a graph, you store it, in one of two ways. An **adjacency list** gives each node the list of its neighbors; in Python, a dictionary, where `graph["A"]` gives `["B", "C"]`. An **adjacency matrix** gives each node a number, and with it a row and a column: A is 0, B is 1, and so on. Entry `adj[i, j]` is 1 when an edge goes from node `i` to node `j`, and 0 when none does; in a weighted graph, it holds the weight. Without directions, every edge stands twice, at `[i, j]` and at `[j, i]`, so the matrix equals its transpose, as in [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en). Row `i` adds up to the degree of node `i`, so `adj.sum(axis=1)` gives every degree at once.

        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **graph** | nodes joined by edges | |
        | **node** | one of the points of a graph, such as a person | `"A"` |
        | **edge** | a link between two nodes, which makes them neighbors | `("A", "B")` |
        | **degree** | the number of edges at a node | `len(graph["A"])` |
        | **directed graph** | a graph whose edges have a direction, drawn as arrows | |
        | **weighted graph** | a graph whose edges carry a number, their weight | `("P", "Q", 4)` |
        | **path** | nodes in a row, each joined by an edge to the next | `["A", "B", "D"]` |
        | **adjacency list** | for each node, the list of its neighbors | `graph["A"]` gives `["B", "C"]` |
        | **adjacency matrix** | a matrix with a row and a column per node, and at row `i`, column `j` a 1, or the weight, for an edge from node `i` to node `j` | `adj[0, 1]` gives `1.0` |
        | `np.zeros((n, n))` | a matrix of `n` rows and `n` columns, all zeros: its shape as a tuple, as `.shape` writes it | `np.zeros((2, 3))` |
        """),
        L.watch_out("In everyday English, a *graph* can also be a chart of numbers. In this course, a chart is a chart, and a graph is always nodes and edges."),
    ])
    return


@app.cell(hide_code=True)
def figure_kinds():
    C.kinds({"undirected": "Undirected: who knows whom", "directed": "Directed: who follows whom", "weighted": "Weighted: roads in km"},
            alt="Three small graphs. Undirected: A, B, C and D, with five lines. Directed: the same four people, with five arrows; "
                "between B and D the arrow goes both ways. Weighted: the towns P, Q, R and S, with roads of 4 km from P to Q, "
                "3 km from Q to R, 9 km from P to R and 2 km from R to S.")
    return


@app.cell(hide_code=True)
def text_after_kinds():
    mo.md(r"""
    In the weighted graph, the road from P straight to R is one edge of 9 km; the way through Q has two edges, but adds up to 4 + 3 = 7 km. Breadth-first search counts edges, so it takes the road of 9 km.
    """)
    return


@app.cell(hide_code=True)
def text_play():
    mo.md(r"""
    The graph of the seven people three times: as a drawing, as an adjacency list and as an adjacency matrix, whose rows and columns carry the number and the name of each node. Slide to pick a node: its ring and its edges in the drawing, its line in the list and its row in the matrix are marked, and its neighbors are filled in grey.
    """)
    return


@app.cell(hide_code=True)
def explore_node_cell():
    def _frame(k):
        k = int(k)
        _name = C.NAMES[k]
        _nbrs = C.GRAPH[_name]
        _d = len(_nbrs)
        _alt = (f"The graph of the seven people, with node {_name} marked, and beside it its adjacency list and adjacency matrix. "
                f"{_name} has the neighbors {names_in(_nbrs)}.")
        _said = (f"Node {_name}, number {k}: degree {_d}. Its list is `{C.GRAPH[_name]}`, and row {k} of the matrix has "
                 f"{'a 1' if _d == 1 else f'{_d} ones'}, in the {'column' if _d == 1 else 'columns'} of {names_in(_nbrs)}. "
                 f"Column {k} is the same as row {k}: the edges have no direction.")
        return C.views(k, {"graph": "The graph", "list": "graph, the adjacency list", "matrix": "adj, the adjacency matrix"}, _alt).text, _said

    explore_node = L.explore("node", "node number", _frame, start=3, low=0, high=len(C.NAMES) - 1, step=1, role="",
                             task="Slide through all seven. Which node has the lowest degree, and where do you see that in each of the three?")
    explore_node
    return


@app.cell(hide_code=True)
def text_after_play():
    mo.md(r"""
    The list holds 18 names, two for each of the 9 edges; the matrix holds 49 numbers, of which 18 are 1. For 1,000 people who each know 10 others, the list would hold 10,000 names and the matrix 1,000,000 numbers, 99 out of 100 of them 0. The matrix tells you in one look whether two nodes are neighbors, `adj[i, j]`; the list is much shorter, and hands you the neighbors of a node at once.
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    From a list of edges to a graph you can compute with:

    1. Write down the nodes, and every edge as a pair of nodes, or with its weight as a third value.
    2. Adjacency list: give every node an empty list. For each edge from `a` to `b`, put `b` in the list of `a`; without directions, also put `a` in the list of `b`.
    3. Adjacency matrix: give every node a number from 0 to `n - 1`, and start from `np.zeros((n, n))`. For each edge from node `i` to node `j`, set `adj[i, j]` to 1, or to its weight; without directions, set `adj[j, i]` as well.
    4. The degree of a node is the length of its list, or the sum of its row.
    5. A search on the graph asks the list for the neighbors of a state: `graph[state]`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The seven people, with their edges. The cell builds the adjacency list and the adjacency matrix, prints the degree of every node both ways, and then runs the breadth-first search of [Searching a maze: breadth-first search](/course/open/search-bfs?lang=en) on the list: `_neighbors(maze, state)` has become `graph[state]`. `for _a, _b in _edges:` takes each pair apart, as `for key, value in d.items():` does. Run the cell.
    """)
    return


@app.cell
def example_graphs():
    import numpy as np
    from collections import deque

    # 1. the nodes, and the edges as pairs: who knows whom
    _names = ["A", "B", "C", "D", "E", "F", "G"]
    _edges = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "E"),
              ("D", "E"), ("D", "F"), ("E", "F"), ("F", "G")]

    # 2. the adjacency list: every node starts with an empty list of neighbors
    _graph = {}
    for _name in _names:
        _graph[_name] = []
    for _a, _b in _edges:
        _graph[_a].append(_b)
        _graph[_b].append(_a)   # no direction: b is a neighbor of a, and a of b
    print(_graph)

    # 3. the adjacency matrix: node number i has row i and column i
    _number = {}
    for _i in range(len(_names)):
        _number[_names[_i]] = _i
    _adj = np.zeros((len(_names), len(_names)))
    for _a, _b in _edges:
        _adj[_number[_a], _number[_b]] = 1
        _adj[_number[_b], _number[_a]] = 1
    print(_adj)

    # 4. the degree of every node, from the list and from its row of the matrix
    for _name in _names:
        print(_name, len(_graph[_name]), _adj[_number[_name]].sum())
    print(_adj.sum(axis=1))

    # 5. breadth-first search on the adjacency list
    def _bfs(graph, start, goal):
        frontier = deque([[start]])
        done = set()
        while len(frontier) > 0:
            path = frontier.popleft()
            state = path[-1]
            if state == goal:
                return path
            if state not in done:
                done.add(state)
                for nxt in graph[state]:
                    frontier.append(path + [nxt])
        return None

    print(_bfs(_graph, "A", "G"))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The degrees from the list and from the matrix agree: 2, 3, 3, 3, 3, 3 and 1, which add up to 18, twice the 9 edges. The matrix shows `1.` because `np.zeros` makes floats. The search finds a path of 4 steps from A to G, through B, D and F. The way through C, E and F is just as short; breadth-first search gives the one it reaches first.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_list():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Your own adjacency list

        `to_list(names, edges)` builds the adjacency list of a graph without directions, as in step 2 of the recipe. Replace the `...` with the lines that put each edge in the lists. The check calls `to_list(NAMES, EDGES)`, with the seven people of this lesson, and tries three other graphs, among them one with a name that has no edges. Any order of the neighbors counts as right.
        """),
        L.task("Make `to_list` put `b` in the list of `a`, and `a` in the list of `b`, for every edge."),
    ])
    return


@app.cell(hide_code=True)
def exercise_list():
    box_list = L.code_box("to-list", lines=8)
    box_list
    return (box_list,)


@app.cell(hide_code=True)
def check_list(box_list):
    result_list = L.check("to-list", box_list.value)
    result_list
    return (result_list,)


@app.cell(hide_code=True)
def hints_list():
    L.hints(
        "to-list",
        [
            "Step 2 of the recipe. An edge without a direction makes two people neighbors of each other: how many lists change for each edge?",
            "`graph[a]` is the list of `a`; `append` puts a name at its end.",
            "`graph[a].append(___)` and `graph[___].append(a)`.",
        ],
        why="`graph[a].append(b)` and `graph[b].append(a)` add to the list that is there, so every name collects all its neighbors, one edge at a time.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_matrix():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The roads as a matrix

        The four towns of the weighted graph above, now with numbers: P is 0, Q is 1, R is 2 and S is 3. The check's list `ROADS` holds each road as `(i, j, w)`, a road of `w` km between town `i` and town `j`:

        ```python
        ROADS = [(0, 1, 4), (1, 2, 3), (0, 2, 9), (2, 3, 2)]
        ```

        `to_matrix(n, edges)` should give the adjacency matrix of `n` nodes, with the length of each road in it, and 0 where there is no road. Replace the two `...`. The check calls `to_matrix(4, ROADS)`, and tries a graph with a node without edges and a graph without any edges.
        """),
        L.task("Start from a matrix of zeros with a row and a column for each node, and put the length of each road in both places where it belongs."),
    ])
    return


@app.cell(hide_code=True)
def exercise_matrix():
    box_matrix = L.code_box("to-matrix", lines=8)
    box_matrix
    return (box_matrix,)


@app.cell(hide_code=True)
def check_matrix(box_matrix):
    result_matrix = L.check("to-matrix", box_matrix.value)
    result_matrix
    return (result_matrix,)


@app.cell(hide_code=True)
def hints_matrix():
    L.hints(
        "to-matrix",
        [
            "Step 3 of the recipe, with the weight in the place of the 1. How big is the matrix, and in how many places does each road go?",
            "`np.zeros((n, n))` makes the matrix; `adj[i, j] = w` puts a number at row `i`, column `j`.",
            "`adj = np.zeros((___, ___))`, then `adj[i, j] = w` and `adj[___, ___] = w`.",
        ],
        why="`np.zeros((n, n))` gives every node a row and a column. A road goes both ways, so its length stands at `[i, j]` and at `[j, i]`, and the matrix equals its transpose.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_column():
    mo.md(r"""
    ### Exercise 3. Rows and columns with directions
    """)
    return


@app.cell(hide_code=True)
def choice_column_cell():
    choice_column = L.choice(
        "column",
        question="A directed graph of who follows whom: `adj[i, j]` is 1 when person `i` follows person `j`. Row `i` shows whom `i` follows. What does column `i` show?",
        options={
            "same": "Again whom `i` follows: a column is the same as its row",
            "followers": "Who follows `i`",
            "nothing": "Nothing: in a directed graph, only the rows count",
        },
        correct="followers",
        why={
            "followers": "Column `i` holds `adj[j, i]` for every `j`, which is 1 when `j` follows `i`. Without directions, the matrix equals its transpose, so a column is the same as its row; with directions, it need not be.",
            "same": "That holds in a graph without directions, whose matrix equals its transpose. Here, `adj[j, i]` is 1 when `j` follows `i`, which says nothing about whether `i` follows `j`. See *The idea*.",
            "nothing": "Every entry counts: `adj[j, i]`, in column `i`, is 1 when `j` follows `i`. See *The idea*.",
        },
    )
    choice_column
    return (choice_column,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The maze of [Searching a maze: breadth-first search](/course/open/search-bfs?lang=en) was a graph all along, with a cell as a node and a step as an edge; now the same search runs on any graph stored as an adjacency list. The adjacency matrix is an array as in [From lists to arrays: shape and dtype](/course/open/np-arrays?lang=en), and its row sums are the degrees. The calculation of [Backpropagation by hand on a tiny graph](/course/open/nn-backward-by-hand?lang=en) was a directed graph, with an arrow from each value to the values made from it. Graphs come back in the domains: networks of flights, recommendations from what people bought together, and neural networks that learn on graphs. In the next lesson, *A language model predicts the next token*, a model learns from text alone which token comes next.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_column, result_list, result_matrix):
    L.wrap_up(result_list, result_matrix, choice_column)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
