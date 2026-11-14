# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-network-data. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-network-data", "en")
    C = L.checks   # the degree distribution, the hubs, the airports with hubs closed, the download, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Graphs: nodes, edges and paths](/course/open/math-graphs?lang=en) and one about [Log scales and power laws](/course/open/math-log-scales?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_degrees():
    quick_degrees = L.choice(
        "quick-degree-sum",
        question="A graph without directions has 5 edges. What do the degrees of all its nodes add up to?",
        options={
            "right": "10",
            "five": "5",
            "depends": "That depends on the number of nodes",
        },
        correct="right",
        why={
            "five": "Every edge has two ends, and adds 1 to the degree of the node at each end. See the adjacency list in [Graphs: nodes, edges and paths](/course/open/math-graphs?lang=en): it holds two names for every edge.",
            "depends": "Every edge adds 1 to the degree of the node at each of its two ends, however many nodes there are. See the adjacency list in [Graphs: nodes, edges and paths](/course/open/math-graphs?lang=en): it holds two names for every edge.",
            "right": "Twice the number of edges: every edge adds 1 to the degree at each of its two ends. Today that gives the average degree of a network of 19,256 edges.",
        },
    )
    quick_degrees
    return (quick_degrees,)


@app.cell(hide_code=True)
def quick_check_slope():
    quick_slope = L.choice(
        "quick-slope",
        question="On a log-log plot, a straight line goes through (1, 1000) and (100, 0.1). What is its slope?",
        options={
            "right": "−2",
            "four": "−4",
            "half": "−0.5",
        },
        correct="right",
        why={
            "four": "From 1000 down to 0.1 is four factors of 10, but along, from 1 to 100, the line also goes two factors of 10. The slope is up divided by along. See *The idea* in [Log scales and power laws](/course/open/math-log-scales?lang=en).",
            "half": "The slope is up divided by along, not along divided by up: four factors of 10 down over two along. See *The idea* in [Log scales and power laws](/course/open/math-log-scales?lang=en).",
            "right": "Four factors of 10 down over two along: −4 / 2 = −2, a power law with exponent −2. Today you read such a slope off the degrees of a network.",
        },
    )
    quick_slope
    return (quick_slope,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Much of the world is a network: airports and the flights between them, web pages and their links, people and who follows whom. These networks are not grids, where every node has the same number of neighbors, and they are not random either. A few nodes have a great many edges, and most have only a few. That shapes what happens on them: how fast a delay or a virus spreads, what breaks when one node fails, and why a few items are seen far more often than all the others. Before you can say who matters in a network, you have to see how its edges are spread over its nodes.

    **An everyday example.** To fly from a small town to a small town on another continent, you almost always change planes at a big airport, such as Amsterdam or Atlanta. Two flights or three, never twenty.
    """)
    return


@app.cell(hide_code=True)
def predict_hubs():
    guess_hubs = L.guess(
        "hubs-guess",
        question="In the network of flights of this lesson, an airport has on average 11.2 neighbors: airports it flies to without changing planes. How many neighbors does the airport with the most have?",
        options={
            "small": "About 25",
            "middle": "About 60",
            "right": "About 250",
        },
        reveal="**About 250.** Amsterdam has 248 neighbors, Frankfurt 244, Paris 240. Half of all airports have 3 or fewer. In a random network with as many airports and edges, the airport with the most would have 27.",
    )
    guess_hubs
    return (guess_hubs,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The network of flights.** OpenFlights collects the routes of airlines: 67,663 of them, as they were in June 2014. A route is one airline flying from one airport to another. Make every airport a node, and join two airports by an edge when at least one route flies between them, in either direction: a graph without directions of 3,425 nodes and 19,256 edges. The degree of an airport is its number of neighbors, the airports you can fly to without changing planes.

        **The degree distribution.** The degrees add up to twice the number of edges, so the average degree is 2 × 19,256 / 3,425 = 11.2. Now count, for every degree, how many airports have it: the **degree distribution**. 768 airports have degree 1 and 744 degree 2; half of all airports have 3 or fewer. A few have far more: Amsterdam 248, Frankfurt 244, Paris 240, Istanbul 236 and Atlanta 217. Those are **hubs**, nodes with far more edges than almost all others. The 34 biggest hubs, 1% of the airports, are at one end of 5,495 of the 19,256 edges: 28.5%.
        """),
        mo.Html(C.degrees_picture({"normal": "Normal axes", "loglog": "Log-log plot", "degree": "degree", "airports": "airports",
                                   "alt": "Two charts of the number of airports with each degree. On normal axes, the dots form a spike at degree 1 to 3, up to 768 airports, and lie flat along the bottom up to degree 248. On a log-log plot, the filled dots fall along a dashed straight line from degree 2 to about 100 and thin out above it; the open dots of a random network form a narrow hump between degree 2 and 27."})),
        mo.md(r"""
        On normal axes, on the left, the degree distribution is a spike at the start and a long flat tail: the hubs vanish along the bottom. On the log-log plot, on the right, the filled dots, one for each degree, fall along a straight line from degree 2 to about 100, with slope about −1.5: an airport with 10 times as many neighbors is about 35 times rarer. A straight line on a log-log plot is a power law, as in [Log scales and power laws](/course/open/math-log-scales?lang=en). A network whose degree distribution follows a power law is a **scale-free network**: it has no typical degree, no scale. Above about 100 the dots thin out: 99 of the 148 degrees from 101 to 248 have no airport at all. The line would give about 100 airports with 101 to 248 neighbors, and about 150 with more; there are 64, and none: no airport has room for a thousand destinations. How many real networks are truly scale-free is debated, but many small nodes and a few huge hubs turn up in many of them, such as web pages and their links.

        The open dots are a random network with as many nodes and edges, each edge between two airports picked at random. Its degrees lie between 2 and 27, most of them close to 11: a hump around the average, and no hubs.

        **A small world.** Breadth-first search from an airport, counting the steps, gives the fewest flights to every other airport. 3,397 airports hang together; the other 28 have routes only among themselves, in seven small groups. Two airports that can reach each other are on average 4.10 flights apart, and of all pairs of airports, 67.6% are at most 4 flights apart. From Abilene in Texas, whose only neighbor is Dallas-Fort Worth, you reach 1,361 other airports with at most 3 flights. That is a **small-world network**: most nodes are a few steps apart, although each has only a few neighbors. The hubs make the shortcuts: from a small airport you fly to a hub, from there to another hub, and on to where you are going.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **degree distribution** | for every degree, how many nodes have it | `counts[3]`: the number of nodes with 3 neighbors |
        | **hub** | a node with far more edges than almost all others | Amsterdam, with 248 neighbors |
        | **scale-free network** | a network whose degree distribution follows a power law: a straight line on a log-log plot, and no typical degree | |
        | **small-world network** | a network in which most nodes are a few steps apart, although each node has only a few neighbors | |
        """),
        L.watch_out("The average degree, 11.2, describes almost no airport: 56% of them have 3 neighbors or fewer, and the hubs pull the average up. In a network with hubs, look at the whole degree distribution, not at the average alone."),
        mo.md(r"""
        What do the hubs do for the small world? The slider closes the biggest hubs, with all their routes: 10, 20, 50, 100 or 200 of them. The solid line is the share of all pairs of open airports that are at most that many flights apart; the dashed line the same when as many airports picked at random close; the grey line every airport open. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_closed_cell():
    def _frame(h):
        h = int(h)
        said = C.closed_said(h, L.lang)
        alt = ("A chart with, from 1 to 10 flights, the percentage of pairs of airports at most that many flights apart: "
               "with the biggest hubs closed, solid; with as many airports closed at random, dashed; with every airport open, grey. " + said)
        return C.closed_picture(h, {"title": "Pairs of airports within this many flights", "flights": "flights",
                                    "pairs": "% of pairs", "alt": alt}), said

    explore_closed = L.explore("close-hubs", "hubs closed", _frame, start=50, values=C.STOPS, role="",
                               task="Slide from 0 to 200. How far apart do the airports get with the hubs closed, and with as many airports closed at random?")
    explore_closed
    return


@app.cell(hide_code=True)
def text_after_slider():
    mo.md(r"""
    Close 100 airports at random, and 3,223 of the 3,325 that are left still hang together, on average 4.15 flights apart: almost as before. Close the 100 biggest hubs instead, 3% of the airports, and only 2,903 do, on average 5.74 flights apart; just 17.4% of the pairs are at most 4 flights apart, against 67.6% with every airport open. A network with hubs hardly notices a random airport closing, but loses its small world when its hubs close.

    ### Recipe

    To look at a network as data:

    1. Make every thing a node, and every link between two things an edge; without directions, store a set of neighbors for every node, so that two links between the same nodes count once.
    2. The degree of a node is the size of its set. Count how many nodes have each degree: the degree distribution.
    3. Draw it on a log-log plot. A straight line is a power law: a scale-free network, with hubs. A hump around the average means no hubs.
    4. Name the hubs: the nodes with the highest degrees.
    5. Count the steps from a node to every other with breadth-first search, to see how small the world is.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell follows steps 1 and 2. `flights.routes()` gives the routes as pairs of airport codes, from and to. The cell gives every airport an empty set, and puts each airport of a route in the set of the other, so that a route back, or a second airline on the same route, adds nothing. Then it adds up the degrees, which are twice the number of edges, and prints the number of airports, of edges and the average degree. Last, it prints the degree of Amsterdam, AMS, and of Abilene, ABI, with the one neighbor of Abilene, and counts the airports with at least 100 neighbors. `flights` is the course's `everything_ai.flights`, which the cell below gives once the routes are downloaded. Run the cell.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download the routes, 2.4 MB")
    return (get_button,)


@app.cell(hide_code=True)
def flights_cell(get_button):
    flights = C.flights_or_stop(get_button, {
        "ask": "*The routes of OpenFlights are not on your computer yet. The button downloads them once from the repository of OpenFlights on GitHub, 2.4 MB. The worked example needs them, and with them the pictures under your checks show all 3,425 airports; the rest of the lesson works without them. The course does not host them: their license, the Open Database License (ODbL), asks that a database made from them is shared under the same license.*",
        "busy": "**Downloading the routes** · {done} of {total} MB",
        "no_net": "*The download did not work: GitHub could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is no room on your disk for 2.4 MB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
        "here": "*The routes of OpenFlights are on your computer: 67,663 routes between 3,425 airports. They contain information from OpenFlights, made available under the ODbL.*",
    })
    return (flights,)


@app.cell
def example_network(flights):
    # the routes as pairs of airport codes: from, to
    _routes = flights.routes()
    print(len(_routes), _routes[:3])

    # 1. every airport and the set of its neighbors, without directions
    _graph = {}
    for _a, _b in _routes:
        if _a != _b:   # one route goes from an airport to itself
            if _a not in _graph:
                _graph[_a] = set()
            if _b not in _graph:
                _graph[_b] = set()
            _graph[_a].add(_b)
            _graph[_b].add(_a)

    # 2. the degrees add up to twice the number of edges
    _total = 0
    for _airport in _graph:
        _total = _total + len(_graph[_airport])
    print(len(_graph), _total / 2, round(_total / len(_graph), 2))

    # the degree of a hub and of a small airport, and the airports with at least 100 neighbors
    print(len(_graph["AMS"]), len(_graph["ABI"]), _graph["ABI"])
    _big = 0
    for _airport in _graph:
        if len(_graph[_airport]) >= 100:
            _big = _big + 1
    print(_big)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    67,663 routes become 3,425 airports and 19,256 edges: many routes fly the same pair of airports, in both directions or by more than one airline. The average degree is 11.24. Amsterdam has 248 neighbors, and Abilene one, Dallas-Fort Worth, written `{'DFW'}`: a set of one code. Only 64 airports, fewer than 2%, have 100 neighbors or more.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_counts():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The degree distribution

        `degree_counts(graph)` should give the degree distribution of `graph`: a dictionary with, for every degree, the number of nodes that have it. `graph` gives every node its neighbors, as the worked example made it. Replace the two `...`: the degree of `node`, and the new count of that degree. The check calls `degree_counts(small)` with the small network below, and tries four other networks.

        ```python
        small = {"H": {"A", "B", "C", "D", "K"}, "K": {"H", "E", "F", "G"}, "A": {"H", "B"}, "B": {"H", "A"},
                 "C": {"H"}, "D": {"H"}, "E": {"K"}, "F": {"K"}, "G": {"K"}}
        ```

        H and K are its hubs. A and B fly to H and to each other; C and D only to H, and E, F and G only to K.
        """),
        L.task("Make `degree_counts` give, for every degree, how many nodes have it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_counts():
    box_counts = L.code_box("degree-counts", lines=9)
    box_counts
    return (box_counts,)


@app.cell(hide_code=True)
def check_counts(box_counts):
    result_counts = L.check("degree-counts", box_counts.value)
    result_counts
    return (result_counts,)


@app.cell(hide_code=True)
def hints_counts():
    L.hints(
        "degree-counts",
        [
            "The degree of a node is its number of neighbors, and `graph[node]` holds them. Each node then adds one to the count of its degree.",
            "The length of a set is its number of elements. The lines `if degree not in counts:` and `counts[degree] = 0` make sure that `counts[degree]` is there, starting at 0.",
            "`degree = len(___)` and `counts[degree] = counts[degree] + ___`.",
        ],
        why="Every node is counted once, at its degree, so the counts add up to the number of nodes: 9 for the small network, 3,425 for the network of flights.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_flights():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. How many flights to every airport

        `flights_from(graph, start)` should give, for every airport that can be reached from `start`, the fewest flights to it: a dictionary with `start` at 0, its neighbors at 1, their neighbors at 2, and so on. It is the breadth-first search of [Graphs: nodes, edges and paths](/course/open/math-graphs?lang=en), but it keeps a number for each airport instead of a path. Replace the two `...`: when a neighbor `nxt` gets a number, and which number. The check calls `flights_from(small, "C")` with the small network of exercise 1, and tries five other cases, among them a ring.
        """),
        L.task("Make `flights_from` give the fewest flights from `start` to every airport it can reach."),
    ])
    return


@app.cell(hide_code=True)
def exercise_flights():
    box_flights = L.code_box("flights-from", lines=13)
    box_flights
    return (box_flights,)


@app.cell(hide_code=True)
def check_flights(box_flights):
    result_flights = L.check("flights-from", box_flights.value)
    result_flights
    return (result_flights,)


@app.cell(hide_code=True)
def hints_flights():
    L.hints(
        "flights-from",
        [
            "Breadth-first search takes the airports in the order it found them, so the first time it meets an airport, it has come the shortest way. What should happen the second time?",
            "`in` tells you whether a dictionary has a key yet. And the number of a neighbor follows from the number of the airport it was found from.",
            "`if nxt ___ flights:` and `flights[nxt] = flights[___] + 1`.",
        ],
        why="The frontier is a queue, so every airport 1 flight away comes out before any airport 2 flights away. The first number an airport gets is the smallest, and an airport that has a number never goes into the frontier again: that also stops the search going round a ring forever.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_fragile():
    mo.md(r"""
    ### Exercise 3. Which network falls apart?
    """)
    return


@app.cell(hide_code=True)
def choice_fragile_cell():
    choice_fragile = L.choice(
        "fragile",
        question="Take the network of flights and the random network with as many airports and edges, whose biggest node has 27 neighbors. In each, close the 100 nodes with the highest degree. In which do the nodes that are left get much further apart?",
        options={
            "right": "In the network of flights",
            "random": "In the random network",
            "same": "In both alike: 100 nodes is 3% of each",
        },
        correct="right",
        why={
            "random": "The random network has no hubs: its 100 biggest nodes have about 20 neighbors each, and the others find a way around them. See the slider in *The idea* for what closing the hubs of the network of flights does.",
            "same": "The 3% is the same, but not the edges they take with them: the 100 biggest hubs of the network of flights hold thousands of its edges, and the shortcuts between them. See the slider in *The idea*.",
            "right": "With its 100 biggest hubs closed, the network of flights goes from 4.10 to 5.74 flights on average, and only 2,903 of the 3,325 airports that are left still hang together, against 3,397 of 3,425 before. The random network goes from 3.63 to 3.72 steps on average, and every node that is left still hangs together.",
        },
    )
    choice_fragile
    return (choice_fragile,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The adjacency list and breadth-first search of [Graphs: nodes, edges and paths](/course/open/math-graphs?lang=en), now on a network of 3,425 real airports; the log-log plot and the power law of [Log scales and power laws](/course/open/math-log-scales?lang=en), now in its degree distribution. The degree says how many neighbors a node has, but not whether it lies on the way between others. In the next lesson, *Who matters in a network? Centrality and PageRank*, you measure how important a node is in other ways, and compute PageRank, the idea Google started with.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_fragile, result_counts, result_flights):
    L.wrap_up(result_counts, result_flights, choice_fragile)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
