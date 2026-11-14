# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-pagerank. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-pagerank", "en")
    C = L.checks   # the karate club, its four centralities, the small web, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Networks as data: degrees and hubs](/course/open/ml-network-data?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_hubs():
    quick_hubs = L.choice(
        "quick-hubs",
        question="In the network of flights, the 100 biggest hubs close, with all their routes: 3% of the airports. What happens to the airports that are left?",
        options={
            "right": "They get much further apart: 5.74 flights on average instead of 4.10",
            "nothing": "Almost nothing: 3% of the airports is too few to matter",
            "apart": "None of them can reach another any more",
        },
        correct="right",
        why={
            "nothing": "That is what happens when 100 airports close at random. The hubs hold the shortcuts: see the slider in [Networks as data: degrees and hubs](/course/open/ml-network-data?lang=en).",
            "apart": "Most still hang together, 2,903 of the 3,325 that are left, but they are further apart. See the slider in [Networks as data: degrees and hubs](/course/open/ml-network-data?lang=en).",
            "right": "And only 2,903 of the 3,325 airports that are left still hang together. The degree told you which airports those are. Today you meet nodes that matter although their degree is not high.",
        },
    )
    quick_hubs
    return (quick_hubs,)


@app.cell(hide_code=True)
def quick_check_bfs():
    quick_bfs = L.choice(
        "quick-bfs",
        question="Breadth-first search gives an airport its number of flights the first time it meets it, and never changes it. Why is that first number the smallest?",
        options={
            "right": "The frontier is a queue: every airport 1 flight away comes out before any airport 2 flights away",
            "fewest": "It always takes the neighbor with the fewest edges first",
            "close": "It is not always the smallest, but it is close enough",
        },
        correct="right",
        why={
            "fewest": "Breadth-first search does not look at the degrees: it takes the airports in the order it found them. See exercise 2 of [Networks as data: degrees and hubs](/course/open/ml-network-data?lang=en).",
            "close": "It is always the smallest, and that is what makes it the fewest flights. See the *why* under the hints of exercise 2 in [Networks as data: degrees and hubs](/course/open/ml-network-data?lang=en).",
            "right": "So one breadth-first search gives the fewest steps from one node to every other. Today you add those steps up, for every node.",
        },
    )
    quick_bfs
    return (quick_bfs,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    The degree counts the neighbors of a node, and in the network of flights it found the hubs. But a node can matter in other ways. It can be close to all others, so that news from it spreads fast. It can lie on the way between two groups, so that what passes between them passes through it. Or it can be linked to by nodes that matter themselves. Each way gives another number, a centrality, and another ranking. The last one is PageRank, the idea Google started with in 1998 to rank web pages by their links. A random walk like the one behind it also ranks products: on a network of customers and products, a walk that keeps going back to one customer ends up most at the products close to them.

    **An everyday example.** In a group of friends, the one who knows the most people is not always the one through whom news reaches the other half of the group.
    """)
    return


@app.cell(hide_code=True)
def predict_between():
    guess_between = L.guess(
        "between-guess",
        question="In the karate club of this lesson, member 1 has 9 friends, and member 31 has 6. Which of the two lies on more shortest paths between two other members?",
        options={
            "one": "Member 1",
            "right": "Member 31",
            "same": "About as many",
        },
        reveal="**Member 31.** Shortest paths worth 73.0 of the 528 pairs of other members run through member 31, against 28.5 through member 1. Of the friends of member 1 besides member 0, all but one are friends of member 0 too, so most paths can go around member 1. Member 31 is friends with members of both halves of the club.",
    )
    guess_between
    return (guess_between,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The karate club.** From 1970 to 1972 Wayne Zachary followed a karate club at an American university. Of its 34 members, numbered 0 to 33 here, he wrote down which pairs also met outside the club: 78 pairs of friends, a graph without directions of 34 nodes and 78 edges. During those years the instructor, member 0, and the administrator, member 33, fell out, and the club split in two: 17 members went with each. In the pictures, the members who went with the instructor are filled, and the others open.

        **Centrality.** A **centrality** is a number for every node that says how much it matters in the network. Which centrality you take depends on what you mean by mattering. Four of them:

        1. **The degree**: the number of friends. Member 33 has 17, member 0 16.
        2. **Closeness centrality**: the number of other nodes divided by the sum of the steps to them, the fewest steps from breadth-first search. That is 1 divided by the average number of steps, so the closer a node is to all others, the higher. Member 0 reaches the 33 others in 58 steps together: 33 / 58 = 0.569, on average 1.76 steps to each.
        3. **Betweenness centrality**: for every pair of other nodes, the share of their shortest paths that run through the node, added up over all pairs. When two members have 3 shortest paths and 1 of them runs through member 0, that pair adds 1/3. The other 33 members make 33 × 32 / 2 = 528 pairs. Member 0 gets 231.1, member 33 160.6; 12 members get 0: no shortest path needs them.
        4. **PageRank**, by a **random walk**: a walker starts at a node, and at every step follows one edge of where it is, picked at random. Now and then, with a chance of 0.15, it jumps to a node picked at random instead. The **PageRank** of a node is the share of all steps the walker spends there in the long run. A node gets a high PageRank when many walks lead to it: when nodes link to it that are visited a lot themselves.

        The slider shows one centrality at a time: 1 is the degree, 2 closeness, 3 betweenness and 4 PageRank. The higher a member's centrality, the larger its circle; the highest, and members 1 and 31, have their number beside them. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_club_cell():
    def _frame(i):
        i = int(i)
        return C.picture(i, L.lang), C.said(i, L.lang)

    explore_club = L.explore("club-centrality", "centrality", _frame, start=1, values=[1, 2, 3, 4], role="",
                             task="Slide from 1 to 4. Which members stay among the highest, and where do members 1 and 31 go?")
    explore_club
    return


@app.cell(hide_code=True)
def text_power():
    mo.vstack([
        mo.md(r"""
        Members 0 and 33, the instructor and the administrator, are among the highest in all four. Below them the rankings differ. Member 1, with 9 friends, is 5th by degree but 9th by closeness. Member 31, with only 6, is 4th by closeness and 5th by betweenness: through it run the paths between the two halves of the club.

        **Power iteration.** A walker needs many steps before its shares settle. There is a faster way to the same numbers: spread the chance itself. Start every node at $1/n$, here $1/34$. Every round, every node passes $d$ times its score on to its neighbors, in equal parts; $d = 0.85$ is the chance that the walker follows an edge. Every node also gets $(1 - d)/n$ for the jumps. For a node $v$:

        $$\text{PR}_\text{new}(v) = \frac{1 - d}{n} + d \sum_{u} \frac{\text{PR}(u)}{\text{degree}(u)}$$

        The sum runs over the neighbors $u$ of $v$. Repeat this round after round: **power iteration**. After every round the scores add up to 1, because what one node passes on, another gets. On the club, the scores stop changing: from round 20 on, no score changes by more than 0.00001. Member 33 ends at 0.101 and member 0 at 0.097.

        **Directions.** Without directions, PageRank stays close to each node's share of the degrees: member 33 has 17 of the 156 ends of edges, 0.109. With directions it can be very different. Sergey Brin and Larry Page used it for the web, where a link goes one way: a page passes its score only to the pages it links to. A link from a page with a high PageRank and few links is then worth much more than many links from pages nobody links to.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **centrality** | a number for every node that says how much it matters in the network | |
        | **closeness centrality** | the number of other nodes divided by the sum of the steps to them: 1 divided by the average number of steps | `(len(steps) - 1) / total` |
        | **betweenness centrality** | for every pair of other nodes, the share of their shortest paths that run through the node, added up | |
        | **random walk** | a walk that at every step goes along an edge picked at random | `random.choice(club[node])` |
        | **PageRank** | the share of steps a random walker spends at a node, with a jump to a random node now and then | `scores[node]` |
        | $d$ | the chance that the walker follows an edge rather than jumps, here 0.85 | `d` |
        | **power iteration** | computing scores by repeating the same round of passing on until they stop changing | `for _ in range(rounds):` |
        """),
        L.watch_out("There is no one member who matters most. Member 33 is first by degree and PageRank, member 0 by closeness and betweenness, and member 31, with only 6 friends, is 4th by closeness. Choose the centrality that fits your question: who reaches everyone fast, who stands between groups, who is linked to by those who matter."),
        mo.md(r"""
        ### Recipe

        To find out who matters in a network:

        1. Say what you mean by mattering: many neighbors, close to everyone, between others, or linked to by nodes that matter.
        2. For the degree, count the neighbors.
        3. For closeness, run breadth-first search from the node, add up the steps, and divide the number of other nodes by that sum.
        4. For PageRank, give every node $1/n$, and repeat the round of passing on until the scores stop changing.
        5. Compare the rankings: a node that is high in one and low in another tells you what its place in the network is.
        """),
    ])
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell writes the club down as its 78 pairs of friends, and gives every member a list of friends. Each pair is in the list once, so no friend gets in a list twice. It prints the number of members, of edges, and the degrees of members 0 and 33. Then a random walker goes 100,000 steps through the club, starting at member 0. At every step it picks a number from 1 to 100: at 85 or below, it goes to a friend picked at random, and else it jumps to a member picked at random. The cell counts how often the walker is at every member, and prints the shares of members 33, 0 and 11. `random.seed(0)` makes every run the same. Run the cell.
    """)
    return


@app.cell
def example_walk():
    import random

    # the 78 pairs of members who also met outside the club (Zachary 1977)
    _pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11), (0, 12), (0, 13),
              (0, 17), (0, 19), (0, 21), (0, 31), (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
              (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32), (3, 7), (3, 12), (3, 13), (4, 6),
              (4, 10), (5, 6), (5, 10), (5, 16), (6, 16), (8, 30), (8, 32), (8, 33), (9, 33), (13, 33), (14, 32),
              (14, 33), (15, 32), (15, 33), (18, 32), (18, 33), (19, 33), (20, 32), (20, 33), (22, 32), (22, 33),
              (23, 25), (23, 27), (23, 29), (23, 32), (23, 33), (24, 25), (24, 27), (24, 31), (25, 31), (26, 29),
              (26, 33), (27, 33), (28, 31), (28, 33), (29, 32), (29, 33), (30, 32), (30, 33), (31, 32), (31, 33),
              (32, 33)]

    # every member and the list of their friends
    _club = {}
    for _m in range(34):
        _club[_m] = []
    for _a, _b in _pairs:
        _club[_a].append(_b)
        _club[_b].append(_a)
    print(len(_club), len(_pairs), len(_club[0]), len(_club[33]))

    # a random walker: follow a friend with chance 85 in 100, else jump to a random member
    random.seed(0)
    _node = 0
    _visits = {}
    for _m in _club:
        _visits[_m] = 0
    for _ in range(100000):
        if random.randint(1, 100) <= 85:
            _node = random.choice(_club[_node])
        else:
            _node = random.randint(0, 33)
        _visits[_node] = _visits[_node] + 1
    print(_visits[33] / 100000, _visits[0] / 100000, _visits[11] / 100000)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    34 members, 78 edges; member 0 has 16 friends and member 33 17. The walker spends 9.97% of its steps at member 33 and 9.80% at member 0, close to their PageRank from power iteration, 0.101 and 0.097. Member 11 has one friend, member 0, and gets 0.89% of the steps. With more steps the shares come closer to the PageRank; power iteration gets there in 20 rounds, without chance.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_closeness():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Closeness centrality

        `closeness(graph, node)` should give the closeness centrality of `node`: the number of other nodes it reaches, divided by the sum of the steps to them. `graph` gives every node its neighbors, as the worked example made the club. The box starts with `steps_from`, the breadth-first search of [Networks as data: degrees and hubs](/course/open/ml-network-data?lang=en): it gives, for every node that `start` reaches, the fewest steps to it, with `start` itself at 0. Replace the two `...`: the new `total`, and what the function gives back. The check calls `closeness(club, 0)` with the club of the worked example, and tries seven other cases, among them a graph in two parts.
        """),
        L.task("Make `closeness` give the number of other nodes divided by the sum of the steps to them."),
    ])
    return


@app.cell(hide_code=True)
def exercise_closeness():
    box_closeness = L.code_box("closeness", lines=21)
    box_closeness
    return (box_closeness,)


@app.cell(hide_code=True)
def check_closeness(box_closeness):
    result_closeness = L.check("closeness", box_closeness.value)
    result_closeness
    return (result_closeness,)


@app.cell(hide_code=True)
def hints_closeness():
    L.hints(
        "closeness",
        [
            "`steps` has a number for every node that `node` reaches, `node` itself included, at 0 steps. Add those numbers up in `total`.",
            "`node` itself is in `steps` too, so the number of other nodes is one less than the number of nodes in `steps`.",
            "`total = total + steps[___]` and `return (len(___) - 1) / total`.",
        ],
        why="Dividing the number of others by the sum is the same as 1 divided by the average number of steps: few steps on average gives a high closeness. `node` itself adds 0 to the sum, so it may stay in the loop, but it must not count as one of the others.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_pagerank():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. PageRank by power iteration

        `pagerank(graph, d, rounds)` should give the PageRank of every node after `rounds` rounds of power iteration. `graph` gives every node the nodes it passes its score on to: its neighbors, or, with directions, the pages it links to. Every node starts at `1 / n`. Every round, `new` starts at `(1 - d) / n` for every node, for the jumps; then every node passes `d` times its score on, in equal parts, to the nodes in `graph[node]`. Replace the two `...`: the part that `node` passes to each of them, and the new score of `nxt` with that part. The check calls `pagerank(club, 0.85, 100)`, and tries five other cases, among them the small web of exercise 3.
        """),
        L.task("Make `pagerank` pass every node's score on, round after round, and give the scores."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pagerank():
    box_pagerank = L.code_box("pagerank", lines=15)
    box_pagerank
    return (box_pagerank,)


@app.cell(hide_code=True)
def check_pagerank(box_pagerank):
    result_pagerank = L.check("pagerank", box_pagerank.value)
    result_pagerank
    return (result_pagerank,)


@app.cell(hide_code=True)
def hints_pagerank():
    L.hints(
        "pagerank",
        [
            "A node passes on `d` times its score, not all of it: the rest is what the jumps hand out. And that part is split in equal parts over the nodes in `graph[node]`.",
            "`new[nxt]` already holds what `nxt` got from the jumps and from the nodes before. The part of this node comes on top of it.",
            "`share = d * scores[node] / len(___)` and `new[nxt] = new[nxt] + ___`.",
        ],
        why="Every node passes on `d` times its score and every node gets `(1 - d) / n` from the jumps, so the scores keep adding up to 1, round after round. A node with many neighbors passes on smaller parts: that is why PageRank does not simply count the links to a node.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_web():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Which page gets more?

        In this small web, every other page links to A. A links only to X. D and E, to which no page links, also link to Y.

        ```python
        web = {"A": {"X"}, "B": {"A"}, "C": {"A"}, "D": {"A", "Y"}, "E": {"A", "Y"},
               "F": {"A"}, "X": {"A"}, "Y": {"A"}}
        ```
        """),
        C.web_picture(L.lang),
    ])
    return


@app.cell(hide_code=True)
def choice_web_cell():
    choice_web = L.choice(
        "web",
        question="With d = 0.85, which page gets the higher PageRank: X, with one link to it, from A, or Y, with two links, from D and E?",
        options={
            "right": "X",
            "y": "Y: it has more links to it",
            "same": "Both alike: each gets its links from pages with few links",
        },
        correct="right",
        why={
            "y": "Counting links is what PageRank does not do. What matters is the score of the page a link comes from, and how many links it splits that score over. How high is the score of A, and of D and E?",
            "same": "A has a link from every other page, so its score is high, and it passes 0.85 of it to X alone. No page links to D or E: they only have what the jumps give them, and split 0.85 of it over two links.",
            "right": "X gets 0.411 and Y 0.035. A gets links from all 7 other pages and has the highest PageRank, 0.461; it passes 0.85 of it to X alone. D and E only get what the jumps give, 0.019 each, and split it over A and Y.",
        },
    )
    choice_web
    return (choice_web,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The graphs and breadth-first search of [Networks as data: degrees and hubs](/course/open/ml-network-data?lang=en) give closeness: breadth-first search from every node, its steps added up. The degree of that lesson is one centrality of four. PageRank adds a walk by chance, and power iteration turns it into scores that add up to 1. With a walk that keeps jumping back to one customer, the same steps rank products on a network of customers and products, the kind that [Customers like you: collaborative filtering](/course/open/ml-collaborative-filtering?lang=en) makes from orders. This ends the module on recommending. The next module is about tables in practice, starting with the model to try first on a table.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_web, result_closeness, result_pagerank):
    L.wrap_up(result_closeness, result_pagerank, choice_web)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
