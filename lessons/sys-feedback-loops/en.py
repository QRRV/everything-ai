# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson sys-feedback-loops. Route: course/route/lessen/systemen.yaml. Checks, the routes, the first week, the runs and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("sys-feedback-loops", "en")
    C = L.checks   # the eight routes and their appeal, the course's first week, the runs of the app and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [When the world changes: drift](/course/open/sys-drift?lang=en) and [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_label():
    quick_label = L.choice(
        "quick-label",
        question="The bike service logs every request and every answer of its model. Which drift can it see only once the truth comes in?",
        options={
            "data": "Data drift: a change in the inputs, such as the temperature",
            "pred": "Prediction drift: a change in the answers the model gives",
            "right": "Label drift: a change in the true answers, the bikes rented",
            "none": "None of them: the service sees all three at every request",
        },
        correct="right",
        why={
            "right": "The bikes rented come in only after the hour has passed; the inputs and the answers are in the log at once. In this lesson, the truth comes in only for what the model chose: a visitor can take only a route the app shows.",
            "data": "The inputs are in every request, so the log has them at once. Which values come in only later? See [When the world changes: drift](/course/open/sys-drift?lang=en).",
            "pred": "The service logs every answer the moment it gives it. Which values come in only later? See [When the world changes: drift](/course/open/sys-drift?lang=en).",
            "none": "The inputs and the answers are in the log at once, but one of the three is not. Which values come in only later? See [When the world changes: drift](/course/open/sys-drift?lang=en).",
        },
    )
    quick_label
    return (quick_label,)


@app.cell(hide_code=True)
def quick_check_explore():
    quick_explore = L.choice(
        "quick-explore",
        question="An agent with ε-greedy and $\\varepsilon$ = 0.1 takes a random action in a tenth of its steps. Why does it not always take the action with the largest $Q$?",
        options={
            "faster": "Random actions take less computing, so it gets through more episodes per second",
            "right": "Its Q-values come only from actions it took: the others would stay unknown",
            "slip": "The world slips now and then, and random actions copy that slipping",
            "forget": "Without random actions, the Q-table would forget what it learned before",
        },
        correct="right",
        why={
            "right": "The $Q$ of an action changes only when the agent takes it. Without random actions, an action that looked bad early is never tried again, even when it is the best. The app of this lesson has the same problem: it learns only about the routes it shows.",
            "faster": "A random action takes as long as any other, and the agent takes it for another reason. What would the agent never find out without random actions? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
            "slip": "The slipping comes from the world, whatever the agent chooses. What would the agent never find out without random actions? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
            "forget": "A $Q$ changes only when the agent takes that action, so nothing is forgotten without random actions. What is never learned then? See [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en).",
        },
    )
    quick_explore
    return (quick_explore,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A model in production does not only answer: what it answers often decides what happens next, and what happens next becomes its data. A shop that puts a product on top sells more of it; a model that learns from those sales then finds the product even better, and puts it on top even more. The model's output feeds back into its own data. Such a loop can lock a model onto a choice that was only a little ahead by chance, and the data never show what the other choices would have done. Comparing with the truth does not catch it either: the truth comes in only for what the model chose.

    **An everyday example.** A path across a lawn starts with a few people who cut a corner. Their steps flatten the grass, so the next people take the same line, and the path gets clearer every week. Nobody finds out whether another line would have been shorter: everyone walks on the worn path, because everyone walks there.
    """)
    return


@app.cell(hide_code=True)
def predict_loop():
    guess_loop = L.guess(
        "loop",
        question="The bike service's app suggests one of eight routes on its home screen. In a first week, it shows each visitor a random route; after that, always the route with the most rides so far. Route E is the route visitors take most often when they see it. The course ran this app on 1,000 different first weeks, each followed by 20,000 visitors. In how many of them does the app end up showing route E?",
        options={
            "all": "In almost all of them: the best route gets the most rides in the end",
            "right": "In about half of them: whichever route led after the first week",
            "most": "In about nine of ten: a first week rarely puts a worse route ahead",
            "few": "In hardly any: route E has too few rides to ever take the lead",
        },
        reveal="**In about half of them: 549 of 1,000.** In each run, the route with the most rides after the first week is shown to every later visitor, so it gets every later ride and stays on top for good. Route E led after the first week in 549 of the 1,000 weeks, and the app ended on E in exactly those. In the other 451, a route that fewer visitors take had a lucky week, most often route C, and kept the screen.",
    )
    guess_loop
    return (guess_loop,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **The app.** The bike service's app suggests one of eight routes, A to H, on its home screen. A visitor who sees a route takes it or not: route E is taken by 32 of every 100 visitors who see it, route F by only 4. The app does not know these numbers; it knows only how often it showed each route and how many rides each route got. In the first week, it showed a random route to each of 400 visitors. From then on, it shows the route with the most rides so far.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        **Feedback loop.** A *feedback loop* is a loop in which a model's output changes the data it learns from later. Here, the app's choice decides which route is seen, and only a route that is seen can get rides.

        **Degenerate feedback loop.** In a **degenerate feedback loop**, the loop narrows what the model shows until it shows almost one thing. After the first week, route G led with 15 rides against 13 for route E. So G is shown to every visitor, gets every ride, and leads by more with every ride; E is never shown again, and its 13 rides stay 13. The ride counts say that G is by far the best route, and they say it because the app made them.

        **Popularity bias.** **Popularity bias** is a model's habit of showing what is already popular more and more, because it gets the most clicks or rides, whether or not it is the best. A count of rides measures how often a route was shown as much as how good it is.

        **Rides per showing.** A better measure divides the rides of each route by its showings: route E got 13 rides in 44 showings, 0.30 per showing, and route G 15 in 53, 0.28. A route does not move up this ranking just by being shown.

        **Exploration.** Rides per showing alone can still get stuck: a route that had a bad first week is never shown again, so its number never gets the chance to rise. As in Q-learning, a share $\varepsilon$ of random showings keeps the app learning about every route: exploration. Each random showing costs rides, since most routes are worse than the best one; that is the price of finding out.

        **Measuring the spread.** To see a loop, count how often the model showed each choice, over time. In a degenerate feedback loop, the model shows fewer and fewer different things, and the share of the top choice creeps toward 1.

        **In practice.** Teams that run such systems log what was shown next to what was clicked or ridden, so that they can divide by the showings later. Many keep a small slice of the traffic in which a random choice is shown, to measure what the other choices would do. Chip Huyen's *Designing Machine Learning Systems* describes degenerate feedback loops, and random showings against them, in its chapter on monitoring.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | feedback loop | | a loop in which a model's output changes the data it learns from later | |
        | **degenerate feedback loop** | | a feedback loop that narrows what a model shows, until it shows almost one thing | |
        | **popularity bias** | | a model's habit of showing what is already popular more and more, because it gets the most clicks or rides | |
        | rides per showing | | the rides of a route divided by how often it was shown | `rides / shown` |
        | showing | | one time the app put a route on a visitor's screen | `shown[route]` |
        """),
        L.watch_out("More data do not fix a degenerate feedback loop. After 20,000 visitors, route G has 4,858 rides and route E still 13: the data grow only where the app looks."),
        L.watch_out("A test set made from the logs has the loop in it too: after the first week, it holds no showing of route E, so it can never show that E would have done better."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Finding and breaking a feedback loop:

    1. Ask whether the model's output decides which data come back: what is shown, who gets a loan, where the bikes go.
    2. Log what the model showed next to what came back, so that every count can be divided by its showings.
    3. Rank by rides per showing, not by rides.
    4. Show a random choice to a small share $\varepsilon$ of the visitors, and keep that share.
    5. Watch the spread: how often each choice is shown, over time.
    6. Weigh the cost: every random showing costs a little now, and tells the app what the other choices would do.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.ROUTES` is the string `"ABCDEFGH"`, the names of the eight routes. `C.SHOWN` holds how often the course's first week showed each route, and `C.RIDES` the rides each route got. `C.APPEAL` holds the chance that a visitor who sees a route takes it, which the app does not know. The cell prints the showings, the rides, the rides per showing rounded to two decimals, and the route with the most rides. Run the cell.
    """)
    return


@app.cell
def example_week():
    import numpy as np

    print(C.SHOWN)
    print(C.RIDES)
    print(np.round(C.RIDES / C.SHOWN, 2))   # rides per showing
    print(C.ROUTES[np.argmax(C.RIDES)])     # the route with the most rides
    return (np,)


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Route G got the most rides, 15, and route E 13. But E was shown only 44 times and G 53 times: per showing, E has 0.30 and G 0.28. The app that counts rides shows G from now on, while the first week already points at E. In Exercise 1, you let that app run.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_simulate():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The app that counts rides

        `simulate(appeal, rides, shown, visitors, seed)` lets the app serve `visitors` visitors after the first week. `rides` and `shown` hold the rides and the showings of each route so far; `appeal` holds the chance of each route, which only the simulation knows. `rng = np.random.default_rng(seed)` makes the random number generator. For each visitor `v` in `range(visitors)`, the app picks `route = np.argmax(rides)`, the route with the most rides so far, and adds 1 to `shown[route]`. Then the visitor takes the route with chance `appeal[route]`, and if so, the route gets a ride. The function gives back `rides`. Replace the `...` with a test that is true with chance `appeal[route]`.

        The check calls `simulate(C.APPEAL, C.RIDES, C.SHOWN, 20000, 1)`: the course's first week, then 20,000 visitors. Then 3,000 visitors with another seed, two routes of which the one in the lead is never taken, and three other routes.
        """),
        L.task("Make the visitor take the shown route with chance `appeal[route]`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_simulate():
    box_simulate = L.code_box("simulate", lines=8)
    box_simulate
    return (box_simulate,)


@app.cell(hide_code=True)
def check_simulate(box_simulate):
    result_simulate = L.check("simulate", box_simulate.value)
    result_simulate
    return (result_simulate,)


@app.cell(hide_code=True)
def hints_simulate():
    L.hints(
        "simulate",
        [
            "`route` is the route the app shows, and `appeal[route]` its chance. As in [Random numbers and seeds](/course/open/np-random?lang=en), `rng.random()` gives a float from 0 up to, but not including, 1.",
            "A random float from 0 up to 1 lies below 0.3 in about 30 of 100 draws. Which comparison is true with chance `appeal[route]`?",
            "`        if rng.random() ___ appeal[route]:`",
        ],
        why="`if rng.random() < appeal[route]:`: the random float lies below the chance of the route in that share of the visitors, so route G, with 0.24, is taken by about 24 of every 100. After the first week, G gets all 20,000 showings and 4,843 rides; route E is never shown again.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_choose():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Rides per showing, and a few random routes

        `choose(rides, shown, epsilon, rng)` gives back the route the app shows to one visitor. With chance `epsilon`, when `rng.random() < epsilon`, it gives a random route with `rng.integers(0, len(rides))`, as ε-greedy does in Q-learning. Replace the `...` with the route that has the most rides per showing.

        The check calls `choose(C.RIDES, C.SHOWN, 0.0, rng)` 200 times with one random number generator, and counts how often it gives each route. Then `C.RIDES_AFTER` and `C.SHOWN_AFTER`, the rides and the showings after the app of Exercise 1 served its 20,000 visitors, then two and three routes, and `epsilon` 0.1 and 1.0.
        """),
        L.task("Make `choose` give back the route with the most rides per showing, outside the random routes."),
    ])
    return


@app.cell(hide_code=True)
def exercise_choose():
    box_choose = L.code_box("choose", lines=4)
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
            "Step 3 of the recipe: rank by rides per showing, not by rides. `np.argmax` gives the place of the largest value of an array, and here a place is a route.",
            "In the worked example, `C.RIDES / C.SHOWN` gave the rides per showing of each route. In `choose`, which two arrays hold the rides and the showings?",
            "`    return np.argmax(___)`",
        ],
        why="`return np.argmax(rides / shown)`: the rides of each route divided by its showings, and the place of the largest. After the first week, that is route E, with 0.30 per showing. With 10% random routes, the app shows E to 18,024 of the 20,000 visitors and gets 30.5 rides per 100 visitors, against 24.2 for the app of Exercise 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_count_explore():
    mo.md(r"""
    ### Exercise 3. Random routes for the app that counts

    The app of Exercise 1 ranks by rides. A team keeps that ranking, and only adds 10% random routes, as in `choose`. Over the course's 1,000 first weeks, the app without random routes ends on route E in 549 of them.
    """)
    return


@app.cell(hide_code=True)
def choice_count_explore_cell():
    choice_count_explore = L.choice(
        "count-explore",
        question="In how many of the 1,000 does it end on route E with 10% random routes, still ranking by rides?",
        options={
            "most": "Almost always: the random routes show E often enough for its rides to catch up",
            "right": "About as often: the leader keeps 9 in 10 showings, and stays far ahead in rides",
            "more": "In about 700: every random showing gives route E a small chance to take the lead",
            "never": "Never: random routes take showings away from route E too, wherever it stands",
        },
        correct="right",
        why={
            "right": "552 of 1,000, against 549 without. A random showing goes to each route about once in 80 visitors, so route E gets about 250 showings and 80 rides over 20,000 visitors, while the leader gets about 18,000 showings and thousands of rides. Random routes help only an app that ranks by rides per showing; in *Let the app run*, you can try both.",
            "most": "Route E gets about 250 random showings over 20,000 visitors, and the leader about 18,000. Can the rides of E catch up with that?",
            "more": "A random showing of route E gives it a ride with chance 0.32, but route E gets about 1 in 80 showings, and the leader 9 in 10. Whose rides grow faster?",
            "never": "Random routes give route E showings it would not get otherwise. Who still gets most of the showings, and so most of the rides?",
        },
    )
    choice_count_explore
    return (choice_count_explore,)


@app.cell(hide_code=True)
def text_exercise_loops():
    mo.md(r"""
    ### Exercise 4. Loop or no loop?

    Six models at work. Put each with or without a feedback loop.
    """)
    return


@app.cell(hide_code=True)
def sort_loops_cell():
    sort_loops = L.sort("loops")
    sort_loops
    return (sort_loops,)


@app.cell(hide_code=True)
def check_loops(sort_loops):
    result_loops = L.check("loops", sort_loops)
    result_loops
    return (result_loops,)


@app.cell(hide_code=True)
def hints_loops():
    L.hints(
        "loops",
        [
            "Ask for each model: does what it gives change what happens next, and does that end up in its data?",
            "Three of the six decide what people see or get; the other three only watch something that happens anyway.",
            "Loop: best sellers first, \\_\\_\\_, \\_\\_\\_. No loop: a rain forecast, \\_\\_\\_, \\_\\_\\_.",
        ],
        why="**Loop:** best sellers first, bikes where the model expects riders, and loans for some: the model's output decides which sales, rentals or repayments come back. **No loop:** a rain forecast, a train's arrival and broken parts: the model only watches.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Let the app run

    Choose how the app ranks the routes and how many random routes it shows, and press the button. The app serves 20,000 visitors after the course's first week. The chart shows the rides per 100 visitors over the last 1,000 visitors; dashed, the 32 the app would get if it showed route E every time. Beside it, how often the app showed each route since the first week. Ranking by rides per showing uses your `choose` once it passes its check, and the course's until then. It takes a few seconds; then you can go back through the run.
    """)
    return


@app.cell(hide_code=True)
def demo_form():
    demo_rank = mo.ui.dropdown(
        options={"the most rides": "rides", "the most rides per showing": "rate"},
        value="the most rides",
        label="Rank by",
    )
    demo_eps = mo.ui.dropdown(
        options={"0%": 0.0, "5%": 0.05, "10%": 0.1, "20%": 0.2},
        value="0%",
        label="Random routes",
    )
    run_button = mo.ui.run_button(label="Let the app run")
    mo.vstack([
        mo.hstack([demo_rank, demo_eps, run_button], justify="start", wrap=True),
        L.task("Start with the most rides and 0%: the app of Exercise 1. Then rank by rides per showing, with 0% and with 10%. Which route does each app show most at the end, and how far under the dashed line does it stay?"),
    ])
    return demo_eps, demo_rank, run_button


@app.cell(hide_code=True)
def demo_run(box_choose, demo_eps, demo_rank, result_choose, run_button):
    if not run_button.value:
        _out = mo.md("*Press the button to let the app run.*")
    else:
        _yours = C.mine(box_choose.value, result_choose.state == "correct")
        _w = C.live_words(L.lang)
        _now = {}   # the live chart, and how often the app showed each route after the last visitor so far

        def _progress(_t, _per_100, _after):
            if "after" not in _now:   # from the first point on, the showings beside the chart
                _now["chart"].watch(lambda: C.now_picture(_now["after"], _now["t"], L.lang))
            _now["after"], _now["t"] = _after, _t
            _now["chart"].add(_t, _per_100, C.progress_line(_t, demo_rank.value, demo_eps.value, L.lang))

        _now["chart"] = L.live({"title": _w["title"], "x": _w["x"], "y": _w["y"], "alt": _w["alt"]},
                               C.VISITORS, role="ink", low=0, high=40, window=1, xticks=C.TICKS)
        _now["chart"].reference([(0, C.BEST_PER_100), (C.VISITORS, C.BEST_PER_100)], _w["best"], role="ink-2")
        _now["chart"].line("**Running** …")
        _run = C.demo(demo_rank.value, demo_eps.value, _yours, show=_progress)
        if _run == "choose":
            _out = _now["chart"].stopped("*Your `choose` gave an error, or no route, during the run. Does it still pass its check?*")
        elif _run == "slow":
            _out = _now["chart"].stopped("*The run took more than two minutes. Does your `choose` do more work than it needs?*")
        else:
            _out = mo.vstack([
                L.explore("rewind", "visitor", lambda _v: (C.rewind_frame(_run, int(_v), L.lang), C.rewind_text(_run, int(_v), L.lang)),
                          start=C.VISITORS, values=C.LOOKS, role="",
                          task="Go back to visitor 200, then forward again: when does the app settle on one route, and on which?"),
                mo.md(C.result_text(_run, demo_rank.value, demo_eps.value, _yours is not None, L.lang)),
            ])
    _out
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    The course ran each of the eight apps on its 1,000 first weeks, each followed by 20,000 visitors:

    | Rank by | Random routes | Ends on route E | Rides per 100 visitors |
    |---|---|---|---|
    | the most rides | 0% | 549 | 29.3 |
    | the most rides | 5% | 550 | 28.7 |
    | the most rides | 10% | 552 | 28.2 |
    | the most rides | 20% | 548 | 27.0 |
    | the most rides per showing | 0% | 780 | 31.0 |
    | the most rides per showing | 5% | 928 | 30.8 |
    | the most rides per showing | 10% | 970 | 30.3 |
    | the most rides per showing | 20% | 990 | 29.0 |

    Ranking by rides, random routes change almost nothing: the leader after the first week keeps the screen. Ranking by rides per showing does most of the work, with route E at the end in 780 of 1,000 weeks; in 201 of the other 220, it ends on route C, which 28 of 100 visitors take. Random routes raise the 780 to 970 with 10% and 990 with 20%, but each random showing costs rides now: 31.0 per 100 visitors without them, 29.0 with 20%. In this world, nothing changes during the 20,000 visitors. A team keeps a small random share anyway: without it, a route that falls below the leader is never shown again, so the app cannot notice when such a route gets better, such as when a new bridge makes route C shorter.
    """)
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [When the world changes: drift](/course/open/sys-drift?lang=en), the world moved by itself, and the truth showed it. Here the model moves its own data, and the truth comes in only for what it chose. The random routes are the ε-greedy of [Learning from your own steps: Q-learning](/course/open/rl-q-learning?lang=en), and every visitor is a draw of a random number generator, as in [Random numbers and seeds](/course/open/np-random?lang=en): with the same seed, the same visitors. In the next lesson, *Project: your model as a service, watched for drift*, your service checks the data of 2012 month by month.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_count_explore, result_choose, result_loops, result_simulate):
    L.wrap_up(result_simulate, result_choose, choice_count_explore, result_loops)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
