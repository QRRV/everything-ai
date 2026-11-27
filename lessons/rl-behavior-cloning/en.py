# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rl-behavior-cloning. Route: course/route/lessen/reinforcement-learning.yaml. Checks, the bridge, the expert, the course's clones and the pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rl-behavior-cloning", "en")
    C = L.checks   # the bridge, the expert, the course's clones and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en) and [When steps can slip: value iteration](/course/open/rl-value-iteration?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_knn():
    quick_knn = L.choice(
        "quick-knn",
        question="Three training examples: (0, 0) with class A, (0, 3) with class B and (4, 0) with class B. Which class does k-NN with k = 1 give the new example (1, 1)?",
        options={
            "majority": "B, since two of the three training examples have class B",
            "right": "A, the class of the nearest training example",
            "mean": "Halfway between A and B",
            "tie": "None, since one neighbor is too few for a vote",
        },
        correct="right",
        why={
            "right": "The distances from (1, 1) are $\\sqrt{2} \\approx 1.41$ to (0, 0), $\\sqrt{5} \\approx 2.24$ to (0, 3) and $\\sqrt{10} \\approx 3.16$ to (4, 0). With k = 1 the nearest decides alone: A. With k = 3, B would win. In this lesson, k-NN with k = 1 copies an expert.",
            "majority": "With k = 3 all three vote, and B wins: how many neighbors vote with k = 1? See [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).",
            "mean": "k-NN gives a class that its neighbors have, not something between two classes: which neighbor decides here? See [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).",
            "tie": "With k = 1 the vote has one voter, the nearest training example, and it decides alone: which one is nearest to (1, 1)? See [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).",
        },
    )
    quick_knn
    return (quick_knn,)


@app.cell(hide_code=True)
def quick_check_slip():
    quick_slip = L.choice(
        "quick-slip",
        question="On a grid, an action goes where it says with probability 0.8 and slips to each side of it with 0.1, as in value iteration. An agent stands in the row right above a row of pits and goes right. What is the probability that this step ends in a pit?",
        options={
            "zero": "0, since right does not go toward the pits",
            "both": "0.2, the probability that the step slips at all",
            "right": "0.1, the slip to the side where the pits are",
            "half": "0.05, since the slip of 0.1 can go to either of two sides, and only one of them has the pits",
        },
        correct="right",
        why={
            "right": "When right slips, it goes up or down, each with 0.1, and down is the pit. On the bridge of this lesson, the expert in that row goes up, back to the middle: then a slip goes left or right, along the pits, never into them.",
            "zero": "Right does not go toward the pits, but a slip goes to a side of the action: which sides does right have? See [When steps can slip: value iteration](/course/open/rl-value-iteration?lang=en).",
            "both": "0.2 is the probability of a slip to either side, and only one side has the pits: which one? See [When steps can slip: value iteration](/course/open/rl-value-iteration?lang=en).",
            "half": "The 0.1 is already the probability for one side, and the other side has its own 0.1: what is it for the side of the pits? See [When steps can slip: value iteration](/course/open/rl-value-iteration?lang=en).",
        },
    )
    quick_slip
    return (quick_slip,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    So far your agents learned from rewards, by trying, for hundreds of episodes, and paid for every mistake along the way. Often someone can already do the task: a driver, a pilot, a person who steers a robot arm by hand. Then you can skip the trying: record what the expert does, and learn a policy from those recordings, the way a classifier learns from examples. That is how ALVINN, one of the first cars that steered itself, learned around 1990 at Carnegie Mellon University: from camera images of the road, each with the steering of the human driver. One problem showed up at once. A good driver keeps the car in the middle of the lane, so the recordings held almost no images of a car that had drifted to the side, and once ALVINN drifted, nobody had shown it the way back. Dean Pomerleau fixed it by making extra images, shifted to the side, each with the steering that brings the car back. Ross and Bagnell later showed, in 2010, that this is not bad luck: in the worst case the cost of copying grows with the square of the length of the task. This lesson shows why.

    **An everyday example.** A new driver who only ever sits beside an instructor sees how to drive in the middle of the lane, but not how to come back from the edge of the road, because the instructor never goes there. The first time the new driver drifts to the edge, it is a situation nobody showed, and a wrong move there takes the car further from anything the instructor did.
    """)
    return


@app.cell(hide_code=True)
def predict_falls():
    guess_falls = L.guess(
        "falls",
        question="The bridge in the picture is three cells wide, between two rows of pits. A step goes where it says with probability 0.8 and slips to each side of it with 0.1, as in value iteration. The expert is the best policy from value iteration: in the middle row it goes right, and in the two outer rows it steps back to the middle, so a slip never takes it into a pit. The expert walks from S to G once, and k-NN with k = 1 learns a clone from that one demonstration: the cell is the example, the expert's action its class. Averaged over 20 such demonstrations, each with its own seed, in how many episodes does the clone fall into a pit, do you think?",
        options={
            "never": "Almost never: in every cell the expert showed, the clone does what the expert did",
            "right": "In about 2 of every 5 episodes",
            "rare": "In about 1 of every 20 episodes",
            "always": "In almost every episode",
        },
        reveal="**In about 2 of every 5 episodes.** Over the 20 demonstrations, the clones fall in 43.8% of their episodes, while the expert falls in none of its 2,000. Where the expert stood, a clone does exactly what the expert did. But one demonstration shows only about 15 of the 35 cells, and a slip soon takes the clone to a cell nobody showed it.",
    )
    mo.vstack([mo.Html(C.world_picture(L.lang)), guess_falls])
    return (guess_falls,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Imitation learning.** An agent that learns from an expert instead of from rewards does **imitation learning**. The simplest way is **behavior cloning**: record what the expert does, and learn it as a classification task, with supervised learning. Every step of the expert gives one example: the state is the input, with its features, and the expert's action is the class. A classifier learns from those examples, and its prediction in a state becomes the action of the agent there: the agent is a *clone* of the expert. The recorded episodes of the expert are its **demonstrations**. The clone never sees a reward.

        **The bridge.** A bridge of three rows and twelve columns, with a row of pits above and below it. The agent starts on S, in the middle row on the left, and the episode ends on G, on the right, with a reward of +1. A step into a pit gives −1 and ends the episode too; every other step gives 0. As in value iteration, an action goes where it says with probability 0.8 and slips to each side of it with 0.1; a step into the edge of the bridge leaves the agent where it is. In code a state is (row, column), counted from the top left, with the pits in rows 0 and 4: S is (2, 0) and G is (2, 11). `CELLS` lists the 35 cells the agent can stand on.

        **The expert.** The expert is the best policy from value iteration, with $\gamma$ = 0.9: `EXPERT` in code, a dict with an action for every cell of `CELLS`. In the middle row it goes right; in the two outer rows it steps back to the middle, so a slip goes left or right, along the pits, never into them. The expert never falls: not once in 2,000 episodes, of 17 steps on average.

        **The clone.** k-NN, as in k-nearest neighbors, with the cell as the features, its row and its column, and the expert's action as the class. With k = 1, the clone does in every cell what the expert did in the nearest cell the expert stood on.
        """),
        mo.Html(C.idea_picture(L.lang)),
        mo.md(r"""
        In the picture, seed 0. Above, the expert's one demonstration: 21 steps on 17 of the 35 cells, shaded. Four times a slip takes the expert to an outer row, and each time it goes back to the middle. Below, the clone of that demonstration. On every shaded cell it does what the expert did; in 16 of the other 18 cells it does something else, outlined: in the outer rows it mostly copies the "right" of the middle row, and walks along the pits. The line is the clone's first episode. In (3, 11), right below G, the expert would step up onto G; the clone goes right, into the edge, stays where it is, and on its 21st step a slip takes it into the pit.

        **Compounding errors.** On the cells of its demonstration, a clone with k = 1 makes no mistakes at all. But it does not stay on those cells. A slip takes it to a cell the expert never stood on, where it can do something other than the expert, and that something else, such as walking along the pits instead of stepping back, keeps it on cells the expert never showed. Over the 20 clones of one demonstration, on average 32.4% of the steps on the cells they stand on are mistakes, and after a mistake the next step is a mistake too in 83.6% of the cases. Every fall comes right after a mistake: none of the expert's own actions can slip into a pit. Mistakes that take the agent to states where it makes more mistakes are **compounding errors**. The clone learned on the cells of the expert and is tested on cells of its own: a **distribution shift**, as when a model meets data unlike its training set.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **imitation learning** | | learning a policy from an expert instead of from rewards | |
        | **demonstration** | | an episode of the expert, recorded as pairs of a state and the expert's action there | `demonstrate` |
        | **behavior cloning** | | imitation learning as supervised learning: the state is the input, the expert's action the class | `clone` |
        | **compounding errors** | | mistakes that take the agent to states the expert never showed, where it makes more mistakes | `mistakes` |
        """),
        L.watch_out("The clone does not know that a pit is bad: it never saw a reward, only what the expert did. Where the expert never stood, it has nothing to go on but the nearest cell where the expert did stand."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Behavior cloning, with k-NN:

    1. Let the expert play episodes, and write down every step as a pair: the cell it stands on, and the action it takes there. These are the demonstrations.
    2. Put the cells of the pairs in `X`, the features, and the actions in `y`, the classes.
    3. For every cell of the bridge, let k-NN predict the action from `X` and `y`: that is the clone's policy.
    4. Let the clone play, and count where it does something other than the expert: on the cells of the demonstrations, and on the cells it stands on itself.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The clone of seed 0, in cell (1, 2): the row above the middle, where the expert never stood. The nearest cells of its demonstration:

    | Cell of the demonstration | Expert's action there | Distance to (1, 2) |
    |---|---|---|
    | (2, 2) | right | $\sqrt{1^2 + 0^2} = 1$ |
    | (2, 1) | right | $\sqrt{1^2 + 1^2} \approx 1.41$ |
    | (2, 3) | right | $\sqrt{1^2 + 1^2} \approx 1.41$ |
    | (1, 4) | down | $\sqrt{0^2 + 2^2} = 2$ |

    With k = 1 the nearest decides: the clone goes right in (1, 2), where the expert would go down, back to the middle row. Going right, the clone stays next to the pits: with probability 0.1 the step slips up, into a pit. With 0.8 it lands in (1, 3), again a cell the expert never stood on, at distance 1 from both (2, 3) and (1, 4). With a tie, `knn_predict` takes the pair that comes first in the demonstration, (2, 3): right again. Only in (1, 4), a cell of the demonstration, does the clone go down, like the expert.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_demonstrate():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Recording the expert

        `demonstrate(expert, episodes, seed)` lets `expert` play `episodes` episodes from `START`, and gives back `pairs`: one pair for every step, of the cell and the action the expert takes there, in the order of the steps. `expert` is a policy as a dict, such as `EXPERT`: `expert[state]` is the action in `state`. `step(state, action, rng)` is one step on the bridge: it gives back the new cell, the reward, and `over`, which is `True` once the agent is on G or in a pit. `rng = np.random.default_rng(seed)` is the random generator for the slips, which is why the box starts with `import numpy as np`. In `for _ in range(episodes):`, the name `_` stands for a number the loop does not need, and `while not over:` repeats its block until the episode is over. Replace the `...` with three lines: the expert's action, the pair into `pairs`, and the step.

        The check calls `demonstrate(EXPERT, 1, 0)`, the demonstration of the picture in *The idea*, and three other cases, one of them with an expert that only goes right.
        """),
        L.task("Make `demonstrate` write down, for every step, a pair of the cell and the expert's action, and then take the step."),
    ])
    return


@app.cell(hide_code=True)
def exercise_demonstrate():
    box_demonstrate = L.code_box("demonstrate", lines=12)
    box_demonstrate
    return (box_demonstrate,)


@app.cell(hide_code=True)
def check_demonstrate(box_demonstrate):
    result_demonstrate = L.check("demonstrate", box_demonstrate.value)
    result_demonstrate
    return (result_demonstrate,)


@app.cell(hide_code=True)
def hints_demonstrate():
    L.hints(
        "demonstrate",
        [
            "Step 1 of the recipe, inside the `while` loop: first `action`, the expert's action in `state`, from the dict `expert`; then the pair into `pairs`, with `pairs.append`; then `step`, which gives back three things: the new `state`, the reward and `over`.",
            "The order matters: the pair holds the cell where the expert takes the action, so it goes into `pairs` before `step` moves the agent. `append` takes one value, so the cell and the action go in together, as one tuple between parentheses.",
            "`            pairs.append(___)`",
        ],
        why="`pairs.append((state, action))` writes down the cell and the expert's action there as one tuple, before `step` takes the agent on. The three lines are `action = expert[state]`, `pairs.append((state, action))` and `state, reward, over = step(state, action, rng)`.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_clone():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The clone

        `clone(pairs, k)` learns a clone from `pairs`, as `demonstrate` gives them, and gives back `policy`: a dict with an action for every cell of `CELLS`. The box starts with `import numpy as np` again. The first loop, `for state, action in pairs:`, takes every pair apart into its cell and its action: put the cell into `X` and the action into `y`, with `append`. Then `np.array` turns both lists into arrays: `X` gets one row per pair, the row and the column of its cell. The second loop goes over every cell of `CELLS`, also the cells the expert never stood on, and gives `policy[state]` what `knn_predict` predicts. That is your function from k-nearest neighbors, `knn_predict(X_train, y_train, x, k)`: it gives back the class most of the `k` training examples nearest to `x` have, here an action. Replace the two `...`.

        The check calls `clone(demonstrate(EXPERT, 1, 0), 1)`, the clone of the picture in *The idea*, and four other cases, with k = 1, 3 and 9.
        """),
        L.task("Make `clone` put the cells into `X` and the actions into `y`, and give every cell of `CELLS` the action k-NN predicts for it."),
    ])
    return


@app.cell(hide_code=True)
def exercise_clone():
    box_clone = L.code_box("clone", lines=14)
    box_clone
    return (box_clone,)


@app.cell(hide_code=True)
def check_clone(box_clone):
    result_clone = L.check("clone", box_clone.value)
    result_clone
    return (result_clone,)


@app.cell(hide_code=True)
def hints_clone():
    L.hints(
        "clone",
        [
            "Steps 2 and 3 of the recipe. In the first loop, `X.append(state)` and `y.append(action)`. In the second loop, `policy[state]` gets the action that `knn_predict` predicts for `state`.",
            "`knn_predict` takes four things, in this order: the training examples, their classes, the new example and k. Which names hold them here?",
            "`        policy[state] = ___`",
        ],
        why="`policy[state] = knn_predict(X, y, state, k)` asks k-NN for the action in every cell, from the cells and actions of the expert. With k = 1, the clone copies the action of the nearest cell where the expert stood.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_mistakes():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Counting mistakes

        `mistakes(policy, expert, states)` gives back the share of the states in `states` where `policy` does something other than `expert`: a number from 0 to 1. `states` is a list of cells, and the same cell can be in it more than once; then it counts every time, as a step the agent takes there. In the loop, `for state in states:`, add 1 to `wrong` for every state where the two policies give different actions; after the loop, give back `wrong` divided by the number of states. Replace the two `...`.

        The check calls `mistakes(clone(demonstrate(EXPERT, 1, 0), 1), EXPERT, CELLS)`: the clone of exercise 2 on all 35 cells. Then the same clone on `demo_cells`, the 21 cells of its demonstration, one per step, and on `clone_cells`, the 238 cells it stands on in 20 episodes of its own; then `STRAIGHT`, an expert that only goes right; and the expert against itself.
        """),
        L.task("Make `mistakes` give back the share of the states where the policy and the expert give different actions."),
    ])
    return


@app.cell(hide_code=True)
def exercise_mistakes():
    box_mistakes = L.code_box("mistakes", lines=5)
    box_mistakes
    return (box_mistakes,)


@app.cell(hide_code=True)
def check_mistakes(box_mistakes):
    result_mistakes = L.check("mistakes", box_mistakes.value)
    result_mistakes
    return (result_mistakes,)


@app.cell(hide_code=True)
def hints_mistakes():
    L.hints(
        "mistakes",
        [
            "Step 4 of the recipe. In the loop, an `if` compares the action of `policy` in `state` with the action of `expert` there, with `!=`, the test for different; under it, `wrong = wrong + 1`. After the loop, divide `wrong` by `len(states)`.",
            "Both policies are dicts with a cell as the key. In the example, 16 of the 35 cells differ, so the function should give back 16 / 35, about 0.4571.",
            "`        if ___:`",
        ],
        why="`if policy[state] != expert[state]:` counts a mistake wherever the clone and the expert give different actions; `return wrong / len(states)` makes a share of it, so that lists of different lengths can be compared.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### From 1 to 50 demonstrations

    More demonstrations show more cells. The block below goes through 1, 2, 5, 10, 20 and 50 demonstrations, each time with 20 seeds, and lets every clone play 500 episodes. On the left, the share of the episodes in which the clones fall, with a vertical line at the number of demonstrations shown; the dashed line at 0 is the expert. On the right, the clone of seed 0.
    """)
    return


@app.cell(hide_code=True)
def explore_demos_cell():
    explore_demos = L.explore("demos", "demonstrations", lambda _v: C.demo_frame(int(_v), L.lang), start=1,
                              values=C.NS, role="",
                              task="Go from 1 to 50: from which number on does the clone of seed 0 do what the expert does in every cell, and how many falls are left there over all 20 seeds?")
    explore_demos
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    With one demonstration the clones fall in 43.8% of their episodes, with 5 in 22.3%, with 20 in 2.3%, and with 50 in none of the 10,000. On the cells of the demonstrations the clones never make a mistake, with any number of demonstrations. What goes down is the share of mistakes on the cells they stand on themselves: from 32.4% with one demonstration to 1.2% with 20. One demonstration shows about 15 of the 35 cells, 20 show about 32. The seeds differ a lot: with one demonstration, the clone of one seed falls in 22.4% of its episodes, and the clone of another in 57.2%.

    ### The cells of one demonstration with NumPy

    `C.course_runs()[1][0]` gives the run of seed 0 with one demonstration: a dict with its `"pairs"` and its `"policy"`, among other things. The cell below makes two arrays the shape of the world, 5 rows by 12 columns. `np.zeros((5, 12), dtype=int)` is full of zeros, whole numbers, and `np.zeros((5, 12), dtype=bool)` full of `False`. `_seen[_state] += 1` adds 1 at the cell `_state`: a tuple (row, column) picks one number of a 2D array. So `_seen` counts how often the expert stood on each cell, and `_wrong` is `True` where the clone does something other than the expert. `_seen[1:4]` takes rows 1 to 3, the bridge without the pits. `.astype(int)` turns `True` and `False` into 1 and 0, `.sum(axis=1)` adds up every row, and `&` is *and* for arrays of `True` and `False`: `_wrong & (_seen > 0)` is `True` where the clone makes a mistake on a cell the expert stood on. Run the cell.
    """)
    return


@app.cell
def example_numpy():
    import numpy as np

    _run = C.course_runs()[1][0]              # seed 0, one demonstration
    _seen = np.zeros((5, 12), dtype=int)      # how often the expert stood on each cell
    for _state, _action in _run["pairs"]:
        _seen[_state] += 1
    _wrong = np.zeros((5, 12), dtype=bool)    # True where the clone does something other than the expert
    for _state in C.CELLS:
        _wrong[_state] = _run["policy"][_state] != C.EXPERT[_state]
    print("how often the expert stood on each cell, rows 1 to 3:")
    print(_seen[1:4])
    print("mistakes of the clone, rows 1 to 3:")
    print(_wrong[1:4].astype(int))
    print("mistakes per row:", _wrong[1:4].sum(axis=1))
    print("mistakes on cells the expert stood on:", int((_wrong & (_seen > 0)).sum()))
    print("cells the expert stood on:", int((_seen > 0).sum()))
    return (np,)


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    The expert stood on every cell of the middle row before G, and on 6 cells of the outer rows: 17 cells. The clone makes 7 mistakes in the row above the middle, 9 in the row below it, and none in the middle row. On a cell the expert stood on, k = 1 copies the expert, so the line with the mistakes on those cells says 0. All 16 mistakes are on cells next to the pits, where one wrong step can be the last.

    ### Asking the expert again: DAgger

    The clone goes wrong on cells the expert never showed. So show them: let the clone play, ask the expert what it would have done in every cell the clone stood on, add those pairs to the demonstration, and learn the clone again. That is the idea of **DAgger**, by Ross, Gordon and Bagnell, in 2011; here it is done once. One demonstration and 5 episodes of the clone, with the expert's actions, give about 88 pairs, as many as 5 demonstrations. Those clones fall in 13.7% of their episodes, against 22.3% for the clones of 5 demonstrations: the extra pairs lie where the clone goes, not where the expert goes. DAgger does need an expert who can answer for any state, not only recordings: a person who watches the robot and says what it should have done.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_k():
    mo.md(r"""
    ### Exercise 4. More neighbors

    In k-nearest neighbors, a larger k often helped: more neighbors vote, so one odd training example matters less. The clones of this lesson all use k = 1. A colleague wants to use k = 9 for the clones of 5 demonstrations, so that single cells throw the clone off less.
    """)
    return


@app.cell(hide_code=True)
def choice_k_cell():
    choice_k = L.choice(
        "bigger-k",
        question="What happens to the falls of the clones of 5 demonstrations with k = 9 instead of k = 1?",
        options={
            "right": "More falls: the nine nearest cells are mostly in the middle row, so the clone goes right in the outer rows",
            "fewer": "Fewer falls: nine votes smooth out the odd cells, as with the penguins",
            "same": "The same falls: on the cells of the demonstrations, k makes no difference",
            "none": "No falls at all: with nine neighbors every cell has a neighbor where the expert stood, and the clone always finds the expert's action",
        },
        correct="right",
        why={
            "right": "With k = 9 the falls go from 22.3% to 55.3%; with k = 3 to 37.4%. An outer row has few cells of the demonstrations, the middle row many, all with right. The nine neighbors of a cell in an outer row are mostly middle cells, so right wins the vote, also in cells where the expert stood and stepped back to the middle: on the cells of the demonstrations the clones now make 16.2% mistakes, against none with k = 1. A vote helps against classes that are wrong by chance; the expert is never wrong, and the rare class is the one that matters.",
            "fewer": "A vote helps when single training examples are wrong by chance, and the expert's actions are never wrong. Which action do most of the nine nearest cells of a cell in an outer row have?",
            "same": "With k = 1 a clone copies the expert on the cells of its demonstrations; with nine voters, does the cell itself still decide?",
            "none": "Also with k = 1, every cell has a nearest cell where the expert stood: the question is what the nine nearest cells vote for. Which action do most of them have, for a cell in an outer row?",
        },
    )
    choice_k
    return (choice_k,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** In [Careful or bold: SARSA and Q-learning](/course/open/rl-sarsa?lang=en) the agent learned from rewards, by trying, and paid for its mistakes while it learned. Behavior cloning learns without any reward, from an expert, with the classifier of [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en): every state a training example, every action of the expert its class. That works on the states the expert showed, and the agent's own mistakes take it elsewhere. Robots that learn from people who steer them by hand often start this way, with many more demonstrations, and DAgger or RL afterwards fix what copying misses. The next lesson, *You get what you reward: reward misspecification*, goes back to rewards: an agent does what the reward says, also when that is not what you meant.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_k, result_clone, result_demonstrate, result_mistakes):
    L.wrap_up(result_demonstrate, result_clone, result_mistakes, choice_k)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
