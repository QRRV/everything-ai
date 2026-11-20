# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-swiss-cheese. Route: course/route/lessen/verantwoord.yaml. Checks: checks.py.
# termen:allow defence in depth

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("rai-swiss-cheese", "en")
    C = L.checks   # holes that line up, a fault tree with AND and OR, and safeguards in a row with a shared trick


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [Designing for safety](/course/open/rai-safe-design-principles?lang=en), and one about [Conditional probability](/course/open/prob-conditional?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_copies():
    quick_copies = L.choice(
        "quick-copies",
        question="Three copies of the same model check every invoice and vote. Why does that protect little against a mistake of the model?",
        options={
            "five": "A vote needs at least five voters to work",
            "right": "The copies make the same mistake on the same invoice",
            "reverse": "Voting only helps when the models are wrong more often than right",
        },
        correct="right",
        why={
            "right": "Copies have the same weak spots, so when one is wrong, all three are: the vote outvotes nothing. Redundancy helps when the parts do not fail for the same reason.",
            "five": "Three is the least for a vote: two can outvote a third. The trouble is that copies make the same mistakes. See [Designing for safety](/course/open/rai-safe-design-principles?lang=en).",
            "reverse": "It is the other way round: voting helps only when each voter is right more often than wrong. And copies do not help even then, because they make the same mistakes. See [Designing for safety](/course/open/rai-safe-design-principles?lang=en).",
        },
    )
    quick_copies
    return (quick_copies,)


@app.cell(hide_code=True)
def quick_check_given():
    quick_given = L.choice(
        "quick-given",
        question="Of 1,000 harmful requests, a first safeguard misses 100. A second safeguard misses 30 of those 100. What is P(second misses | first misses)?",
        options={
            "right": "0.3: 30 of the 100",
            "all": "0.03: 30 of all 1,000 requests",
            "first": "0.1: the first safeguard's 100 of 1,000",
        },
        correct="right",
        why={
            "right": "Given that the first safeguard missed, you count only among the 100 requests it missed: 30 of 100 is 0.3.",
            "all": "The condition decides which cases you count among: only the 100 requests the first safeguard missed. 30 of those 100 is 0.3. See [Conditional probability](/course/open/prob-conditional?lang=en).",
            "first": "0.1 is the chance that the first safeguard misses, P(first misses). The question asks about the second, among the 100 the first missed: 30 of 100 is 0.3. See [Conditional probability](/course/open/prob-conditional?lang=en).",
        },
    )
    quick_given
    return (quick_given,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A made-up company runs a chatbot. To stop requests for dangerous help, it puts three safeguards in a row: a filter on the incoming question, a model trained to refuse, and a checker that reads every answer before it goes out. None of them is perfect; each one misses some harmful requests. So is a row of three safeguards that each stop 90% as good as one safeguard that stops 99.9%? Only if their misses have nothing to do with each other.

    Nuclear power plants, aviation and spaceflight have put safeguards in a row for decades. This lesson gives the picture safety science uses for it, the Swiss cheese model, and the engineers' way of computing it, the fault tree, and shows where both go wrong when safeguards miss for the same reason.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Three safeguards each miss 10% of ordinary harmful requests, independently of each other. But 1% of the harmful requests use a trick that fools all three. Of 1,000,000 harmful requests, about how many get through all three?",
        options={
            "independent": "About 1,000",
            "right": "About 11,000",
            "one": "About 100,000",
        },
        reveal="**About 11,000.** The 990,000 ordinary requests meet three safeguards that each miss 10%, independently: 990,000 × 0.1 × 0.1 × 0.1 = 990 get through. The 10,000 requests with the trick all get through. Together that is 10,990, almost eleven times the 1,000 that would get through without the trick.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Slices of cheese with holes

    Picture each safeguard as a slice of Swiss cheese. Every slice has holes, the places where it fails, and harm gets through only where holes line up. In the words of Dan Hendrycks's textbook *Introduction to AI Safety, Ethics, and Society* (2024): "If there are any places where holes in all the slices line up, creating a continuous hole through all of them, this represents a possible route to an accident." That is the **Swiss cheese model**.

    The model comes from James Reason, who started it in 1987 and 1988 while writing the book *Human Error* (1990). In a note of 2006, Reason, Erik Hollnagel and Jean Paries write that the name was not Reason's own: it was probably Rob Lee's, then director of the Bureau of Air Safety Investigation in Canberra. What the model says, in their words: "it conveys the fact that no one failure, human or technical, is sufficient to cause an accident." And where the holes come from: "Short-term breaches may be created by the errors and violations of front-line operators. Longer-lasting and more dangerous gaps are created by the decisions of designers, builders, procedure writers, top-level managers and maintainers."

    The textbook draws the conclusion: "This model highlights the importance of defense in depth, since having more layers of defense reduces the probability of there being a pathway to an accident that can bypass them all." There are two ways to make a system safer: "We can improve safety by increasing the number of slices, or by reducing the holes."

    ### Defense in depth

    The idea of layers is older than the cheese, and so are the doubts about it. Nuclear safety worked with layers early on, and Reason, Hollnagel and Paries note that Perrow, in the book *Normal Accidents* (1984), "had stressed the role of defences-in-depth in creating complexity, tight-coupling and opacity in the system": layers also make a system harder to see through. The US Nuclear Regulatory Commission defines **defense in depth** as "An approach to designing and operating nuclear facilities that prevents and mitigates accidents that release radiation or hazardous materials. The key is creating multiple independent and redundant layers of defense to compensate for potential human and mechanical failures so that no single layer, no matter how robust, is exclusively relied upon." In 1996, INSAG, the nuclear safety advisers of the International Atomic Energy Agency, described how the idea grew from three levels to five: "Should one level fail, the subsequent level comes into play." They added a warning: "The existence of several elements of defence in depth does not justify continued operation in the absence of one element."

    AI companies use the same words. In May 2025, when Anthropic switched on stricter protections for one of its models, it quoted its own policy: "Use a ‘defense in depth’ approach by building a series of defensive layers, each designed to catch misuse attempts that might pass through previous barriers." The layers have a price: the measures it added, classifiers among them, "may still occasionally affect legitimate queries (that is, they may produce false positives)."

    ### Layers as nines

    The textbook links the layers to [Nines of reliability](/course/open/rai-nines?lang=en): "We can think of layers of defense as giving us additional nines of reliability. In many cases, it seems reasonable to assume that adding a layer of defense helps reduce remaining risks by approximately an order of magnitude by eliminating 90% of the risks still present." Note "it seems reasonable to assume": the 90% is an assumption, not a measurement. Its example has three layers that aim at different things: adversarial fine-tuning that filters out 90% of harmful requests; an artificial conscience that 10% of the time may still act against human well-being; and AI watchdogs that catch AIs acting against our wishes nine times out of ten. If each lets 10% through and they fail independently, together they let 0.1 × 0.1 × 0.1 = 0.001 through: three nines. The textbook does not multiply them out, and does not say whether they fail independently.
    """)
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### When safeguards are not independent

    Multiplying is only allowed for independent events. NASA's guide to probabilistic risk assessment (2011) defines them as follows: "Two events, A and B, are said to be independent if the knowledge that B is TRUE does not affect our probability of A". With conditional probability: P(A | B) = P(A). The chance that both happen, P(A and B), is P(B) times P(A | B), the share of those cases in which A happens too; so independence makes P(A and B) = P(A) × P(B). That is why fault trees try to use independent events, "so that the probability of the joint occurrence of two basic events can be quantified simply as the product of the two basic event probabilities."

    Real safeguards often miss together. NASA calls that a common cause failure: "the failure (or unavailable state) of more than one component due to a shared cause during the system mission." One of its examples: "A worker accidentally tripping two redundant pumps by placing a ladder near pump motors to paint the ceiling at a nuclear power plant". And it warns: "In cases where a system provides multiple layers of defense against total system or functional failure, ignoring the effects of dependency can result in overestimation of the level of reliability." INSAG names fire, flooding and earthquakes as hazards that can strike several levels at once. And Dekker, in *The Field Guide to Human Error Investigations* (2002), as quoted in the note of 2006: "The layers of defence are not static or constant, and not independent of each other either. They can interact, support or erode one another." For AI, the same request can slip past the filter and the model behind it, when both share a blind spot, such as requests in a language that neither handles well: then their holes line up.

    ### Fault trees

    "Fault tree analysis (FTA) is a top-down process that starts by considering specific negative outcomes", the textbook writes, and works backward to their causes. That is **fault tree analysis**. At the top stands the unwanted event, the top event. Below it, gates join the causes. In NASA's words, an OR gate means "that the top event occurs if any of the inputs occur", and an AND gate "that both inputs must be satisfied in order for its output condition to occur."

    With independent events, the gates have simple rules. Under an AND gate, the chances multiply. At an OR gate, the event above does not happen only when none of the events below happen: for two independent events, P(A or B) = 1 − (1 − P(A)) × (1 − P(B)). Simply adding the chances is the rare-event approximation: close when the chances are small, but "the probability of the top event is overestimated by this approximation". In NASA's example of a system that fails when at least two of its three parts fail, each with chance 0.1, the exact answer is 0.028 and the approximation 0.030. That 0.028 is the same chance as for three different models that vote in the previous lesson, each wrong 10% of the time: a vote is a small fault tree.

    ### What the cheese leaves out

    The textbook is frank about these methods: "Within the field of safety engineering, these approaches are largely considered overly simplistic and outdated." They follow a chain of failures back from an accident, and some accidents have no failed part. The Mars Polar Lander switched off its thrusters before it had landed, because the sensors on its legs took the legs unfolding for a landing, and it crashed: "In this case, there was no component failure." Reason, Hollnagel and Paries say of the model that it "illustrates how an accident could happen, but not how it must happen."
    """)
    return


@app.cell(hide_code=True)
def text_words():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **Swiss cheese model** | safeguards pictured as slices of cheese with holes: harm gets through where the holes of all slices line up |
        | **defense in depth** | several layers of safeguards, so that no single layer is relied on alone |
        | **fault tree analysis** | working back from one unwanted event to the failures that can cause it, joined by AND and OR gates |
        | independent | knowing that one event happened does not change the chance of the other: P(A \| B) = P(A) |
        | common cause failure | several parts that fail together through one shared cause |
        | top event | the unwanted event at the top of a fault tree |
        | AND gate | the event above happens only when all events below it happen |
        | OR gate | the event above happens when at least one event below it happens |
        | `and` | Python's test that two things are both true: `"x" in a and "x" in b` |
        """),
        L.watch_out("Multiply the chances of safeguards only when they miss independently. A shared cause, the same trick or the same blind spot, makes their holes line up, and then three layers can be worth little more than one."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Name the top event: the harm you want to keep out, such as a harmful answer that reaches a user.
    2. List the safeguards in its way, with the chance that each one misses.
    3. Ask whether two safeguards can miss for the same reason: the same examples, the same blind spot, the same trick.
    4. Draw the fault tree: an AND gate where every safeguard must miss, an OR gate where one cause is enough.
    5. Multiply under an AND gate; at an OR gate, take one minus the chance that nothing below it happens. Both only for independent events: put a shared cause in its own branch.
    6. Close holes as well as adding slices: a shared cause sets a floor that more layers do not lower.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The chatbot of *Why this matters*, with its three safeguards in a row, gets 1,000,000 harmful requests. Each safeguard misses 10% of what reaches it, independently of the ones before it. The cell follows the requests through the row.
    """)
    return


@app.cell
def example_layers():
    safeguards = ["input filter", "refusing model", "output checker"]
    left = 1000000                                   # harmful requests that come in
    for safeguard in safeguards:
        left = left * 0.1                            # only the 10% it misses goes on
        print(safeguard + ":", round(left), "get through")
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Each safeguard divides what reaches it by ten: 1,000 of 1,000,000 get through: three nines, one extra nine for each layer, as the textbook suggests. That holds only because each safeguard misses independently of the ones before it. In the exercises you find the holes that line up in three safeguards, and compute a fault tree with a shared trick.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_holes():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. gets_through: holes that line up

        A safeguard's holes are the kinds of trick it misses, such as `"role play"` or `"base64"`. Write `gets_through(first, second, third)`. It gets the holes of three safeguards in a row as three lists, and returns a list `through` with the holes that are in all three lists, in the order of `first`: a request with such a trick gets through every safeguard. Go through `first` with a `for` loop, with `hole` for each hole. Replace the `...`. The check calls `gets_through(["role play", "base64", "prefix"], ["base64", "role play", "many-shot"], ["role play", "other language"])`, then other safeguards.
        """),
        L.task("The holes that line up through three slices."),
    ])
    return


@app.cell(hide_code=True)
def exercise_holes():
    box_holes = L.code_box("gets_through", lines=6)
    box_holes
    return (box_holes,)


@app.cell(hide_code=True)
def check_holes(box_holes):
    result_holes = L.check("gets_through", box_holes.value)
    result_holes
    return (result_holes,)


@app.cell(hide_code=True)
def hints_holes():
    L.hints(
        "gets_through",
        [
            "The Swiss cheese model: harm gets through only where the holes of all slices line up. A hole of the first safeguard counts only if the second and the third safeguard have it too.",
            "Start `through` as an empty list. In a `for` loop over `first`, ask for each `hole` whether `second` has it and `third` has it; if so, add it with `append`. After the loop, return `through`.",
            "`        if hole in second ___ hole in third:`",
        ],
        why="`through = []`, then `for hole in first:` with `if hole in second and hole in third:` and `through.append(hole)` inside it; after the loop, `return through`. In the first case only `\"role play\"` is in all three lists.",
    )
    return


@app.cell(hide_code=True)
def text_after_holes():
    mo.md(r"""
    In the first case only role play gets through. Base64 is a hole in two slices, but the third safeguard catches it. A safeguard with holes is not the problem; a hole that lines up with holes in all the others is.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_tree():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. top_event: a fault tree with AND and OR

        The top event is that a harmful request gets through. It happens when every safeguard misses it, an AND gate over the safeguards, or when the request uses a shared trick that fools them all; an OR gate joins the two. Write `top_event(misses, shared)`. `misses` is a list with, for each safeguard, the chance that it misses an ordinary harmful request, independently of the others; `shared` is the chance that a request uses the shared trick, independently of the rest. The list can be empty: a tree without safeguards. Compute the chance that every safeguard misses in a variable `all_miss`, with a `for` loop and `miss` for each chance, and return the chance of the top event. Replace the `...`. The check calls `top_event([0.1, 0.1, 0.1], 0.01)`, then other trees.
        """),
        L.task("The chance of the top event of a fault tree."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tree():
    box_tree = L.code_box("top_event", lines=5)
    box_tree
    return (box_tree,)


@app.cell(hide_code=True)
def check_tree(box_tree):
    result_tree = L.check("top_event", box_tree.value)
    result_tree
    return (result_tree,)


@app.cell(hide_code=True)
def hints_tree():
    L.hints(
        "top_event",
        [
            "Two gates. Under the AND gate every safeguard must miss, so with independent safeguards the chances multiply. At the OR gate, the top event does not happen only when neither branch happens.",
            "Start `all_miss` at 1 and multiply it by each `miss` in a `for` loop. For the OR gate: the chance that neither branch happens is the chance of no shared trick times the chance that not every safeguard misses; return one minus that.",
            "`    return 1 - (1 - shared) * ___`",
        ],
        why="`all_miss = 1`, then `for miss in misses:` with `all_miss = all_miss * miss`; then `return 1 - (1 - shared) * (1 - all_miss)`. For `[0.1, 0.1, 0.1]` and `0.01`: `all_miss` is 0.001, and the top event 1 − 0.99 × 0.999 = 0.01099.",
    )
    return


@app.cell(hide_code=True)
def text_after_tree():
    mo.md(r"""
    With three safeguards that each miss 10% and no shared trick, 0.001 gets through; with a shared trick in 1% of the requests, 0.01099, almost eleven times as much. The shared trick is a common cause: one reason that makes every safeguard miss at once.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: more safeguards in a row

    Each safeguard misses 10% of the ordinary harmful requests, independently of the others. With the shared trick, 1% of the requests fool every safeguard at once. Choose how many safeguards stand in a row, and compare how many of 1,000,000 harmful requests get through, and how many nines that is, with and without the shared trick.
    """)
    return


@app.cell(hide_code=True)
def demo_layers():
    layers = L.explore("layers", "safeguards in a row", lambda _v: C.demo_frame(_v, L.lang),
                       start=1, values=C.LAYERS, role="",
                       task="Find the smallest number of safeguards on the slider at which the shared trick lets more than ten times as many harmful requests through as the safeguards let through without it.")
    layers
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    From three safeguards on: 10,990 against 1,000. Without the shared trick, each extra safeguard adds a nine. With it, the nines stay just under two, because the 10,000 requests with the trick get through however many safeguards stand in a row; five safeguards still let 10,010 through. What helps then is not a sixth slice, but finding the trick and closing that hole in at least one slice: reducing the holes, not only adding slices.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_common_cause():
    mo.md(r"""
    ### Exercise 3. Three classifiers

    A company stacks three safeguards against harmful requests: three classifiers, all trained on the same set of examples of harmful requests. Each one stops 90% of the harmful requests in a test.
    """)
    return


@app.cell(hide_code=True)
def choice_common_cause_cell():
    choice_common_cause = L.choice(
        "common_cause",
        question="Why can the three together stop much less than the 99.9% that multiplying promises?",
        options={
            "slow": "Three classifiers make every answer slower, so users switch them off",
            "right": "A kind of request their examples never showed slips past all three at once",
            "two": "Multiplying chances only works for exactly two safeguards in a row, never for three",
        },
        correct="right",
        why={
            "right": "The three learned from the same examples, so they share blind spots: their holes line up. They do not miss independently, and then the product overestimates how much they stop.",
            "slow": "Speed can be a real cost, but it is not why the numbers do not multiply. The classifiers learned from the same examples, so they miss the same requests. See *When safeguards are not independent* in *The idea*.",
            "two": "Multiplying works for any number of independent safeguards: 0.1 × 0.1 × 0.1 = 0.001. These three are not independent, because they learned from the same examples. See *When safeguards are not independent* in *The idea*.",
        },
    )
    choice_common_cause
    return (choice_common_cause,)


@app.cell(hide_code=True)
def text_exercise_missing_layer():
    mo.md(r"""
    ### Exercise 4. One layer down

    A chatbot has four safeguards in a row. One morning, the checker on the outgoing answers stops working. The team lead says: we still have three safeguards, so we keep the chatbot running until the checker is fixed.
    """)
    return


@app.cell(hide_code=True)
def choice_missing_layer_cell():
    choice_missing_layer = L.choice(
        "missing_layer",
        question="What does defense in depth say about this?",
        options={
            "nines": "Running is safe: three safeguards that each stop 90% still give three nines",
            "right": "The other layers do not justify running without the missing one",
            "logs": "Running is fine as long as someone checks the logs afterwards",
        },
        correct="right",
        why={
            "right": "As INSAG, the nuclear safety advisers, puts it: several elements of defense in depth do not justify continued operation in the absence of one element. The design counted on all four layers; with one gone, a request that slips through the other three now reaches the user.",
            "nines": "That counts on the other three missing independently and on the 90% being right; and the design counted on four layers, not three. See *Defense in depth* in *The idea*.",
            "logs": "Logs help to find out afterwards what went wrong, but they stop nothing while the chatbot runs. Defense in depth says that the other layers do not justify running without the missing one. See *Defense in depth* in *The idea*.",
        },
    )
    choice_missing_layer
    return (choice_missing_layer,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Designing for safety](/course/open/rai-safe-design-principles?lang=en) gave redundancy, with the warning that copies fail together, and [Conditional probability](/course/open/prob-conditional?lang=en) gave P(A | B). This lesson put safeguards in a row: the Swiss cheese model, defense in depth, and fault trees that multiply under AND and combine under OR, valid only when the safeguards miss independently. The next lesson, *Drifting into failure, and black swans*, looks at what these models leave out: accidents without a failed part, and rare events that no tree foresaw.

    **An everyday example.** An office has a badge reader at the front door, a receptionist and a locked server room: three layers. But if one stolen badge opens both the front door and the server room, two of the three share a hole.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_common_cause, choice_missing_layer, result_holes, result_tree):
    L.wrap_up(result_holes, result_tree, choice_common_cause, choice_missing_layer)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
