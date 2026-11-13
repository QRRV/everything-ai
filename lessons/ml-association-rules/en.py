# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-association-rules. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-association-rules", "en")
    C = L.checks   # the 19,773 orders of the shop, Apriori on them for every min support, the rules, and the picture


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Conditional probability](/course/open/prob-conditional?lang=en) and one about [One number per group: groupby](/course/open/data-groupby?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_conditional():
    quick_conditional = L.choice(
        "quick-conditional",
        question="Of 200 orders, 40 hold tea, and 30 of those 40 also hold cake. What is P(cake | tea)?",
        options={
            "right": "0.75",
            "both": "0.15",
            "tea": "0.2",
        },
        correct="right",
        why={
            "both": "0.15 is 30 / 200: the share of all orders that hold tea and cake both. P(cake | tea) looks only at the 40 orders with tea. See *The idea* in [Conditional probability](/course/open/prob-conditional?lang=en).",
            "tea": "0.2 is 40 / 200: P(tea). P(cake | tea) is the share of the orders with tea that hold cake too. See *The idea* in [Conditional probability](/course/open/prob-conditional?lang=en).",
            "right": "30 / 40. Today that number gets a name of its own: the confidence of the rule tea → cake.",
        },
    )
    quick_conditional
    return (quick_conditional,)


@app.cell(hide_code=True)
def quick_check_groupby():
    quick_groupby = L.choice(
        "quick-groupby",
        question="`df` has a row for every product in an order, with the columns `invoice`, the order, and `product`. What does `df.groupby(\"invoice\")[\"product\"].count()` give?",
        options={
            "right": "For every order, how many products it holds",
            "product": "For every product, how many orders hold it",
            "total": "How many orders there are",
        },
        correct="right",
        why={
            "product": "`groupby(\"invoice\")` makes a group per order, not per product. For every product, you would group by `\"product\"`. See *The idea* in [One number per group: groupby](/course/open/data-groupby?lang=en).",
            "total": "`groupby` gives one number per group, so a number for every order. How many orders there are is the number of groups. See *The idea* in [One number per group: groupby](/course/open/data-groupby?lang=en).",
            "right": "One number per order. Today `groupby` makes one set of products per order.",
        },
    )
    quick_groupby
    return (quick_groupby,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A British online shop that sells gifts, many of them to other shops, had 19,773 orders from December 2010 to December 2011, with 3,765 different products. Which products end up in the same order? A shop that knows can put them side by side, suggest the second when the first goes into the basket, or sell them as a set. But 3,765 products make 7,085,730 pairs, and 3,658,456 of them were bought together at least once: far too many to look at one by one.

    **An everyday example.** A supermarket puts the salsa next to the tortilla chips. Not because they are alike, but because people who take the one often take the other.
    """)
    return


@app.cell(hide_code=True)
def predict_together():
    guess_together = L.guess(
        "together-guess",
        question="The GREEN REGENCY TEACUP AND SAUCER, a teacup with a saucer in a green pattern, is in 1,013 of the 19,773 orders. The same cup in a roses pattern, the ROSES REGENCY TEACUP AND SAUCER, is in 1,065: 5.4% of all orders. Of the 1,013 orders with the green cup, how many also hold the roses cup?",
        options={
            "chance": "About 5%, as in all orders",
            "some": "About 25%",
            "right": "About 75%",
        },
        reveal="**About 75%.** 767 of the 1,013: 75.7%, against 5.4% of all orders. The roses cup is 14 times as common in an order with the green cup as in any order.",
    )
    guess_together
    return (guess_together,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **From orders to rules.** Every order is a set of products. Three numbers say how products go together. Below, GREEN is the green teacup and ROSES the roses teacup.

        - The **support** of a set of products is the share of all orders that hold every product of the set. GREEN is in 1,013 of the 19,773 orders: a support of 0.051. GREEN and ROSES together are in 767: 0.039.
        - An **association rule** A → B says that orders with A often hold B too. A and B are each one product or a set of them.
        - The **confidence** of A → B is the support of A and B together divided by the support of A: of the orders with A, the share that holds B too. That is the conditional probability P(B | A). GREEN → ROSES: 767 / 1,013 = 0.757.
        - The **lift** of A → B is its confidence divided by the support of B: how many times as often B is in an order with A as in any order. ROSES is in 5.4% of all orders and in 75.7% of the orders with GREEN, so GREEN → ROSES has a lift of 14.1. A lift of 1 means that A makes no difference to B; below 1, B turns up less often with A.

        A **frequent itemset** is a set of products whose support is at least a minimum you choose, the min support. Rules are only worth making for frequent sets: a set in two orders says little.

        **Apriori.** Every order with GREEN and ROSES also holds GREEN, so the support of a set is never larger than that of any part of it. A set can only be frequent if all its parts are. **Apriori** uses that: it counts the single products and keeps the frequent ones, makes pairs of those only and keeps the frequent pairs, makes sets of three only when all three of their pairs are frequent, and so on, until no set is frequent any more.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **support** | | the share of all orders that hold every product of a set | `count / len(orders)` |
        | **association rule** | | a rule that says orders with A often hold B too | |
        | A → B | if A, then B | the rule with A on the left and B on the right | |
        | **confidence** | | the support of A and B together divided by the support of A: P(B \| A) | `n_both / n_a` |
        | **lift** | | the confidence divided by the support of B | `confidence / (n_b / n)` |
        | **frequent itemset** | | a set of products whose support is at least the min support | |
        | **Apriori** | | finding the frequent itemsets by making larger sets only of frequent smaller ones | |
        | `items <= order` | | on two sets: is every value of `items` also in `order`? | `{"a"} <= {"a", "b"}` is `True` |
        """),
        L.watch_out("Lift says nothing about how often. 125 products were bought in one order only, and ten pairs of them in the same order: each such pair has a lift of 19,773, the highest a lift can be here, from one order. That is why rules come only from frequent sets. And confidence alone misleads when B is in many orders anyway: JUMBO BAG RED RETROSPOT → WHITE HANGING HEART T-LIGHT HOLDER has a confidence of 0.214, but the holder is in 11.5% of all orders, so the lift is only 1.87."),
        mo.md(r"""
        The picture runs Apriori on the 19,773 orders. On the left, for single products, pairs and sets of three: how many sets Apriori counts, the open bars, and how many of them are frequent, the filled bars, with the number. The dashed line over the pairs is all 7,085,730 pairs. Every line up is ten times the one below. On the right, every rule of the frequent pairs, two per pair: its confidence and its lift, the lift also on that scale. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_support_cell():
    def _frame(ms):
        said = C.said(ms, L.lang)
        alt = ("Left, a bar chart on a log scale with open bars for the sets Apriori counts and filled bars for the frequent ones, "
               "for single products, pairs and sets of three, with a dashed line at all 7,085,730 pairs. Right, the rules of the "
               "frequent pairs as dots, confidence from 0 to 1 against lift from 1 to 100 on a log scale. " + said)
        return (C.picture(ms, {"left": f"Apriori, min support {C.short(ms, L.lang)}", "right": "Rules of the frequent pairs", "sets": "sets",
                               "sizes": ["products", "pairs", "sets of 3"], "lang": L.lang, "alt": alt}), said)

    explore_support = L.explore("min-support", "min support", _frame, start=0.02, values=C.SUPPORTS, role="hyper",
                                task="Go from min support 0.01 to 0.04. How many pairs does Apriori count at 0.02, and from which min support on is no set of three frequent?")
    explore_support
    return


@app.cell(hide_code=True)
def text_after_support():
    mo.md(r"""
    At min support 0.01, 830 products are frequent, and Apriori counts 344,035 pairs of them instead of 7,085,730; 910 are frequent. At 0.02 it counts 45,451 pairs, keeps 90, and makes only 67 sets of three, of which 3 are frequent, such as the green, the roses and the pink regency teacup together. From 0.03 on, no set of three is frequent; at 0.04 one pair is left, JUMBO BAG PINK POLKADOT and JUMBO BAG RED RETROSPOT. A lower min support finds more rules, with lifts up to 76.83 at 0.01, but also more sets that just a few orders make.

    ### Recipe

    To find which products are bought together:

    1. Make every order a set of products.
    2. Choose a min support, and find the frequent sets with Apriori: single products, then pairs of frequent products, then sets of three whose pairs are all frequent.
    3. For every frequent pair, make both rules, A → B and B → A, with their confidence and lift.
    4. Keep the rules with a high confidence and a lift well above 1, and look at a few of their orders yourself.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The rule GREEN → ROSES, counted by hand. The cell reads the orders, makes a set of products for every order with `groupby("invoice")["product"].agg(set)`: besides a name such as `"mean"`, `agg` takes a function, and `set` makes a set of each group's values. It counts the orders with GREEN, with ROSES, and with both. It prints the number of orders and those three counts, then the support of GREEN, the confidence and the lift. The comments with `#` say which step of the recipe each part is. Run the cell.
    """)
    return


@app.cell
def example_rule():
    import pandas as pd

    # 1. every order as a set of products
    _df = pd.read_csv("data/online-retail-baskets.csv.gz")
    _orders = list(_df.groupby("invoice")["product"].agg(set))

    # 3. the rule GREEN -> ROSES: count the orders with A, with B, and with both
    _a = "GREEN REGENCY TEACUP AND SAUCER"
    _b = "ROSES REGENCY TEACUP AND SAUCER"
    _n_a = 0
    _n_b = 0
    _n_both = 0
    for _order in _orders:
        if _a in _order:
            _n_a = _n_a + 1
        if _b in _order:
            _n_b = _n_b + 1
        if _a in _order and _b in _order:
            _n_both = _n_both + 1

    _n = len(_orders)
    _confidence = _n_both / _n_a
    _lift = _confidence / (_n_b / _n)
    print(_n, _n_a, _n_b, _n_both)
    print(round(_n_a / _n, 3), round(_confidence, 3), round(_lift, 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    `agg(set)` gives one set per group: here the products of every order. 19,773 orders; GREEN is in 1,013, ROSES in 1,065, and both in 767. The support of GREEN is 0.051, the confidence of GREEN → ROSES 0.757, and the lift 14.06: the numbers of *The idea*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_support():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Support

        `support(orders, items)` should give back the support of the set `items`: the share of the orders that hold every product of it. `orders` is a list of sets. Replace the two `...`: when an order counts, and what the function gives back. The check calls `support(orders, both)` with six small orders and `both` the GREEN TEACUP and the ROSES TEACUP, and tries four more: one product, a set in no order, three products, and eight other orders.
        """),
        L.task("Make `support` give the share of the orders that hold a set."),
    ])
    return


@app.cell(hide_code=True)
def exercise_support():
    box_support = L.code_box("support", lines=7)
    box_support
    return (box_support,)


@app.cell(hide_code=True)
def check_support(box_support):
    result_support = L.check("support", box_support.value)
    result_support
    return (result_support,)


@app.cell(hide_code=True)
def hints_support():
    L.hints(
        "support",
        [
            "An order counts when every product of `items` is in it. The support is then a share: of all orders, how many counted?",
            "The New words table has a comparison that asks whether one set lies inside another. The share is the count divided by the number of orders.",
            "`if items ___ order:` and `return count / ___`.",
        ],
        why="Of the six orders, the first and the fourth hold both the GREEN TEACUP and the ROSES TEACUP: 2 of 6, a support of 0.333.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_rule():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Confidence and lift

        `rule(n_a, n_b, n_both, n)` should give back the confidence and the lift of A → B, from four counts: the orders with A, the orders with B, the orders with both, and all orders. Replace the two `...`. The check calls `rule(1013, 1065, 767, 19773)`, the counts of GREEN → ROSES, and tries four more: a rule with a lift of 2.5, one with a lift below 1, one with a lift of exactly 1, and one whose confidence is 1.
        """),
        L.task("Make `rule` give the confidence and the lift of a rule."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rule():
    box_rule = L.code_box("rule", lines=5)
    box_rule
    return (box_rule,)


@app.cell(hide_code=True)
def check_rule(box_rule):
    result_rule = L.check("rule", box_rule.value)
    result_rule
    return (result_rule,)


@app.cell(hide_code=True)
def hints_rule():
    L.hints(
        "rule",
        [
            "The confidence is the share of the orders with A that hold B too. The lift is the confidence divided by the support of B.",
            "Of the `n_a` orders with A, how many hold B too? And the support of B is the orders with B as a share of all orders.",
            "`confidence = n_both / ___` and `lift = confidence / (___ / n)`.",
        ],
        why="The confidence is 767 / 1,013 = 0.757. The support of B is 1,065 / 19,773 = 0.0539, and the confidence divided by that is 14.06.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_apriori():
    mo.md(r"""
    ### Exercise 3. Why Apriori may skip sets
    """)
    return


@app.cell(hide_code=True)
def choice_apriori_cell():
    choice_apriori = L.choice(
        "apriori",
        question="The green, the roses and the pink regency teacup are together in 541 orders: a support of 0.027. What do you know about the support of GREEN and ROSES together, without counting?",
        options={
            "right": "It is at least 0.027: every order with all three also holds those two",
            "less": "It is at most 0.027: fewer orders hold a pair than a set of three",
            "unknown": "Nothing: you have to count it",
        },
        correct="right",
        why={
            "less": "It is the other way round: a pair asks less of an order than a set of three, so at least as many orders hold it. GREEN and ROSES are in 767 orders, the three teacups in 541.",
            "unknown": "Every order with all three teacups also holds GREEN and ROSES, so the pair is in at least those 541 orders, a support of 0.027, before you count. Counting gives 767.",
            "right": "The pair is in at least those 541 orders; counting gives 767, a support of 0.039. Apriori uses this the other way round: a set with a part that is not frequent cannot be frequent itself, so it never counts it.",
        },
    )
    choice_apriori
    return (choice_apriori,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The confidence of a rule is the conditional probability of [Conditional probability](/course/open/prob-conditional?lang=en), and the lift compares it with the probability of B on its own. The orders are the sets of [Tuples, sets and sorting](/course/open/py-tuples-sets?lang=en), made with the `groupby` of [One number per group: groupby](/course/open/data-groupby?lang=en). A rule says what goes together, not why, and not what one customer wants: the lessons on recommending start from these same orders, with the customers who placed them. In the next lesson, *Time as an index: dates and resampling in pandas*, data gets a clock.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_apriori, result_rule, result_support):
    L.wrap_up(result_support, result_rule, choice_apriori)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
