# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-recommend-problem. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-recommend-problem", "en")
    C = L.checks   # the orders split by time, the popularity baseline and picking at random for every k, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Always beat a simple baseline](/course/open/ml-baselines?lang=en) and one about [Project: your table, explored and cleaned](/course/open/ml-project-table-explore?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_baseline():
    quick_baseline = L.choice(
        "quick-baseline",
        question="In a test set, 98 of 100 patients are healthy. A model that spots a rare disease is right about 97 of the 100. What do you say?",
        options={
            "right": "It does worse than always saying healthy, which is right about 98",
            "good": "It is good: 97% right",
            "train": "Measure it on the training set instead",
        },
        correct="right",
        why={
            "good": "97% sounds high, but saying healthy to everyone, without looking, is right about 98. A score means something only next to its baseline. See *The idea* in [Always beat a simple baseline](/course/open/ml-baselines?lang=en).",
            "train": "A model is measured on the test set, and so is its baseline: the same set, the same metric. See *Recipe* in [Always beat a simple baseline](/course/open/ml-baselines?lang=en).",
            "right": "The model does not beat the baseline of the most common class. Today the baseline recommends what everyone buys.",
        },
    )
    quick_baseline
    return (quick_baseline,)


@app.cell(hide_code=True)
def quick_check_duplicates():
    quick_duplicates = L.choice(
        "quick-duplicates",
        question="`df` has a row for every product of every order, with the columns `invoice`, `customer` and `product`. A customer who bought the same mug in three orders has three rows for it. What gives one row for every customer and product?",
        options={
            "right": "`df.drop_duplicates(subset=[\"customer\", \"product\"])`",
            "all": "`df.drop_duplicates()`",
            "product": "`df.drop_duplicates(subset=\"product\")`",
        },
        correct="right",
        why={
            "all": "Without a column, `drop_duplicates` drops only rows that are the same in every column, and the three rows of the mug have three different invoices. See *The idea* in [Project: your table, explored and cleaned](/course/open/ml-project-table-explore?lang=en).",
            "product": "That keeps one row for every product: the mug once, for one customer only. See the table of new words in [Project: your table, explored and cleaned](/course/open/ml-project-table-explore?lang=en).",
            "right": "Of the rows with the same customer and product, only the first stays. Today those rows are the cells of a matrix.",
        },
    )
    quick_duplicates
    return (quick_duplicates,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A British online shop that sells gifts, many of them to other shops, had 3,633 different products and 4,334 customers whose orders it knows, from December 2010 to December 2011. A page or an email has room for ten products. Which ten should each customer see? A shop, a streaming service or a news site answers that question for every visitor, many times a day. And it cannot ask the customer whether the answer was good: it can only wait and see what they do next.

    **An everyday example.** A bookseller who knows you puts three books aside for your next visit. If they are the three best sellers of the week, they will often be right, and you may have read them already.
    """)
    return


@app.cell(hide_code=True)
def predict_popular():
    guess_popular = L.guess(
        "popular-guess",
        question="Give every customer the 10 most popular products they have not bought yet. Then look at their next order. For how many customers is at least one of the ten in it?",
        options={
            "few": "About 1 in 25",
            "right": "About 1 in 5",
            "half": "About half",
        },
        reveal="**About 1 in 5.** 541 of 2,498 customers: 21.7%. Ten products picked at random from the ones a customer has not bought find something for 4.2%. Popularity is five times better, and still finds nothing for four customers out of five. That is the baseline: a recommender that does not beat it has learned nothing about the customer.",
    )
    guess_popular
    return (guess_popular,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Users, items and interactions.** A **recommender system** chooses, for every user, which items to show: here customers and products. One customer buying one product is an **interaction**. Put them in a **user-item matrix**: a row for every customer, a column for every product, and a 1 where the customer bought the product. This one has 4,334 × 3,633 = 15,745,422 cells, and 266,035 of them are 1: 1.69%. A matrix that is almost all 0 is **sparse**, and how empty it is, its **sparsity**. A recommender guesses which of the empty cells will be filled next. A set of products for every customer is the same matrix with only its 1s: 266,035 values instead of almost 16 million.

        **Implicit feedback.** A purchase is **implicit feedback**: the customer did not say what they think of a product, they bought it. A 1 says something; a 0 says little. The customer may not know the product, may have bought it elsewhere, or may not need one. Stars that a customer gives a product say more, but few customers give them.

        **Test on what comes later.** Split by time: every customer's last order is the test, and the orders before it are all a recommender may use. Recommend `k` products the customer has not bought before, and look at the products of the last order that are new to them. The **hit rate** at `k` is the share of customers for whom at least one of the `k` products is among those. 2,829 customers have an order before their last, and for 2,498 of them the last order holds at least one new product: they are the test.

        **The popularity baseline.** The simplest recommender counts, for every product, how many customers bought it before their last order, and gives every customer the most popular products they have not bought yet. Everyone gets almost the same list. That is the **popularity baseline**: a recommender that knows something about the customer should beat it.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **recommender system** | a system that chooses which items to show each user | |
        | **interaction** | one thing a user did with an item, here a purchase | a row of the table |
        | **user-item matrix** | a row for every user, a column for every item, and a 1 for every interaction | |
        | **sparsity** | how empty a matrix is: here 98.31% of the cells are 0 | `1 - len(pairs) / (customers * products)` |
        | **implicit feedback** | what users do, such as buying, instead of what they say they like | |
        | **hit rate** at `k` | the share of users for whom at least one of `k` recommended items turns up later | |
        | **popularity baseline** | the most popular items a user has not used yet, the same order for everyone | |
        | `transform("max")` | after `groupby`, gives every row the maximum of its group, so the table keeps its rows | `df.groupby("customer")["invoice"].transform("max")` |
        | `counts.index` | of what `value_counts` gives: the values that were counted, the most frequent first | `counts.index[0]` |
        """),
        L.watch_out("Never count on the orders you test on. Count the popularity on all orders, the last ones included, and the hit rate at 10 says 26.1% instead of 21.7%: the counts then hold the very orders the test hides, and the recommendations only look better. Split first, then count on the orders before the split."),
        mo.md(r"""
        The picture shows a corner of the matrix: 60 of the test customers, a row each, and the 120 most popular products, a column each, most popular on the left. A filled dot is a product the customer bought before their last order. An open dot is a product of their last order that they had not bought before: what the test hides. Of these 7,200 cells, 647 are filled, 9.0%, against 1.69% in the whole matrix: these are the products bought most often. 132 are open.
        """),
        mo.Html(C.matrix({"title": "60 customers, 120 products", "xlabel": "product, most popular first", "ylabel": "customer",
                          "alt": "A grid of dots: 60 customers as rows against the 120 most popular products as columns. Filled dots, 647, are products a customer bought before their last order; open dots, 132, are new products of their last order. The left columns, the most popular products, hold the most dots."})),
    ])
    return


@app.cell(hide_code=True)
def text_before_slider():
    mo.md(r"""
    The slider sets `k`, how many products every customer gets. The line with dots is the hit rate of the popularity baseline over the 2,498 test customers; the dashed line with open dots is picking `k` products at random from the ones a customer has not bought, as an average over every way to pick them. `k` runs on a log scale: 1, 5, 10, 20, 50. Move the slider.
    """)
    return


@app.cell(hide_code=True)
def explore_k_cell():
    def _frame(k):
        said = C.said(k, L.lang)
        alt = ("A line chart of the hit rate in percent against k on a log scale, from 1 to 50: the popularity baseline as "
               "filled dots on a line, picking at random as open dots on a dashed line, with a dashed mark at the k of the slider. " + said)
        return (C.picture(k, {"title": "Hit rate over 2,498 customers", "ylabel": "hit rate (%)", "popular": "popularity",
                              "random": "at random", "alt": alt}), said)

    explore_k = L.explore("recommend-k", "k", _frame, start=10, values=C.KS, role="hyper",
                          task="Go from k = 1 to k = 50. How many times as often as picking at random does popularity find something at k = 1, and at k = 50?")
    explore_k
    return


@app.cell(hide_code=True)
def text_after_slider():
    mo.md(r"""
    With one product each, popularity finds something for 95 customers, 3.8%: about nine times as often as picking at random, 0.4%. With 10, for 21.7%; with 50, for 52.0%, three times as often as picking at random, 17.7%. A longer list finds more, but a page shows only a few products, and a customer reads fewer. Popularity is hard to beat: the 120 most popular products are 3.4% of the 3,576, but 14.2% of the new products in the last orders. But it gives everyone almost the same list, and that is where a recommender that looks at the customer can win.

    ### Recipe

    To build a test for recommendations and its baseline:

    1. Make a table with a row for every interaction: here every product of every order, with its customer.
    2. Split by time: every user's last order is the test, the rest is what a recommender may use. Keep the users with something new in their last order.
    3. Count how many users had each item before the split: `drop_duplicates` on user and item, then `value_counts`.
    4. For every user, take the `k` most popular items they have not had.
    5. Measure the hit rate at `k`, and compare every recommender with this baseline on the same users and the same `k`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell follows steps 1 to 3. It reads the orders, keeps the rows with a customer, and counts the customers, the products and the cells with a 1. `transform("max")` puts the number of the customer's last order on every row of that customer; the invoices count up over time, so the largest is the last. `agg(set)` gives a set for every customer: `seen`, what they bought before their last order, and the products of the last order. Like a dictionary, what it gives has `items()`, and `seen[customer]` is the set of one customer. Then it keeps, for every customer, the products of the last order that are not in `seen`, and counts the popularity of every product. The customer numbers are floats, such as `14905.0`, because the column had empty cells before `dropna`, and pandas keeps such a column as floats. Last, for customer 14905, it prints for each of the 12 most popular products whether they bought it before and whether it is in their last order. Run the cell.
    """)
    return


@app.cell
def example_split():
    import pandas as pd

    # 1. every product of every order with a known customer, and the size of the user-item matrix
    _rows = pd.read_csv("data/online-retail-baskets.csv.gz").dropna(subset=["customer"])
    _pairs = _rows.drop_duplicates(subset=["customer", "product"])
    _customers = len(set(_pairs["customer"]))
    _products = len(set(_pairs["product"]))
    print(_customers, _products, len(_pairs), round(len(_pairs) / (_customers * _products), 4))

    # 2. split by time: the last order of every customer is the test, the orders before it what a recommender may use
    _rows["last"] = _rows.groupby("customer")["invoice"].transform("max")
    _train = _rows[_rows["invoice"] < _rows["last"]]
    _test = _rows[_rows["invoice"] == _rows["last"]]
    _seen = _train.groupby("customer")["product"].agg(set)
    _last = _test.groupby("customer")["product"].agg(set)
    _hidden = {}
    for _customer, _bought in _seen.items():
        _new = []
        for _product in _last[_customer]:
            if _product not in _bought:
                _new.append(_product)
        if len(_new) > 0:
            _hidden[_customer] = set(_new)
    print(len(_seen), len(_hidden))

    # 3. the popularity of every product: how many customers bought it before their last order
    _counts = _train.drop_duplicates(subset=["customer", "product"])["product"].value_counts()
    print(len(_counts), _counts.index[0], _counts.iloc[0])

    # customer 14905: of the 12 most popular products, bought before? in the last order?
    for _product in _counts.index[:12]:
        print(_product in _seen[14905.0], _product in _hidden[14905.0], _product)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    4,334 customers and 3,633 products; 266,035 cells hold a 1, a share of 0.0169. 2,829 customers have an order before their last, and 2,498 of them bought something new in it. 3,576 products were bought before a last order; the REGENCY CAKESTAND 3 TIER by the most customers, 663. Customer 14905 placed two orders. Of the 12 most popular products they bought the second and the third before, the WHITE HANGING HEART T-LIGHT HOLDER and the PARTY BUNTING, so their ten run on to the 12th. Two of the ten are in their last order: the JUMBO BAG RED RETROSPOT, 6th, and the HEART OF WICKER SMALL, 12th. For this customer, the baseline hits.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_popular():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The most popular, not yet bought

        `popular(counts, seen, k)` should give the list of the `k` most popular products that are not in `seen`, the most popular first. `counts` is what `value_counts` gives: `counts.index` holds the products, the most bought first. Replace the `...`: when a product goes into `top`. The check calls `popular(counts, seen, 10)` with the counts of the worked example and `seen` the products customer 14905 bought before their last order, and tries four more on a shop of eight products.
        """),
        L.task("Make `popular` give the k most popular products a customer has not bought."),
    ])
    return


@app.cell(hide_code=True)
def exercise_popular():
    box_popular = L.code_box("popular", lines=6)
    box_popular
    return (box_popular,)


@app.cell(hide_code=True)
def check_popular(box_popular):
    result_popular = L.check("popular", box_popular.value)
    result_popular
    return (result_popular,)


@app.cell(hide_code=True)
def hints_popular():
    L.hints(
        "popular",
        [
            "A product goes into `top` when two things are both true: one is about the customer, one about how long `top` is already.",
            "The customer must not have bought it, and `top` must still have room for one more. `and` joins two conditions.",
            "`if product ___ seen and len(top) ___ k:`.",
        ],
        why="`counts.index` runs from the most popular product down. The 2nd and the 3rd are in `seen`, so they are skipped, and the ten are the 1st, the 4th to the 12th.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_hit_rate():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The hit rate

        `hit_rate(recommended, hidden)` should give the share of customers for whom at least one recommended product is in their hidden products. Both are dictionaries with a customer as key: `recommended[customer]` is a list of products, `hidden[customer]` the set of new products of their last order. Replace the two `...`: when a recommended product is a hit, and what the function gives back. The check calls `hit_rate(recommended, hidden)` with the ten products of the popularity baseline for all 2,498 test customers, and tries three small dictionaries.
        """),
        L.task("Make `hit_rate` give the share of customers who find at least one product."),
    ])
    return


@app.cell(hide_code=True)
def exercise_hit_rate():
    box_hit_rate = L.code_box("hit-rate", lines=10)
    box_hit_rate
    return (box_hit_rate,)


@app.cell(hide_code=True)
def check_hit_rate(box_hit_rate):
    result_hit_rate = L.check("hit-rate", box_hit_rate.value)
    result_hit_rate
    return (result_hit_rate,)


@app.cell(hide_code=True)
def hints_hit_rate():
    L.hints(
        "hit-rate",
        [
            "A product is a hit when the customer bought it in their last order. The hit rate is a share: of all customers, how many found something?",
            "Look a product up in the set of that one customer, not in all of `hidden`. And a share is a count divided by a total: which total?",
            "`if product in ___:` and `return hits / len(___)`.",
        ],
        why="`found` turns `True` at the first hit and stays `True`, so a customer with two hits counts once. Over the 2,498 customers, 541 count: 0.217.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_implicit():
    mo.md(r"""
    ### Exercise 3. What a 0 says
    """)
    return


@app.cell(hide_code=True)
def choice_implicit_cell():
    choice_implicit = L.choice(
        "implicit",
        question="Customer 14905 never bought the REGENCY CAKESTAND 3 TIER, the most popular product. What does the 0 in their row of the user-item matrix say?",
        options={
            "right": "Little: they may not know it, may have bought one elsewhere, or may not need one",
            "dislike": "That they do not like it",
            "never": "That they will not buy it",
        },
        correct="right",
        why={
            "dislike": "A purchase is implicit feedback: it says what a customer did, not what they think. Many customers do not know most of the 3,633 products. See *Implicit feedback* in *The idea*.",
            "never": "A 0 is about the past. The open dots in the picture of the matrix are products that were a 0 before the last order and a 1 in it. See *Implicit feedback* in *The idea*.",
            "right": "With implicit feedback, a 1 says something and a 0 little. That is why a recommender is tested on what customers buy later, and not on the zeros.",
        },
    )
    choice_implicit
    return (choice_implicit,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The popularity baseline is the baseline of [Always beat a simple baseline](/course/open/ml-baselines?lang=en), and the hit rate the metric to compare on: the same customers, the same hidden orders, the same `k`. The table is cleaned with the `drop_duplicates` of [Project: your table, explored and cleaned](/course/open/ml-project-table-explore?lang=en), and the split by time keeps the test honest. Popularity gives everyone almost the same list. In the next lesson, *More of what you bought: content-based filtering*, the list looks at what a customer bought and at what each product is: the words of its name.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_implicit, result_hit_rate, result_popular):
    L.wrap_up(result_popular, result_hit_rate, choice_implicit)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
