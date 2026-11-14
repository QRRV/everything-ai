# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-collaborative-filtering. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-collaborative-filtering", "en")
    C = L.checks   # the user-item matrix, the similarities of the products, the three recommenders, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [More of what you bought: content-based filtering](/course/open/ml-content-based?lang=en) and one about [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_profile():
    quick_profile = L.choice(
        "quick-profile",
        question="Content-based filtering recommended ten hearts, nine of them wicker, to customer 13960. What did it need to know for that?",
        options={
            "right": "The words of the product names, and what the customer bought",
            "others": "What other customers bought",
            "stars": "How many stars the customer gave each product",
        },
        correct="right",
        why={
            "others": "Content-based filtering looks only at one customer's own products and at the names of all products: other customers play no part. See *The user profile* in [More of what you bought: content-based filtering](/course/open/ml-content-based?lang=en).",
            "stars": "The shop has no stars: a purchase is implicit feedback. See *Implicit feedback* in [What to show next: the recommendation problem](/course/open/ml-recommend-problem?lang=en).",
            "right": "The names give the item features, the purchases the profile. Today the names play no part: only who bought what.",
        },
    )
    quick_profile
    return (quick_profile,)


@app.cell(hide_code=True)
def quick_check_transpose():
    quick_transpose = L.choice(
        "quick-transpose",
        question="`R` has shape (2829, 3576): a row for every customer, a column for every product. `R[:, j]` is column `j`, with shape (2829,). What is the shape of `R.T @ R[:, j]`?",
        options={
            "right": "(3576,): a number for every product",
            "customers": "(2829,): a number for every customer",
            "error": "An error: the shapes do not fit",
        },
        correct="right",
        why={
            "customers": "`R.T` has shape (3576, 2829): a row for every product. Times a vector of 2,829 numbers, that gives one number per row of `R.T`, so per product. See *The idea* in [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en).",
            "error": "`R.T` has shape (3576, 2829), and its 2,829 columns fit the 2,829 numbers of `R[:, j]`. See the shape rule in [A whole batch at once: matrix multiplication](/course/open/math-matrix-multiply?lang=en).",
            "right": "Each number is the dot product of a column of `R` with column `j`. Today that dot product is how many customers bought both products.",
        },
    )
    quick_transpose
    return (quick_transpose,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Content-based filtering needs to know what a product is, and a name says little: PARTY BUNTING and REGENCY CAKESTAND 3 TIER share no word, yet many customers buy them for the same party. The shop knows something better than the names: who bought what, for thousands of customers. A product that many of the same customers buy is probably a good match, whatever its name.

    **An everyday example.** A record shop owner notices that the customers who buy one album often buy a certain other one, and puts the two side by side. They do not need to know what either album sounds like.
    """)
    return


@app.cell(hide_code=True)
def predict_collaborative():
    guess_collaborative = L.guess(
        "collaborative-guess",
        question="Recommend to every customer the ten products that customers who bought the same products bought too, without looking at the names. For how many of the 2,498 customers is at least one of the ten in their last order?",
        options={
            "fewer": "Fewer than with popularity, 21.7%: without names it knows less",
            "same": "About as many as with the names, 23.8%",
            "right": "More than with either: about a third",
        },
        reveal="**About a third.** 800 of the 2,498 customers, 32.0%: more than the names, 595, and more than popularity, 541. Who bought what says more about a product than its name. And together the three find something for 1,195 customers, 47.8%.",
    )
    guess_collaborative
    return (guess_collaborative,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **A product is who bought it.** Take the user-item matrix of the orders before every customer's last: 2,829 customers and 3,576 products, with 195,264 ones. Call it `R`. A row is a customer; a column is a product, described by who bought it: the column of HEART OF WICKER SMALL has a 1 for each of its 423 buyers. The names play no part.

        **Alike when the same customers buy them.** The dot product of two columns counts the customers who bought both products. `R.T @ R[:, j]` gives it for column `j` and every column at once. But a count favors what is popular: of the 423 buyers of the small heart of wicker, 216 also bought the WHITE HANGING HEART T-LIGHT HOLDER, but that holder has 642 buyers, and many of them would have bought it anyway. The cosine similarity of the two columns divides by both lengths. A column of zeros and ones is as long as the square root of its number of buyers: 216 / √(423 × 642) = 0.414 for the holder, against 254 / √(423 × 368) = 0.644 for HEART OF WICKER LARGE, with 368 buyers. That is the **item-item similarity**: how alike two products are by who bought them.

        **Collaborative filtering.** Give every product a score: its similarities with the products a customer bought, added up. Recommend the `k` with the highest scores that the customer has not bought, equal scores in the order of popularity. That is **collaborative filtering**: the customers together decide what is alike, and the recommender needs to know nothing about the products.

        **Cold start.** A new product that nobody has bought has a column of zeros: it is like nothing, and nothing brings it up. A new customer has bought nothing, so there is nothing to add up. That is the **cold start** problem. In this shop, 1,505 of the 4,334 customers placed a single order: for that order, neither collaborative filtering nor content-based filtering had anything to go on. New products are rare here: of the 37,914 products new to a customer in their last order, 26 have no column in `R`: no customer bought them before their own last order.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **item-item similarity** | how alike two items are by the users who had them: the cosine similarity of their columns in the user-item matrix | |
        | **collaborative filtering** | recommending from what many users did, without knowing what the items are | |
        | **cold start** | a new user or item that a recommender knows nothing about yet | |
        """),
        mo.md(r"""
        The picture shows both for HEART OF WICKER SMALL. On the left, the five products bought together with it by the most customers; on the right, the five most alike by cosine similarity. With each, how many customers bought it.
        """),
        mo.Html(C.pairs_picture(C.PRODUCT, {"together": "Bought together most often", "alike": "Most alike (cosine similarity)",
                                            "buyers": "{n} buyers",
                                            "alt": "Two lists of five products. Bought together most often with HEART OF WICKER SMALL: HEART OF WICKER LARGE, 254 customers; WHITE HANGING HEART T-LIGHT HOLDER, 216; NATURAL SLATE HEART CHALKBOARD, 157; SMALL WHITE HEART OF WICKER, 155; REGENCY CAKESTAND 3 TIER, 138. Most alike by cosine similarity: HEART OF WICKER LARGE, 0.644; SMALL WHITE HEART OF WICKER, 0.482; LARGE WHITE HEART OF WICKER, 0.438; WHITE HANGING HEART T-LIGHT HOLDER, 0.414; WICKER STAR, 0.369."}, L.lang)),
        mo.md(r"""
        The counts bring in the products everyone buys: the REGENCY CAKESTAND 3 TIER, the most popular product of the shop, comes fifth with 138 customers, but with 663 buyers its cosine similarity is only 0.261. The cosine similarity puts the white hearts of wicker and a WICKER STAR in their place: what buyers of this heart buy more often than other customers.
        """),
        L.watch_out("Add up counts instead of similarities, and the popular products win for everyone. Scored with how many customers bought both, the ten products find something for 657 customers, 26.3%; with the cosine similarities, for 800, 32.0%."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To recommend with item-item collaborative filtering:

    1. Make the user-item matrix `R` of what the users had before the split: a row for every user, a column for every item.
    2. For an item `j`, count for every item how many users had both: `R.T @ R[:, j]`.
    3. Divide by the lengths of both columns: the item-item similarity.
    4. For a user, add up the similarities with every item they had, and recommend the `k` highest that they have not had.
    5. Measure the hit rate, and compare with the other recommenders on the same users.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell follows steps 1 and 2. It splits the orders by time as before, gives every product a column, the most popular first, and every customer with an order before their last a row. A 1 goes where the customer bought the product. Then it counts, for HEART OF WICKER SMALL, how many of its buyers bought each product, and prints the six highest counts: the count, how many customers bought that product, and its name. Run the cell.
    """)
    return


@app.cell
def example_together():
    import numpy as np
    import pandas as pd

    # what every customer bought before their last order, and the products, the most popular first
    _rows = pd.read_csv("data/online-retail-baskets.csv.gz").dropna(subset=["customer"])
    _rows["last"] = _rows.groupby("customer")["invoice"].transform("max")
    _train = _rows[_rows["invoice"] < _rows["last"]]
    _seen = _train.groupby("customer")["product"].agg(set)
    _names = list(_train.drop_duplicates(subset=["customer", "product"])["product"].value_counts().index)

    # 1. the user-item matrix: a row for every customer, a column for every product
    _col_of = {}
    for _i in range(len(_names)):
        _col_of[_names[_i]] = _i
    _customers = list(_seen.index)
    _R = np.zeros((len(_customers), len(_names)))
    for _r in range(len(_customers)):
        for _product in _seen[_customers[_r]]:
            _R[_r, _col_of[_product]] = 1
    print(_R.shape, _R.sum())

    # 2. for every product: how many of the buyers of HEART OF WICKER SMALL bought it too
    _j = _col_of["HEART OF WICKER SMALL"]
    _together = _R.T @ _R[:, _j]
    for _i in np.argsort(-_together, kind="stable")[:6]:
        print(int(_together[_i]), int(_R[:, _i].sum()), _names[_i])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    2,829 rows and 3,576 columns, with 195,264 ones. The first line of the counts is the heart itself: all 423 of its buyers bought it. Then the large heart of wicker, 254 of 368 buyers, and the white hanging heart, 216 of 642. The counts are the dot products of the columns; the lengths that turn them into cosine similarities are the square roots of the second numbers.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_similar():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Alike by who bought them

        `similar_to(R, j)` should give the cosine similarity of column `j` of `R` with every column: an array with a number for every product. `together` already holds how many customers bought each product together with product `j`, and `lengths` the length of every column. Replace the `...`. The check calls `similar_to(R, j)` with the matrix of the worked example and `j` the column of HEART OF WICKER SMALL, and tries three small matrices.
        """),
        L.task("Make `similar_to` give the cosine similarity of column `j` with every column."),
    ])
    return


@app.cell(hide_code=True)
def exercise_similar():
    box_similar = L.code_box("similar-to", lines=7)
    box_similar
    return (box_similar,)


@app.cell(hide_code=True)
def check_similar(box_similar):
    result_similar = L.check("similar-to", box_similar.value)
    result_similar
    return (result_similar,)


@app.cell(hide_code=True)
def hints_similar():
    L.hints(
        "similar-to",
        [
            "It is the same cosine similarity as in `cosine_scores` of the previous lesson, now with the columns of `R`. Which two lengths divide the count?",
            "`lengths` has a length for every column, so dividing by `lengths` alone covers only one of the two columns. Where in `lengths` is the length of column `j`?",
            "`return together / (___ * lengths[___])`.",
        ],
        why="For two columns of zeros and ones, the dot product counts the customers who bought both, and each length is the square root of a number of buyers. Dividing by both makes a product that everyone buys no more alike than one that only the same customers buy.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_scores():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The scores for one customer

        `cf_scores(R, bought)` should give the score of every product for one customer: its similarities with each product in `bought`, the columns of what the customer bought, added up. `similar_to` is already in the box, the course's version. Replace the `...`: what `scores` becomes in every round of the loop. The check calls `cf_scores(R, bought)` with the four products customer 13960 bought before their last order, and tries three small matrices. To recommend, take the highest scores that the customer has not bought, with `recommend` of the previous lesson.
        """),
        L.task("Make `cf_scores` add up the similarities with every bought product."),
    ])
    return


@app.cell(hide_code=True)
def exercise_scores():
    box_scores = L.code_box("cf-scores", lines=14)
    box_scores
    return (box_scores,)


@app.cell(hide_code=True)
def check_scores(box_scores):
    result_scores = L.check("cf-scores", box_scores.value)
    result_scores
    return (result_scores,)


@app.cell(hide_code=True)
def hints_scores():
    L.hints(
        "cf-scores",
        [
            "In every round, one bought product adds its similarities to the scores so far, as the rows of the products were added up into a profile in the previous lesson.",
            "`similar_to(R, j)` gives an array with a number for every product, as long as `scores`. Two arrays of the same length add up value by value.",
            "`scores = scores + ___(R, ___)`.",
        ],
        why="A product that is alike to several of the customer's products gets a bit from each. For customer 13960, the WHITE HANGING HEART T-LIGHT HOLDER comes first: it is fairly alike to all four of their products, without being the most alike to any of them.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_new():
    mo.md(r"""
    ### Exercise 3. A first order
    """)
    return


@app.cell(hide_code=True)
def choice_new_cell():
    choice_new = L.choice(
        "new-customer",
        question="A customer places their very first order with the shop today. Which of the three recommenders can choose ten products for them?",
        options={
            "right": "Only the popularity baseline",
            "collaborative": "Only collaborative filtering",
            "content": "Content-based filtering and collaborative filtering",
            "all": "All three",
        },
        correct="right",
        why={
            "collaborative": "Collaborative filtering adds up the similarities with what the customer bought, and they bought nothing yet: every score is 0. See *Cold start* in *The idea*.",
            "content": "Both need what the customer bought: content-based filtering for the profile, collaborative filtering for the similarities to add up. A new customer has neither. See *Cold start* in *The idea*.",
            "all": "Content-based filtering and collaborative filtering both start from what the customer bought, and a new customer bought nothing. See *Cold start* in *The idea*.",
            "right": "Popularity needs nothing from the customer: everyone gets the most popular products. That is why it stays useful next to the other two, here for 1,505 of the 4,334 customers, who placed only one order.",
        },
    )
    choice_new
    return (choice_new,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The user-item matrix and the hit rate come from [What to show next: the recommendation problem](/course/open/ml-recommend-problem?lang=en), the cosine similarity and the adding up from [More of what you bought: content-based filtering](/course/open/ml-content-based?lang=en), now with the columns of the matrix instead of the words of the names, and `R.T` from [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en). Collaborative filtering finds the most customers here, 32.0%, but knows nothing about a new customer or a new product; popularity and the names fill that gap. Who bought what is also a network: customers and products as nodes, every purchase an edge. In the next lesson, *Networks as data: degrees and hubs*, you look at networks themselves.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_new, result_scores, result_similar):
    L.wrap_up(result_similar, result_scores, choice_new)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
