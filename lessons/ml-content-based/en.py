# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson ml-content-based. Route: course/route/lessen/klassiek-ml.yaml. Checks and pictures: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("ml-content-based", "en")
    C = L.checks   # the words of the product names, the profiles, both recommenders for every customer, and the pictures


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en) and one about [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_cosine():
    quick_cosine = L.choice(
        "quick-cosine",
        question="Two texts, as counts of three words: `a = [2, 0, 1]` and `b = [4, 0, 2]`. `b` uses the same words in the same mix, twice as often. What is their cosine similarity?",
        options={
            "right": "1",
            "half": "0.5, because `b` is twice as long",
            "two": "2",
        },
        correct="right",
        why={
            "half": "The cosine similarity divides by both lengths, so length drops out: only the direction counts, and `b` points the same way as `a`. See *The idea* in [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en).",
            "two": "The cosine similarity is never more than 1: the dot product is divided by both lengths. See *The idea* in [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en).",
            "right": "The same direction, so 1, whatever the lengths. Today the texts are the names of products.",
        },
    )
    quick_cosine
    return (quick_cosine,)


@app.cell(hide_code=True)
def quick_check_argsort():
    quick_argsort = L.choice(
        "quick-argsort",
        question="What does `np.argsort(np.array([0.3, 0.9, 0.1]), kind=\"stable\")` give?",
        options={
            "right": "`[2, 0, 1]`",
            "values": "`[0.1, 0.3, 0.9]`",
            "largest": "`[1, 0, 2]`",
        },
        correct="right",
        why={
            "values": "`np.argsort` gives positions, not values: where the smallest value is, then the next. See the table of new words in [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).",
            "largest": "`np.argsort` goes from the smallest value up, not from the largest down. See the table of new words in [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).",
            "right": "The smallest value, 0.1, is at position 2, then 0.3 at 0 and 0.9 at 1. Today you need the largest first.",
        },
    )
    quick_argsort
    return (quick_argsort,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Before their last order, customer 13960 of the online shop of the previous lesson bought four products: a small and a large HEART OF WICKER, a HANGING METAL HEART LANTERN and a doormat. The popularity baseline gives them the ten products it gives almost everyone, and none of them is in their last order. A shop assistant who saw those four products would show them more hearts. To do that, a recommender has to know what a product is, and not only how often it was bought.

    **An everyday example.** A friend who knows you read detective novels recommends another detective novel, even one that came out yesterday and that nobody has read yet.
    """)
    return


@app.cell(hide_code=True)
def predict_content():
    guess_content = L.guess(
        "content-guess",
        question="Give every customer the ten products whose names look most like the names of what they bought. Compared with the ten most popular products, for how many of the 2,498 customers is at least one of the ten in their last order?",
        options={
            "fewer": "For far fewer customers",
            "right": "For about as many, but not the same customers",
            "double": "For twice as many",
        },
        reveal="**About as many, but not the same customers.** The names find something for 595 customers, 23.8%; popularity for 541, 21.7%. But only 188 customers get a hit from both. The names find 407 customers that popularity misses, and popularity 353 that the names miss.",
    )
    guess_content
    return (guess_content,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Item features.** Describe every item by what it is: its **item features**. Here they are the words of the product names. The 3,576 products that were bought before a last order have 2,163 different words in their names, on average 4.36 different words per name. Give every word a column and every product a row, with a 1 where the word is in the name: a matrix of 3,576 rows and 2,163 columns, like the word counts of [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en). The row of HEART OF WICKER SMALL has a 1 in the columns of HEART, OF, WICKER and SMALL, and a 0 in the 2,159 others.

        **The user profile.** Add up the rows of everything a customer bought: their **user profile**, a vector with, for every word, how many of their products have it in their name. The four products of customer 13960 have 18 words, 14 of them different: HEART three times, OF and WICKER twice, and the other eleven once.

        **Closest to the profile.** Give every product a score: the cosine similarity of its row with the profile. Recommend the `k` products with the highest scores that the customer has not bought. Many products get the same score: their names are as long, and share as much with the profile. Equal scores keep the order of the products, the most popular first. That is **content-based filtering**: the products closest to what a customer bought, measured by what the products are.

        Why the cosine similarity, and not the dot product alone? The dot product of a row and the profile counts the words the name shares with the profile, each as often as it is in the profile. SET OF 3 HEART COOKIE CUTTERS shares OF and HEART with the profile of customer 13960, and FINE WICKER HEART shares WICKER and HEART: both get a dot product of 5. But the cookie cutters have six words and the heart three. The cosine similarity divides by both lengths, so it counts what share of the name matches: 0.386 for the cookie cutters, 0.546 for the wicker heart.

        ### New words and symbols

        | Word or symbol | What it means | In code |
        |---|---|---|
        | **item features** | what a recommender knows about an item itself, such as the words of its name | a row of `features` |
        | **user profile** | the item features of everything a user had, added up | `profile = profile + features[row]` |
        | **content-based filtering** | recommending the items whose features are closest to the profile of a user | |
        | `text.split()` | cuts a text into its words at the spaces | `"HEART OF WICKER".split()` gives `['HEART', 'OF', 'WICKER']` |
        """),
        L.watch_out("Content-based filtering only knows what a customer bought, so it recommends more of the same. Customer 14905 bought two T-light holders among nine products. Their ten are all holders, nine of them for T-lights, and none is in their last order; the ten most popular products find two. Look at the customers a recommender misses, not only at its hit rate."),
        mo.md(r"""
        The slider goes through five customers. On the left, the words most often in their profile; on the right, the ten products of each recommender, a square each: filled when the product is in their last order, open when it is not. Move the slider.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_customer_cell():
    def _frame(i):
        customer = C.CUSTOMERS[i - 1]
        said = C.said(customer, L.lang)
        alt = ("Two charts. Left: bars for the words most often in the profile of the customer, the longest at the top. "
               "Right: two rows of ten squares, content-based filtering above and popularity below, filled when the "
               "product is in the customer's last order. " + said)
        return (C.picture(customer, {"words": "Words of the profile", "times": "products with the word",
                                     "ten": "Their ten", "place": "place in the list", "content": "content-based",
                                     "popular": "popularity", "alt": alt}), said)

    explore_customer = L.explore("customer-slider", "customer", _frame, start=1, values=[1, 2, 3, 4, 5], role="",
                                 task="Go through the five customers. For which of them do the names find more than popularity, and for which less?")
    explore_customer
    return


@app.cell(hide_code=True)
def text_after_slider():
    mo.md(r"""
    For customers who keep buying one kind of product, the names work. Customer 13242 bought five jumbo bags, and six of the ten jumbo bags they get are in their last order; customer 16356 bought metal signs, and three of their ten are in their last order. Popularity finds nothing for either. For customers 14905 and 14875 it is the other way around: what they bought next is popular, but not what their profile points to. Over all 2,498 test customers the two find about as many, 595 against 541, but different customers: at least one of the two finds something for 948 customers, 38.0%.

    ### Recipe

    To recommend with content-based filtering:

    1. Give every item a vector of its features: here a column for every word of the names, with a 1 where the word is in the name.
    2. For every user, add up the rows of the items they had: their profile.
    3. Score every item with the cosine similarity of its row and the profile.
    4. Recommend the `k` items with the highest scores that the user has not had; equal scores in the order of popularity.
    5. Measure the hit rate on the hidden orders, and compare with the popularity baseline on the same users.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell follows steps 1 and 2. It splits the orders by time as in the previous lesson, and lists the products, the most popular first. Then it gives every word of their names a column: `_words[_word]` is the number of the column of a word, and `_row_of[_product]` the number of the row of a product. `np.zeros` makes the matrix, and a 1 goes where a word is in a name. The customer numbers are floats, such as `13960.0`, as in the previous lesson. Last, it adds up the rows of what customer 13960 bought before their last order, and prints those products, and the words that are in more than one of them. Run the cell.
    """)
    return


@app.cell
def example_profile():
    import numpy as np
    import pandas as pd

    # what every customer bought before their last order, and the products, the most popular first
    _rows = pd.read_csv("data/online-retail-baskets.csv.gz").dropna(subset=["customer"])
    _rows["last"] = _rows.groupby("customer")["invoice"].transform("max")
    _train = _rows[_rows["invoice"] < _rows["last"]]
    _seen = _train.groupby("customer")["product"].agg(set)
    _names = list(_train.drop_duplicates(subset=["customer", "product"])["product"].value_counts().index)

    # 1. a column for every word of the names, a row for every product, and a 1 where the word is in the name
    _words = {}
    _row_of = {}
    for _i in range(len(_names)):
        _row_of[_names[_i]] = _i
        for _word in _names[_i].split():
            if _word not in _words:
                _words[_word] = len(_words)
    _features = np.zeros((len(_names), len(_words)))
    for _i in range(len(_names)):
        for _word in _names[_i].split():
            _features[_i, _words[_word]] = 1
    print(_features.shape, round(_features.sum() / len(_names), 2))

    # 2. the profile of customer 13960: the rows of what they bought, added up
    _profile = np.zeros(len(_words))
    for _product in sorted(_seen[13960.0]):
        print(_product)
        _profile = _profile + _features[_row_of[_product]]
    for _word in _words:
        if _profile[_words[_word]] > 1:
            print(_word, _profile[_words[_word]])
    print(_profile.sum(), round(np.linalg.norm(_profile), 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    3,576 rows and 2,163 columns, with on average 4.36 ones in a row: most of the matrix is 0, like the user-item matrix. Customer 13960 bought the four products above. HEART is in three of them, OF and WICKER in two; the profile adds up to 18, and its length is 5.29. The doormat adds six words that no heart has: DOORMAT, KEEP, CALM, AND, COME and IN.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_cosine():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Every product against the profile

        `cosine_scores(features, profile)` should give, for every row of `features`, its cosine similarity with `profile`: an array with a score for every product. `dots` already holds the dot product of every row with the profile, and `lengths` the length of every row, as `np.linalg.norm(..., axis=1)` gave the distances in [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en). Replace the `...`. The check calls `cosine_scores(features, profile)` with the features of the worked example and the profile of customer 13960, and tries three small arrays.
        """),
        L.task("Make `cosine_scores` give the cosine similarity of every row with the profile."),
    ])
    return


@app.cell(hide_code=True)
def exercise_cosine():
    box_cosine = L.code_box("cosine-scores", lines=7)
    box_cosine
    return (box_cosine,)


@app.cell(hide_code=True)
def check_cosine(box_cosine):
    result_cosine = L.check("cosine-scores", box_cosine.value)
    result_cosine
    return (result_cosine,)


@app.cell(hide_code=True)
def hints_cosine():
    L.hints(
        "cosine-scores",
        [
            "The cosine similarity divides the dot product by two lengths. Which two?",
            "`dots` and `lengths` have a value for every product; the length of the profile is one number. Arrays of the same length divide value by value, and a number divides every value.",
            "`return dots / (___ * np.linalg.norm(___))`.",
        ],
        why="Dividing by the length of each row makes a long name count no more than a short one with the same matching words; dividing by the length of the profile changes no order for one customer, but keeps every score between 0 and 1.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_recommend():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The ten closest, not yet bought

        `recommend(scores, names, seen, k)` should give the `k` products with the highest scores that are not in `seen`, the highest first; equal scores keep the order of `names`, the most popular first. `scores[i]` is the score of `names[i]`. Replace the two `...`: the positions to go through, from the highest score down, and when a product goes into `top`. The check calls `recommend(scores, names, seen, 10)` with the scores of customer 13960 and the products they bought before their last order, and tries three small examples.
        """),
        L.task("Make `recommend` give the k products with the highest scores that a customer has not bought."),
    ])
    return


@app.cell(hide_code=True)
def exercise_recommend():
    box_recommend = L.code_box("recommend", lines=9)
    box_recommend
    return (box_recommend,)


@app.cell(hide_code=True)
def check_recommend(box_recommend):
    result_recommend = L.check("recommend", box_recommend.value)
    result_recommend
    return (result_recommend,)


@app.cell(hide_code=True)
def hints_recommend():
    L.hints(
        "recommend",
        [
            "`np.argsort` gives the positions from the smallest value up, and you want the highest score first. A product goes into `top` under the same two conditions as in `popular` of the previous lesson.",
            "A minus sign turns the order around: the highest score becomes the smallest. Keep equal scores in their order, as in [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en).",
            "`for i in np.argsort(___, kind=\"stable\"):` and `if names[i] ___ seen and len(top) ___ k:`.",
        ],
        why="The two products with the highest scores share four words with the profile: HEART, OF, WICKER, and SMALL or LARGE. The next four share three. Equal scores stay in the order of `names`, so of two equal products the more popular comes first.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_new():
    mo.md(r"""
    ### Exercise 3. A new product
    """)
    return


@app.cell(hide_code=True)
def choice_new_cell():
    choice_new = L.choice(
        "new-product",
        question="The shop adds a new product, GREEN HANGING WICKER HEART, which nobody has bought yet. Which recommender can show it to customer 13960?",
        options={
            "right": "Only content-based filtering",
            "popular": "Only the popularity baseline",
            "both": "Both",
            "neither": "Neither: nobody has bought it",
        },
        correct="right",
        why={
            "popular": "The popularity baseline counts how many customers bought a product before, and for a new product that is 0: it comes last. See *The popularity baseline* in [What to show next: the recommendation problem](/course/open/ml-recommend-problem?lang=en).",
            "both": "Content-based filtering can, but the popularity baseline counts customers, and 0 customers bought the new product: it comes last. See *The popularity baseline* in [What to show next: the recommendation problem](/course/open/ml-recommend-problem?lang=en).",
            "neither": "Content-based filtering does not need buyers: it compares the words of the name with the profile. See *Item features* in *The idea*.",
            "right": "Its name is all content-based filtering needs: GREEN HANGING WICKER HEART scores 0.567 against the profile of customer 13960, as high as the products in 3rd to 6th place of their list. Popularity counts buyers, and it has none.",
        },
    )
    choice_new
    return (choice_new,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The score is the cosine similarity of [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en), here between a name and a profile, and the products closest to the profile are found with the `np.argsort` of [Ask the neighbors: k-nearest neighbors](/course/open/ml-knn?lang=en). The hit rate and the popularity baseline of [What to show next: the recommendation problem](/course/open/ml-recommend-problem?lang=en) show that the names find about as many customers, but other ones. Content-based filtering needs to know what a product is. In the next lesson, *Customers like you: collaborative filtering*, the recommender does not look at the names at all, only at who bought what.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_new, result_cosine, result_recommend):
    L.wrap_up(result_cosine, result_recommend, choice_new)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
