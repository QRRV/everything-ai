# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-dedup. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-dedup", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Cleaning text: quality filters](/course/open/llm-quality-filter?lang=en) and [Tuples, sets and sorting](/course/open/py-tuples-sets?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rules():
    quick_rules = L.choice(
        "quick-rules",
        question="A page asks you to prove that you are a human, in four long, well-written sentences. Do the line rules of C4 keep it?",
        options={
            "right": "Yes: its lines end a sentence and have at least 5 words",
            "no": "No: C4 knows that such notices are not real text, and removes them",
            "page": "No: a page about humans is thrown out",
        },
        correct="right",
        why={
            "no": "The line rules look only at how a line ends, how many words it has and whether it mentions JavaScript: not at whether the text is worth reading. See [Cleaning text: quality filters](/course/open/llm-quality-filter?lang=en).",
            "page": "No rule of C4 or Gopher asks what a page is about: they look at lines, words and marks, and C4 at a list of bad words. See [Cleaning text: quality filters](/course/open/llm-quality-filter?lang=en).",
            "right": "Rules about the form of text let such a notice through, on every site that shows it. This lesson finds the copies.",
        },
    )
    quick_rules
    return (quick_rules,)


@app.cell(hide_code=True)
def quick_check_set():
    quick_set = L.choice(
        "quick-set",
        question="What does this program show?\n\n```python\nshingles = {\"the cat\", \"cat sat\"}\nprint(shingles[0])\n```",
        options={
            "first": "`the cat`: the value written first",
            "sorted": "`cat sat`: a set keeps its values from A to Z",
            "right": "An error: a set has no places, so it has no index",
        },
        correct="right",
        why={
            "first": "A set has no places, so there is no first value: `shingles[0]` stops with an error. See [Tuples, sets and sorting](/course/open/py-tuples-sets?lang=en).",
            "sorted": "A set keeps no order, not even from A to Z, so it has no place 0: `shingles[0]` stops with an error. See [Tuples, sets and sorting](/course/open/py-tuples-sets?lang=en).",
            "right": "A set only knows which values it holds, each once. That is all this lesson needs: which runs of words two pages share, not where.",
        },
    )
    quick_set
    return (quick_set,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Many web pages are almost the same as another one: the same notice on every site that uses the same service, the same article with another date or menu, the same text copied to a hundred shops. A model that sees a text many times learns it by heart. In C4, one sentence of 61 words was there more than 60,000 times, and of what models trained on corpora like it wrote without a prompt, more than 1% was copied word for word from their training data. After deduplication, of exact copies and near-copies, they did that ten times less often, and needed fewer steps to learn as well (Lee and others, 2022). But comparing every page with every other is far too slow for millions of pages. This lesson shows a trick that finds the near-copies without that.

    **An everyday example.** Two photos of the same street, taken a minute apart, are not the same file, but you see at once that they are almost the same picture. To find such pairs among 10,000 photos, you would rather not hold every photo next to every other.
    """)
    return


@app.cell(hide_code=True)
def predict_copies():
    _m = C.MEASURED
    guess_copies = L.guess(
        "copies",
        question=f"Of the {C.number(C.Q.MEASURED['pages'], 0, L.lang)} raw pages of the download of [Cleaning text: quality filters](/course/open/llm-quality-filter?lang=en), {C.number(_m['pages'], 0, L.lang)} pass all its rules. How many of those do you think are near-copies of another page, and could go while one of each group stays?",
        options={
            "none": "None or nearly none",
            "right": "About 1 in 75",
            "fifth": "About 1 in 5",
        },
        reveal=f"**About 1 in 75: {_m['removed']} pages, in {_m['groups']} groups.** The largest group is {_m['largest']} copies of a notice that asks you to solve a CAPTCHA; others are the notice at the top of an XML sitemap, and a forum's notice that you must log in. Only {_m['exact_extra']} of the {_m['removed']} are exact copies; the others differ in a title, a name, a phone number or a date. In a whole crawl, with millions of pages from the same sites, there are far more.",
    )
    guess_copies
    return (guess_copies,)


@app.cell(hide_code=True)
def text_idea():
    _agree, _j = C.example_agree(24)
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Exact copies are easy.** Put every text in a set or a dict: a text that is already there is a copy.

        **Near-copies need a measure.** Cut every page into shingles: all its runs of $k$ words in a row, here 5. Two pages that differ in one word still share almost all their shingles. Their overlap, the **Jaccard similarity**, is the number of shingles they share divided by the number of different shingles of the two together: 1 for the same page, 0 for pages without a run of 5 words in common. Two pages that share at least 0.8 count as near-duplicates.

        **All pairs are too many.** 5,614 pages make 15,755,691 pairs; a billion pages make about $5 \cdot 10^{17}$. So every page gets a short signature instead, with MinHash:

        1. Give every shingle a number with a hash function: a fixed function from text to a number, the same text always to the same number. Here CRC-32, from Python's zlib, taken `% P` so that it is below $P = 2^{31} - 1$.
        2. Make 128 more hash functions from it: function $i$ turns a hash $h$ into $(a_i h + b_i) \bmod P$, with its own random $a_i$ and $b_i$. Each puts the shingles in another order, as if shuffled.
        3. For every function, keep the smallest number it gives any shingle of the page: those 128 numbers are its signature.

        **Why it works.** Take one function, and all the shingles of two pages together. Each of them is about equally likely to get the smallest number. The two signatures agree in that place exactly when that shingle is in both pages. So the chance that a place agrees is about their overlap, and the share of the 128 places that agree estimates it: on the 594 pairs the demo compares, within 0.012 on average.

        **Banding finds the pairs.** Cut every signature into 16 bands of 8 numbers. Two pages that agree in all 8 numbers of a band land in the same bucket, and only pages that share a bucket are compared. A pair with an overlap of 0.9 shares a band with a chance of 99.99%, a pair of 0.8 with a chance of 95%, and a pair of 0.3 with a chance of 0.1%. This is locality-sensitive hashing: here, 594 pairs to compare instead of 15.8 million.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **deduplication** | | removing copies from a dataset, exact copies and near-duplicates, so that a model does not see the same text many times | |
        | **near-duplicate** | | a document that is almost the same as another, such as the same notice with another name; a check for exact copies does not find it | |
        | **shingling** | | cutting a document into all its overlapping runs of $k$ words, its shingles | `shingles(text, 5)` |
        | **Jaccard similarity** | | the overlap of two sets: the items they share divided by all their different items together; 1 for the same set, 0 for sets without a shared item | <code>len(a &amp; b) / len(a &#124; b)</code> |
        | **MinHash** | | a signature of a document: for each of many hash functions, the smallest number it gives a shingle; two signatures agree in about as many places as the documents overlap | `minhash(hashes, a, b)` |
        | **locality-sensitive hashing** | | putting similar items in the same bucket with a high chance, so that only items in the same bucket need to be compared; here, banding the signatures | |
        | `&`, <code>&#124;</code>, `-` on sets | | `a & b` is the set of the items in both sets, <code>a &#124; b</code> of the items in at least one, `a - b` of the items in `a` but not in `b` | `{1, 2} & {2, 3}` gives `{2}` |
        | **outer product** | | a table of every number of one array times every number of another: row `r`, column `c` holds `h[r] * a[c]` | `np.outer(h, a)` |
        | $\bmod$ | mod | the remainder after dividing: $10 \bmod 3 = 1$ | `10 % 3` |
        """),
        mo.md(r"""
        The picture: the signatures of the two texts of the worked example, with 24 hash functions, place by place. A filled cell is a place where the two signatures have the same number.
        """),
        mo.Html(C.signatures_picture(_agree, {"title": "Where two signatures agree", "row": f"the {len(_agree)} places of the signatures of text 1 and text 2",
                                              "notes": [f"{sum(_agree)} of {len(_agree)} places agree: an estimate of {C.number(sum(_agree) / len(_agree), 2, L.lang)}",
                                                        f"the overlap of the two texts: {C.number(_j, 2, L.lang)}"],
                                              "alt": f"A row of {len(_agree)} cells, one per hash function. {sum(_agree)} cells are filled, where the signatures of the two texts agree: an estimate of {C.number(sum(_agree) / len(_agree), 2, L.lang)}, for an overlap of {C.number(_j, 2, L.lang)}."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To remove the near-duplicates of a corpus:

    1. Cut every page into shingles of $k$ words, and give every shingle a number with a hash function.
    2. Make a signature for every page: for each of many hash functions, the smallest number it gives the page's shingles.
    3. Cut the signatures into bands; pages that agree in a whole band are candidates.
    4. Compare every candidate pair by its overlap, and put pages that share at least 0.8 in one group.
    5. Keep one page of every group, and remove the rest.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 1 and the overlap for two short texts that differ in one word, with runs of 2 words.
    """)
    return


@app.cell
def example_overlap():
    _a = "the cat sat on the mat and looked at the birds all day".split()
    _b = "the cat sat on the hat and looked at the birds all day".split()
    # 1. every run of 2 words, as a set
    _sa = set()
    for _i in range(len(_a) - 1):
        _sa.add(" ".join(_a[_i:_i + 2]))
    _sb = set()
    for _i in range(len(_b) - 1):
        _sb.add(" ".join(_b[_i:_i + 2]))
    # the overlap: the shingles both have, divided by all the different shingles of the two
    print(len(_sa & _sb), "shared of", len(_sa | _sb), "together:", round(len(_sa & _sb) / len(_sa | _sb), 3))
    print("only in text 1:", _sa - _sb)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    One word differs, and it is in two shingles: "the mat" and "mat and" are only in text 1, "the hat" and "hat and" only in text 2. The other 10 are shared, of 14 together: an overlap of 0.714. With runs of 5 words, as in the demo, one word changes up to 5 shingles, so short texts need a smaller $k$. A small $k$ makes unrelated pages share common runs such as "one of the", so pages need a larger one.
    """)
    return


@app.cell(hide_code=True)
def text_numpy_rung():
    mo.md(r"""
    ### One rung up: the same in NumPy

    The same with arrays. With NumPy, one hash function runs on all hashes of a page at once, and two signatures are compared in one line. The numbers below are made up, small enough to read.
    """)
    return


@app.cell
def example_numpy():
    import numpy as np

    _P = 2 ** 31 - 1
    _hashes = np.array([12, 7, 30])
    # one hash function on every hash at once, and its smallest number
    _values = (5 * _hashes + 3) % _P
    print(_values, _values.min())
    # two signatures: the share of places that agree
    _sig1 = np.array([4, 18, 9, 27])
    _sig2 = np.array([4, 18, 11, 27])
    print((_sig1 == _sig2).mean())
    return


@app.cell(hide_code=True)
def text_after_numpy():
    mo.md(r"""
    | Python | NumPy |
    |---|---|
    | a loop over the hashes, with `(a * h + b) % P` | `(a * hashes + b) % P`, an array |
    | keep the smallest in the loop | `.min()` |
    | count the places that agree, then divide | `(sig1 == sig2).mean()` |

    `sig1 == sig2` gives an array of `True` and `False`, and the mean of it counts `True` as 1: here 3 of 4, 0.75.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_shingles():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Shingles

        Complete `shingles(text, k)`: step 1 of the recipe. It gives back the set of all runs of `k` words in a row of `text` in lower case, each as one text with a space between the words. A text with fewer than `k` words has none. The check calls `shingles("A dog ran after the Red ball", 3)`, and others.
        """),
        L.task("Make `shingles` give every run of `k` words, once."),
    ])
    return


@app.cell(hide_code=True)
def exercise_shingles():
    box_shingles = L.code_box("shingles", lines=8)
    box_shingles
    return (box_shingles,)


@app.cell(hide_code=True)
def check_shingles(box_shingles):
    result_shingles = L.check("shingles", box_shingles.value)
    result_shingles
    return (result_shingles,)


@app.cell(hide_code=True)
def hints_shingles():
    L.hints(
        "shingles",
        [
            "The worked example does it for `k` = 2. A run of `k` words starts at every place `i` from which `k` words still follow.",
            "The last run starts at word `len(words) - k`. How far must the `range` go to include that place? The run from place `i` is a slice of `words`.",
        ],
        why="A set keeps every shingle once, so a page that repeats a sentence does not count it twice, and two sets can be compared with `&` and `|`. Lower case makes \"The\" and \"the\" the same word.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_minhash():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. MinHash in Python

        Complete `minhash(hashes, a, b)`: step 2 of the recipe. `hashes` are the numbers of the shingles of a page; `a` and `b` hold the numbers of the hash functions, function `i` with `a[i]` and `b[i]`. It gives back the signature, a list with for every function the smallest `(a[i] * h + b[i]) % P` over all `h` in `hashes`, or `P` when there are none. The check calls `minhash([5, 9, 2], [3, 7], [1, 4])`, and others.
        """),
        L.task("Make `minhash` keep, for every hash function, the smallest number it gives."),
    ])
    return


@app.cell(hide_code=True)
def exercise_minhash():
    box_minhash = L.code_box("minhash", lines=12)
    box_minhash
    return (box_minhash,)


@app.cell(hide_code=True)
def check_minhash(box_minhash):
    result_minhash = L.check("minhash", box_minhash.value)
    result_minhash
    return (result_minhash,)


@app.cell(hide_code=True)
def hints_minhash():
    L.hints(
        "minhash",
        [
            "Inside the loop over the functions, a second loop over `hashes`, as when you look for the smallest number of a list yourself.",
            "Start at `smallest = P`: every outcome of `% P` is below it, so the first one replaces it.",
        ],
        why="Every function puts the shingles in its own order; the smallest is the first in that order. Two pages agree in place `i` when the same shingle comes first for both, and the chance of that is about their overlap.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_signature():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Signatures with NumPy

        Complete `signature(hashes, a, b)`: the same signature as your `minhash`, with all hash functions on all hashes at once. `np.outer(h, a)` makes a table with a row per hash and a column per function: `h[r] * a[c]` in row `r` and column `c`. Then `+ b` adds `b[c]` to every column. The check calls `signature([5, 9, 2], [3, 7], [1, 4])`, and others; the demo calls it 5,614 times, with 128 functions.
        """),
        L.task("Make `signature` compute the table of every function on every hash, and the smallest of every column."),
    ])
    return


@app.cell(hide_code=True)
def exercise_signature():
    box_signature = L.code_box("signature", lines=12)
    box_signature
    return (box_signature,)


@app.cell(hide_code=True)
def check_signature(box_signature):
    result_signature = L.check("signature", box_signature.value)
    result_signature
    return (result_signature,)


@app.cell(hide_code=True)
def hints_signature():
    L.hints(
        "signature",
        [
            "Take the line of your `minhash` that computes `v`, and put `np.outer(h, a)` where it has `a[i] * h`, and `b` where it has `b[i]`: that gives every `v` at once.",
            "The smallest of every column is `values.min(axis=...)`: which axis runs down the rows?",
        ],
        why="Axis 0 runs down the rows, over the hashes, so `values.min(axis=0)` keeps one number per column, per function: the signature. NumPy does the two loops of your `minhash` in compiled code, many times faster.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(f"""
    ### What your functions find in the cleaned web pages

    The button below downloads the web pages of [Cleaning text: quality filters](/course/open/llm-quality-filter?lang=en), once, if you have not done so there: the first {round(C.Q.hub.PINNED[C.C4_SAMPLE]['size'] / 1e6)} MB of a shard of C4 before its cleaning. The course cleans them with its own versions of that lesson's rules, and then your `shingles` cuts every kept page into shingles of {C.K} words and your `signature` gives it {C.N_HASHES} numbers. Banding by the course, {C.BANDS} bands of {C.ROWS}, finds the candidate pairs, and every candidate pair is compared by its overlap. It takes about ten seconds.
    """)
    return


@app.cell(hide_code=True)
def download_button_cell():
    download_button = mo.ui.run_button(label="Download the web pages")
    download_button
    return (download_button,)


@app.cell(hide_code=True)
def demo_dedup(download_button, box_shingles, box_signature, result_shingles, result_signature):
    _ready = result_shingles.state == "correct" and result_signature.state == "correct"
    mo.stop(not _ready, mo.md("*Finish exercises 1 and 3 first: the demo runs your `shingles` and your `signature`.*"))
    C.sample_or_stop(download_button, {
        "ask": "*Click the button to download the web pages, 20 MB.*",
        "busy": "**Downloading** · {done} of {total} MB",
        "no_net": "*There is no connection to Hugging Face. Try again when you are online.*",
        "not_pinned": "*The file from Hugging Face is not the one the course pinned, so it was thrown away. Report it on GitHub, with the link at the bottom of the lesson.*",
        "no_room": "*There is no room left on the disk for the download. Make room, and click again.*",
        "failed": "*The download did not work: {why}.*",
    })
    _sh = run_code(box_shingles.value).names["shingles"]
    _sg = run_code(box_signature.value).names["signature"]
    _r = C.find_duplicates(_sh, _sg)
    _named = [C.GROUPS[_p][L.lang] for _p in _r["places"][:4] if _p in C.GROUPS]
    _which = "" if not _named else " The largest groups: " + "; ".join(_named) + "."
    mo.md(f"""Of the {C.number(_r['all_pairs'], 0, L.lang)} pairs of {C.number(_r['pages'], 0, L.lang)} pages, banding makes {C.number(_r['candidates'], 0, L.lang)} candidates, and {C.number(_r['similar'], 0, L.lang)} of them share at least {C.number(C.SIMILAR, 1, L.lang)}. On those candidates, the share of places where two signatures agree is on average {C.number(_r['error'], 3, L.lang)} away from the true overlap. Together they make {_r['groups']} groups of near-duplicates, the largest of {_r['largest']} pages: {_r['removed']} pages can go, if one of each group stays. Only {_r['exact_extra']} of them are exact copies.{_which}""")
    return


@app.cell(hide_code=True)
def text_exercise_memorize():
    mo.md(r"""
    ### Exercise 4. Why remove copies?
    """)
    return


@app.cell(hide_code=True)
def choice_memorize_cell():
    choice_memorize = L.choice(
        "memorize",
        question="One sentence of 61 words was in C4 more than 60,000 times. What does that do to a model trained on it?",
        options={
            "nothing": "Nothing: a model learns patterns, not sentences",
            "right": "It learns the sentence by heart, and wastes steps on it",
            "better": "It learns that sentence better, and that helps it on every text it reads",
        },
        correct="right",
        why={
            "nothing": "Models do copy what they saw often: those trained on corpora with many copies copied more than 1% of what they wrote without a prompt word for word from their training data (Lee and others, 2022).",
            "better": "The 60,000th time teaches it nothing new; it takes steps away from text the model has not seen. After deduplication, models needed fewer steps to learn as well.",
            "right": "After deduplication, models copied text word for word ten times less often, and reached the same results in fewer steps (Lee and others, 2022). Copies of test questions are worse still: the next lesson.",
        },
    )
    choice_memorize
    return (choice_memorize,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The pages that pass the filters of [Cleaning text: quality filters](/course/open/llm-quality-filter?lang=en), now without their near-copies; the sets of [Tuples, sets and sorting](/course/open/py-tuples-sets?lang=en), now as shingles. The next lesson looks for a special kind of copy: test questions that slipped into the training data.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_shingles, result_minhash, result_signature, choice_memorize):
    L.wrap_up(result_shingles, result_minhash, result_signature, choice_memorize)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
