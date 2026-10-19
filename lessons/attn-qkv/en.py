# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson attn-qkv. Route: course/route/lessen/stam-2-gevorderd-expert.yaml. Checks: checks.py.
# termen:allow transpose

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("attn-qkv", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Attention weights from dot products](/course/open/attn-scores?lang=en) and [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_weights_shape():
    quick_weights_shape = L.choice(
        "quick-weights-shape",
        question="A sentence has 6 words with embeddings of 10 numbers: `X` has shape (6, 10). What shape do the attention weights, the softmax of every row of `X @ X.T`, have?",
        options={
            "right": "(6, 6): a weight for every pair of words",
            "per-number": "(6, 10): a weight for every number of every word",
            "numbers": "(10, 10): a weight for every pair of numbers",
        },
        correct="right",
        why={
            "per-number": "`X @ X.T` is (6, 10) times (10, 6): the 10s fit, and (6, 6) remains. See [Attention weights from dot products](/course/open/attn-scores?lang=en).",
            "numbers": "That is `X.T @ X`, which compares the numbers of the embeddings, not the words. See [Attention weights from dot products](/course/open/attn-scores?lang=en).",
            "right": "A row and a column for every word. Today the scores come from two new matrices, but their shape stays the same.",
        },
    )
    quick_weights_shape
    return (quick_weights_shape,)


@app.cell(hide_code=True)
def quick_check_transpose():
    quick_transpose = L.choice(
        "quick-transpose",
        question="`Q` and `K` both have shape (6, 4). What is the number in row `i` and column `j` of `Q @ K.T`?",
        options={
            "same-place": "`Q[i, j]` times `K[i, j]`",
            "columns": "The dot product of column `i` of `Q` and column `j` of `K`",
            "right": "The dot product of row `i` of `Q` and row `j` of `K`",
        },
        correct="right",
        why={
            "same-place": "That would be `Q * K`, number by number, with shape (6, 4). See [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en).",
            "columns": "That is `Q.T @ K`, of shape (4, 4). The `.T` on `K` turns its rows into columns. See [Flipping a matrix: the transpose](/course/open/math-transpose?lang=en).",
            "right": "Column `j` of `K.T` is row `j` of `K`. Today, row `i` of `Q` is what word `i` looks for, and row `j` of `K` what word `j` offers.",
        },
    )
    quick_transpose
    return (quick_transpose,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In [Attention weights from dot products](/course/open/attn-scores?lang=en), every word did three jobs with one and the same embedding: it compared itself with the others, the others compared themselves with it, and it passed itself on. Then a word tends to match best with words like itself. In "a red car", "red" takes most from "red", while what it needs is the thing it describes: "car". A word should be able to look for something other than what it is. Real attention gives every word three learned vectors, one for each job.
    """)
    return


@app.cell(hide_code=True)
def predict_lookup():
    guess_lookup = L.guess(
        "lookup",
        question="A Python dict looks up a key: `prices[\"apple\"]` gives the value stored under the key equal to \"apple\". Attention also works with queries, keys and values. What do you think attention gives for one query?",
        options={
            "exact": "The value of the key that is equal to the query, as a dict does",
            "right": "An average of all values, weighted by how well each key matches",
            "key": "The key that matches the query best, instead of a value",
        },
        reveal="**An average of all values, weighted by how well each key matches.** A dict finds the one key that is exactly equal and gives its value. Attention looks up softly: every key matches the query a little, by a dot product, the softmax turns those matches into weights, and the values are averaged with those weights. The names come from this comparison with looking something up.",
    )
    guess_lookup
    return (guess_lookup,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Self-attention** gives every word three vectors, each made from its embedding by a learned matrix of its own: a **query** $\mathbf{q}_i = \mathbf{x}_i \mathbf{W}_Q$, what the word looks for; a **key** $\mathbf{k}_i = \mathbf{x}_i \mathbf{W}_K$, what it offers to be found by; and a **value** $\mathbf{v}_i = \mathbf{x}_i \mathbf{W}_V$, what it passes on. The score of word $i$ for word $j$ is $\mathbf{q}_i \cdot \mathbf{k}_j$. For all words at once, with one embedding per row of $\mathbf{X}$:
        """),
        L.formula(r"\begin{aligned} [[hidden:Q]] &= [[hidden:\mathbf{X}]] [[param:\mathbf{W}_Q]], \; [[hidden:K]] = [[hidden:\mathbf{X}]] [[param:\mathbf{W}_K]], \; [[hidden:V]] = [[hidden:\mathbf{X}]] [[param:\mathbf{W}_V]] \\ [[hidden:\mathbf{H}]] &= \mathrm{softmax}([[hidden:Q]] [[hidden:K]]^\top) \, [[hidden:V]] \end{aligned}"),
        mo.md(r"""
        The softmax goes along every row, as in [Attention weights from dot products](/course/open/attn-scores?lang=en). Queries and keys must have the same length, $d_k$, to take their dot product; the values may have another length. Because a query and a key come from different matrices, a word no longer has to match itself best, and $QK^\top$ need not be symmetric any more: word $i$ can look for word $j$ without word $j$ looking for word $i$. It is called *self*-attention because the queries, keys and values all come from the same words. The three matrices are learned, like the weights of any layer; the attention weights are worked out anew for every sentence. In PyTorch, each matrix is an `nn.Linear(d, d_k, bias=False)`: `bias=False` leaves out the bias terms, so the layer is only its matrix, and `self.query(x)` gives the queries of all rows of `x` at once. Its `.weight` holds $\mathbf{W}_Q^\top$, of shape $(d_k, d)$, as in [Layers as modules: nn.Linear](/course/open/torch-nn-module?lang=en). `torch.softmax(S, dim=1)` takes the softmax of every row, and subtracts the largest score of every row itself.

        ### New words and symbols

        | Word or symbol | Say it | What it means | In code |
        |---|---|---|---|
        | **query** | | what a word looks for: its embedding times $\mathbf{W}_Q$ | `Q = X @ Wq` |
        | **key** | | what a word offers to be found by: its embedding times $\mathbf{W}_K$ | `K = X @ Wk` |
        | **value** | | what a word passes on: its embedding times $\mathbf{W}_V$ | `V = X @ Wv` |
        | **self-attention** | | attention in which the queries, keys and values all come from the same words | `A @ V` |
        | $\mathbf{W}_Q$, $\mathbf{W}_K$, $\mathbf{W}_V$ | capital W Q, K, V | the three learned matrices, of shape $(d, d_k)$ | `Wq`, `Wk`, `Wv` |
        | $d_k$ | d k | the length of the queries and keys | `d_k` |
        """),
        L.watch_out("A key and a value here are not those of a Python dict. A dict finds the one key that is equal to what you ask, and gives its value; attention gives a weighted average of all values, and every key counts a little. And a value is not one number but a vector: what a word passes on to the words that look at it."),
        mo.md(r"""
        The picture: the attention weights of "a red car", with made-up embeddings of 2 numbers, how much a word describes and how much it is a thing. On top, with the scores $\mathbf{X}\mathbf{X}^\top$ of the last lesson: "red" and "car" each take 0.95 from themselves. Below, with a query matrix that swaps the two numbers, made up by the course, and the embeddings themselves as keys: a word that describes now looks for a thing, and "red" takes 0.95 from "car". A real model learns its matrices.
        """),
        mo.vstack(C.weights_pictures({"plain": "Weights with X Xᵀ", "learned": "Weights with Q Kᵀ", "rows": "word", "cols": "takes from", "decimal": ".",
                                      "plain_alt": "Attention weights of a red car with the scores X X transpose: a takes 0.25, 0.38 and 0.38; red takes 0.03, 0.95 and 0.02; car takes 0.03, 0.02 and 0.95.",
                                      "learned_alt": "Attention weights of a red car with queries and keys: a takes 0.25, 0.38 and 0.38; red takes 0.03, 0.02 and 0.95; car takes 0.03, 0.95 and 0.02."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Self-attention for all words at once, with one embedding per row of `X`:

    1. The queries, keys and values: `X` times each of the three matrices.
    2. The scores: every query with every key, `Q @ K.T`.
    3. The weights: the softmax of every row, without overflow.
    4. The new vectors: the weights times the values, `A @ V`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    "a red car" with the made-up embeddings and matrices of the picture; the values are the embeddings themselves here, to keep it short. The comments with `#` say which step of the recipe each part is. Step 2 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_qkv():
    import numpy as np

    _X = np.array([[0.3, 0.3], [2.0, 0.0], [0.0, 2.0]])
    _Wq = np.array([[0.0, 1.0], [1.0, 0.0]])
    _Wk = np.array([[1.0, 0.0], [0.0, 1.0]])
    _Wv = np.array([[1.0, 0.0], [0.0, 1.0]])
    # 1. queries, keys and values
    _Q, _K, _V = _X @ _Wq, _X @ _Wk, _X @ _Wv
    print(_Q)
    # 2. every query with every key
    _S = _X @ _X.T  # complete: _Q @ _K.T
    # 3. the weights: the softmax of every row
    _S = _S - _S.max(axis=1, keepdims=True)
    _A = np.exp(_S) / np.sum(np.exp(_S), axis=1, keepdims=True)
    print(np.round(_A, 3))
    # 4. the new vectors
    print(np.round(_A @ _V, 3))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The queries swap the two numbers: "red" asks for $(0, 2)$, a thing. As it is, the scores are still `_X @ _X.T`, and "red" takes 0.951 from itself: its new vector is $(1.911, 0.044)$, so it still describes. Completed, with `_Q @ _K.T`, the row of "red" becomes `[0.032 0.017 0.951]`: it takes 0.951 from "car", and its new vector is $(0.044, 1.911)$, close to the value of "car". "car" does the same the other way round. "a" stays as it was: its query, $(0.3, 0.3)$, matches both a little.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_value_shape():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The length of the new vectors
    """)
    return


@app.cell(hide_code=True)
def choice_value_shape_cell():
    choice_value_shape = L.choice(
        "value-shape",
        question="`X` has shape (5, 16). $\\mathbf{W}_Q$ and $\\mathbf{W}_K$ have shape (16, 8), and $\\mathbf{W}_V$ has shape (16, 4). What shape do the new vectors, `A @ V`, have?",
        options={
            "query": "(5, 8): as long as a query",
            "embedding": "(5, 16): as long as an embedding",
            "right": "(5, 4): as long as a value",
        },
        correct="right",
        why={
            "query": "The queries and keys only make the weights, of shape (5, 5); what is averaged are the values.",
            "embedding": "The embeddings are not passed on as they are: every word passes on its value, of 4 numbers.",
            "right": "`A` is (5, 5) and `V` is (5, 4): every word gets an average of the values, so 4 numbers. The length of a query and key, 8, is gone once the scores are made.",
        },
    )
    choice_value_shape
    return (choice_value_shape,)


@app.cell(hide_code=True)
def text_exercise_qkv():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Self-attention in NumPy

        Complete `qkv_attention(X, Wq, Wk, Wv)`: `X` holds one embedding per row, and `Wq`, `Wk` and `Wv` are the three matrices; it gives the new vectors, one per row. Step 3 is given. Replace the five `...` following steps 1, 2 and 4 of the recipe. The check calls `qkv_attention` for "a car is red", with the embeddings of the picture and "is" with the embedding of "a", the same query and key matrices, and a value matrix that halves the second number, `[[1, 0], [0, 0.5]]`, which should give `[[0.7224, 0.3612], [1.8617, 0.0261], [0.7224, 0.3612], [0.0522, 0.9308]]`, and tries other sentences and matrices, also with values of another length.
        """),
        L.task("Make `qkv_attention` give the new vectors from queries, keys and values."),
    ])
    return


@app.cell(hide_code=True)
def exercise_qkv():
    box_qkv = L.code_box("qkv-attention", lines=16)
    box_qkv
    return (box_qkv,)


@app.cell(hide_code=True)
def check_qkv(box_qkv):
    result_qkv = L.check("qkv-attention", box_qkv.value)
    result_qkv
    return (result_qkv,)


@app.cell(hide_code=True)
def hints_qkv():
    L.hints(
        "qkv-attention",
        [
            "Steps 1, 2 and 4 of the completed worked example, with the names of the function.",
            "Every row of `X` is a word, so the matrix comes after `X`. The scores pair every row of `Q` with every row of `K`, as in the second quick-check question.",
        ],
        why="`X @ Wq` gives one query per word, and `Q @ K.T` puts query $i$ against key $j$ at row $i$, column $j$. The averaging is over the values, `A @ V`, whose length may differ from that of the queries.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_module():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Self-attention in PyTorch

        Complete the class `SelfAttention(d, d_k)`: in `__init__` three `nn.Linear` layers from `d` to `d_k` numbers, without bias terms, called `self.query`, `self.key` and `self.value`, made in that order; in `forward` the new vectors of a tensor `x` with one embedding per row, by the recipe, with `torch.softmax(..., dim=1)` for step 3. The check makes the module after `torch.manual_seed(0)`, with `d` 4 and `d_k` 3, gives it 5 random embeddings, and compares the shape and the new vector of the first word; then it tries other sizes, and embeddings of zeros.
        """),
        L.task("Make `SelfAttention` give the new vectors with three learned matrices."),
    ])
    return


@app.cell(hide_code=True)
def exercise_module():
    box_module = L.code_box("self-attention-module", lines=16)
    box_module
    return (box_module,)


@app.cell(hide_code=True)
def check_module(box_module):
    result_module = L.check("self-attention-module", box_module.value)
    result_module
    return (result_module,)


@app.cell(hide_code=True)
def hints_module():
    L.hints(
        "self-attention-module",
        [
            "In `__init__`, the three layers as attributes of `self`, each `nn.Linear(d, d_k, bias=False)`. In `forward`, apply each of them to `x`.",
            "Then the weights, the softmax of the scores along every row, and the weights times the values: steps 2 to 4 of exercise 2, with `torch.softmax` instead of the two lines of step 3.",
        ],
        why="An `nn.Linear` without bias terms is a learned matrix: `self.query(x)` is $\\mathbf{X}\\mathbf{W}_Q$ for all rows at once. Training changes the three matrices, and with them what every word looks for, offers and passes on.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The attention of [Attention weights from dot products](/course/open/attn-scores?lang=en), now with three learned matrices, written as a module with the `nn.Linear` of [Layers as modules: nn.Linear](/course/open/torch-nn-module?lang=en). With queries and keys, a word can look for something other than itself. The dot product of a query and a key grows with their length: with $d_k$ of 64, as in each attention of GPT-2, the scores get large and the softmax very sharp. In the next lesson, *Scaling the scores*, you divide them by $\sqrt{d_k}$ to keep the softmax, and its gradients, working.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_value_shape, result_qkv, result_module):
    L.wrap_up(choice_value_shape, result_qkv, result_module)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
