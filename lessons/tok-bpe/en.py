# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson tok-bpe. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw, play
    from everything_ai.lesson import lesson

    L = lesson("tok-bpe", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en) and [Dictionaries](/course/open/py-dicts?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rare():
    quick_rare = L.choice(
        "quick-rare",
        question="The tokenizer of GPT-2 gets a rare Dutch word that is not in its vocabulary as a whole. What does it do with it?",
        options={
            "unknown": "It gives it one special token that means: unknown word",
            "skip": "It leaves the word out",
            "right": "It cuts it into several pieces that are in its vocabulary",
        },
        correct="right",
        why={
            "unknown": "GPT-2 has no such token: every text becomes tokens of its vocabulary. See [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en).",
            "skip": "Then the model could not read or write that word. It falls apart into smaller tokens instead. See [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en).",
            "right": "Today you learn how such a vocabulary of pieces is made, and you make one yourself.",
        },
    )
    quick_rare
    return (quick_rare,)


@app.cell(hide_code=True)
def quick_check_key():
    quick_key = L.choice(
        "quick-key",
        question="What happens with these lines?\n\n```python\ncounts = {}\ncounts[\"ab\"] = counts[\"ab\"] + 1\n```",
        options={
            "one": "`counts` becomes `{\"ab\": 1}`",
            "right": "It stops with an error",
            "zero": "`counts` becomes `{\"ab\": 0}`",
        },
        correct="right",
        why={
            "one": "A key that is not there cannot be looked up, so `counts[\"ab\"]` stops before anything is added. See [Dictionaries](/course/open/py-dicts?lang=en).",
            "zero": "A dictionary has no value 0 for keys it does not have: looking up `\"ab\"` stops. See [Dictionaries](/course/open/py-dicts?lang=en).",
            "right": "`\"ab\"` is not a key yet, and looking it up stops the program. That is why a count starts with `if pair in counts:`, as when you counted pairs of characters; today you count pairs of tokens.",
        },
    )
    quick_key
    return (quick_key,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Why this matters

        The vocabulary of a tokenizer is made from text. Single characters are too small: every word becomes many tokens, so a model sees less text at once and must learn every word from its letters. Whole words are too many, and a word that is not in the list could not be read or written at all. **Byte pair encoding** sits in between: start with the characters, and keep merging the pair that comes most often into one new token. Frequent words end up as one token, rare words stay in pieces. GPT-2 made its vocabulary this way. In this project step you build it yourself: the first part of your own tokenizer.
        """),
        draw.project_steps(["Count letters and pairs", "A bigram model by counting", "The same model, learned by a neural network",
                            "Your own tokenizer", "Your own GPT", "Pretrain a small LLM", "Teach it to follow instructions and preferences"],
                           3, here_label="← today", title="Where your project stands: your own LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_first():
    guess_first = L.guess(
        "first-merge",
        question="Your BPE will learn 100 merges from the first 20,000 characters of Tiny Shakespeare. Which pair of characters do you think it merges first?",
        options={
            "th": "`t` and `h`, as in the, that and thou",
            "ou": "`o` and `u`, as in you, your and thou",
            "right": "`e` and a space, the end of many words",
        },
        reveal="**`e` and a space**: 517 times in those 20,000 characters, more than any other pair. Many words end in e, such as the, be, me, are and have, and a space follows. After the first merge, `t` and `h` come most often, 402 times, and become the second merge. BPE counts what is there, spaces included.",
    )
    guess_first
    return (guess_first,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Byte pair encoding** (BPE) learns tokens from a text. Start with the text as a list of characters, `list(text)`: the first tokens. Count every pair of neighboring tokens, and merge the pair that comes most often into one new token, everywhere in the text. That pair is a **merge rule**. Then count again, with the new token among the others, and merge again: once for every new token you want.

        Take "aaabdaaabac". The pair (`a`, `a`) comes 4 times, more than any other, so it becomes `aa`: `aa a b d aa a b a c`. Now (`aa`, `a`) and (`a`, `b`) both come twice; take the first one, so `aaa`. Then (`aaa`, `b`) becomes `aaab`: `aaab d aaab a c`, 5 tokens instead of 11. The rules build on each other: `aaab` came from three merges.

        To tokenize a new text, start from its characters and apply the merge rules in the order they were learned. The characters stay in the vocabulary, so any text of known characters can be tokenized: a word BPE never saw stays in smaller pieces. GPT-2 learned 50,000 merge rules this way. A pair of tokens is a tuple, such as `("aa", "a")`, and a tuple can be a key of a dictionary, like a string. Gluing the two strings into one key would not do: `"a"` and `"ab"` give the same `"aab"` as `"aa"` and `"b"`.

        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **byte pair encoding** (BPE) | learning tokens: start with characters, and merge the most frequent pair of neighboring tokens into a new token, again and again | `train_bpe(text, n)` |
        | **merge rule** | one learned step of BPE: this pair of tokens becomes one token; a tokenizer applies them in the order they were learned | `("t", "h")` |
        | a tuple as a key | a pair of tokens can be a key of a dictionary, as a string can | `counts[("t", "h")]` |
        """),
        L.watch_out("Merge from left to right, and go on after both tokens of a merged pair: `a a a` with the rule (`a`, `a`) gives `aa a`, not `aa aa`. In one pass, a token can only be in one merge."),
        mo.md(r"""
        The picture: a line of Hamlet, tokenized with the first $n$ merge rules that the course learned from the first 20,000 characters of Tiny Shakespeare. ␣ is a space, ↵ a new line. Move the slider, or drag across the chart, to set $n$: the chart shows how many tokens the line has for every $n$, and the sentence under it which pair the newest rule merges.
        """),
    ])
    return


@app.cell(hide_code=True)
def explore_rules_cell():
    def _frame(n):
        tokens = C.line_tokens(n)
        rules = "1 merge rule" if n == 1 else f"{n} merge rules"
        said = f"""With {rules}, the line has <span class="ea-num">{len(tokens)}</span> tokens."""
        if n:
            a, b = C.rule(n)
            said += f" Rule {n} merges `{C.shown(a)}` and `{C.shown(b)}` into `{C.shown(a + b)}`."
        return (C.line_frame(n, {"title": "One line, more and more merged", "row": rules + ": {k} tokens",
                                 "alt": f"To be, or not to be, that is the question, in {len(tokens)} tokens after {rules}: {' '.join(C.shown(t) for t in tokens)}.",
                                 "count_title": "Tokens in the line, for every n", "x": "merge rules", "y": "tokens",
                                 "count_alt": "How many tokens the line has for every number of merge rules from 0 to 100: it falls from 41 to 20, most of it in the first 40 rules."}),
                said)

    explore_rules = L.explore("merge-rules", "n", _frame, start=10, low=0, high=100, step=1, role="hyper", task="At which merge rule does `the␣` become one token, and from which two tokens? Why does `be` wait until rule 78, though the line has it twice?")
    explore_rules
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    At rule 19, from `th` and `e␣`, the tokens of rules 2 and 1: the rules build on each other. `be` waits until rule 78 because BPE learns from the first 20,000 characters of Tiny Shakespeare, not from this line: there `e␣` comes 517 times and `b` `e` only 63, so 77 other merges come first. Even after 100 rules, "question" is not one token: it ends as `q` `u` `es` `t` `i` `on`.

    ### Recipe

    Learning $n$ merge rules from a text:

    1. The text as a list of characters: the first tokens.
    2. Count every pair of neighboring tokens; the most frequent pair, the first one if there is a tie, is the next merge rule.
    3. Merge that pair everywhere, from left to right: the two tokens become one string, and the next pair starts after them.
    4. Steps 2 and 3 again, $n$ times in all.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    One round of BPE on "aaabdaaabac". The comments with `#` say which step of the recipe each part is. Step 3 is not finished: run the cell as it is, then complete it where the comment says, and run it again.
    """)
    return


@app.cell
def example_round():
    # 1. the characters: the first tokens
    _tokens = list("aaabdaaabac")
    # 2. count every pair of neighbors, and take the most frequent, the first one if there is a tie
    _counts = {}
    for _i in range(len(_tokens) - 1):
        _pair = (_tokens[_i], _tokens[_i + 1])
        if _pair in _counts:
            _counts[_pair] = _counts[_pair] + 1
        else:
            _counts[_pair] = 1
    print(_counts)
    _best = None
    _most = 0
    for _pair in _counts:
        if _counts[_pair] > _most:
            _best = _pair
            _most = _counts[_pair]
    print(_best)
    # 3. merge it everywhere, from left to right
    _out = []
    _i = 0
    while _i < len(_tokens):
        if _i < len(_tokens) - 1 and _tokens[_i] == _best[0] and _tokens[_i + 1] == _best[1]:
            # complete: add _tokens[_i] + _tokens[_i + 1], and go on with _i = _i + 2
            _out.append(_tokens[_i])
            _i = _i + 1
        else:
            _out.append(_tokens[_i])
            _i = _i + 1
    print(_out)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The counts are `{('a', 'a'): 4, ('a', 'b'): 2, ('b', 'd'): 1, ('d', 'a'): 1, ('b', 'a'): 1, ('a', 'c'): 1}`, so the best pair is `('a', 'a')`. As it is, the merged branch adds one token and moves on by one, so `_out` is the 11 characters again. Completed, it is `['aa', 'a', 'b', 'd', 'aa', 'a', 'b', 'a', 'c']`: 9 tokens, and in `aaa` only the first two are merged.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_order():
    mo.md(r"""
    ## Your turn

    ### Exercise 1. The order of the rules
    """)
    return


@app.cell(hide_code=True)
def choice_order_cell():
    choice_order = L.choice(
        "order",
        question="A BPE learned two merge rules: first (`h`, `e`), then (`t`, `h`). Into which tokens does it cut the word \"the\"?",
        options={
            "right": "`t` and `he`: (`h`, `e`) goes first",
            "the": "`the`: both rules fit, so it becomes one token",
            "th": "`th` and `e`: the rule (`t`, `h`) makes `th`, and `e` stays alone",
        },
        correct="right",
        why={
            "the": "To become `the`, it would need a rule (`t`, `he`) or (`th`, `e`), and it has neither. After (`h`, `e`), there is no `h` token left for (`t`, `h`).",
            "th": "The rules go in the order they were learned. (`h`, `e`) was learned first, so it goes first, and takes the `h`.",
            "right": "(`h`, `e`) makes `he`, and then `t` and `h` are no longer neighbors. A tokenizer applies its rules in the order it learned them, every time, so the same text always gets the same tokens.",
        },
    )
    choice_order
    return (choice_order,)


@app.cell(hide_code=True)
def text_exercise_merge():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. Merge a pair

        Complete `merge(tokens, pair)`: it gives a new list of tokens in which every place where `pair[0]` is followed by `pair[1]` is one token, the two strings put together: step 3 of the recipe. The loop and the `if` are given. The check calls `merge(["l", "o", "w", " ", "l", "o", "w", "e", "r"], ("l", "o"))`, which should give `["lo", "w", " ", "lo", "w", "e", "r"]`, and tries other pairs. When it is right, the check saves it in your project, as `work/projects/llm/merge.py`.
        """),
        L.task("Make `merge` turn every place of `pair` into one token."),
    ])
    return


@app.cell(hide_code=True)
def exercise_merge():
    box_merge = L.code_box("merge", lines=13)
    box_merge
    return (box_merge,)


@app.cell(hide_code=True)
def check_merge(box_merge):
    result_merge = L.check("merge", box_merge.value)
    result_merge
    return (result_merge,)


@app.cell(hide_code=True)
def hints_merge():
    L.hints(
        "merge",
        [
            "Under the `if`, two tokens become one: put the two strings together with `+`, add that to `out`, and skip over both. Under `else`, the token goes to `out` as it is.",
            "Step 3 of the completed worked example, with `tokens`, `pair` and `i` instead of `_tokens`, `_best` and `_i`.",
        ],
        why="Going on after both tokens means merges never overlap: in `[\"a\", \"a\", \"a\"]`, the first two become `\"aa\"`, and the third stays alone.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_train(result_merge):
    result_merge  # the note changes when exercise 2 saves your merge
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Learn the merge rules

        Complete `train_bpe(text, n)`: it learns `n` merge rules from `text` and gives them as a list of pairs, in the order it learned them: the recipe. `pair_counts` and the loop are given; the box's first line imports your `merge` from your project. The check calls `train_bpe("low lower lowest", 3)`, which should give `[("l", "o"), ("lo", "w"), (" ", "low")]`, and tries other texts. When it is right, the check saves it as `work/projects/llm/bpe.py`.
        """),
        L.course_versions(),
        L.task("Make `train_bpe` learn `n` merge rules, each from the tokens of the round before."),
    ])
    return


@app.cell(hide_code=True)
def exercise_train():
    box_train = L.code_box("train-bpe", lines=26)
    box_train
    return (box_train,)


@app.cell(hide_code=True)
def check_train(box_train, result_merge):
    result_merge  # check again when exercise 2 saves your merge
    result_train = L.check("train-bpe", box_train.value)
    result_train
    return (result_train,)


@app.cell(hide_code=True)
def hints_train():
    L.hints(
        "train-bpe",
        [
            "Finding `best` is step 2 of the worked example, with `counts`: start with `best = None` and a highest count of 0, and let a pair take both over only when its count is higher.",
            "After `merges.append(best)`, merge the pair with your `merge`, and keep what it gives in `tokens`, for the next round.",
        ],
        why="Every round counts the pairs of the tokens of the round before, so a rule can build on an earlier one: (`l`, `o`) and then (`lo`, `w`) make \"low\" one token.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### What your project can do now

    The cell below uses your project. It trains your BPE on the first 20,000 characters of Tiny Shakespeare, 100 merge rules, and then tokenizes with your `merge`: what you type, and the next 5,000 characters, which it did not learn from. It works once exercises 2 and 3 are done, and takes about a second.
    """)
    return


@app.cell(hide_code=True)
def demo_bpe(result_merge, result_train):
    result_merge, result_train  # run again when exercise 2 or 3 saves your module
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Finish exercises 2 and 3 first: then your `merge` and your `train_bpe` are in your project, and this cell uses them.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*Your BPE takes too long here: more than a minute. Does your `train_bpe` or your `merge` do more work than it needs, such as a loop that runs too often?*",
            "bpe": "*`merge` or `train_bpe` in `work/projects/llm/` stops with an error. Did you change its file? Delete it, and check exercise 2 or 3 again: its check then saves your code again.*",
        }[_result])
    else:
        _rules = ", ".join(f"`{C.shown(a)}` + `{C.shown(b)}`" for a, b in _result["merges"][:10])
        _who = "" if not _result["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", as the note at exercise 3 says."
        _out = mo.vstack([
            mo.md(f"""Your first 10 merge rules: {_rules}. ␣ is a space, ↵ a new line. Type a text below: your BPE tokenizes it as you type."""),
            mo.ui.anywidget(play.Tokens(*C.tokenizer(_result), text=C.LINE, examples=[C.LINE, "First Citizen:\nWe are accounted poor citizens", "Long live the king!"],
                                        decimal="," if L.lang == "nl" else ".", labels={"task_kick": L.w["task"], "task": "Type <code>Citizen</code> and <code>king</code>: which one does your tokenizer make one token? Why that one?",
                                                "box": "A text to tokenize", "again": "Another example", "chars": "characters", "tokens": "tokens",
                                                "per": "characters per token", "token": "Token", "bytes": "bytes", "char": "character", "one": "token", "byte": "byte"})),
            mo.md(f"""`Citizen` is one token, and `king` two: `k` and `ing`. The 20,000 characters your BPE learned from are the start of a play, *Coriolanus*, where "First Citizen:" alone comes 23 times. BPE learns what is frequent in the text it learns from, not in English as a whole.

On the 5,000 characters after them, which it did not learn from, a token holds {C.number(_result['per_token'], 2, L.lang)} characters on average. Many tokens end in a space, such as `to␣`: here the space sticks to the word before it. GPT-2 first cuts a text into words, each with the space before it, and merges only within them; that is why its tokens start with a space, such as `␣the`.{_who}"""),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** You counted pairs of characters in [A bigram model by counting](/course/open/lm-bigram-counts?lang=en); BPE counts pairs of tokens, and turns the most frequent into new tokens. That is how the vocabulary of [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en) was made, from far more text. Your project `llm` now has `merge.py` and `bpe.py`. Next: *Byte-level BPE*, the same on the bytes of a text instead of its characters, so that any text, in any language or with emoji, can be tokenized.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_order, result_merge, result_train):
    L.wrap_up(choice_order, result_merge, result_train)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
