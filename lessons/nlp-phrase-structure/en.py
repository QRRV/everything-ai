# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-phrase-structure. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-phrase-structure", "en")
    C = L.checks   # the course's grammar, its pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Parts of speech and phrases](/course/open/nlp-word-classes?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_tag():
    quick_tag = L.choice(
        "quick-tag",
        question="Which tag of Universal Dependencies does *after* get in *the train left after midnight*?",
        options={
            "adv": "`ADV`, since it says when",
            "right": "`ADP`, since a noun follows",
            "sconj": "`SCONJ`, as in *after the train left*",
        },
        correct="right",
        why={
            "right": "*after* stands before the noun *midnight*: a preposition, which UD tags `ADP`, as it tags *in* and *of*. In this lesson, `ADP` becomes a symbol of a grammar.",
            "adv": "*after* does say when, but it stands before a noun, *midnight*, and that makes it a preposition: `ADP`. See the table of 17 tags in [Parts of speech and phrases](/course/open/nlp-word-classes?lang=en).",
            "sconj": "With a whole clause after it, as in *after the train left*, *after* is a `SCONJ`. Here only the noun *midnight* follows: a preposition, `ADP`. See the table of 17 tags in [Parts of speech and phrases](/course/open/nlp-word-classes?lang=en).",
        },
    )
    quick_tag
    return (quick_tag,)


@app.cell(hide_code=True)
def quick_check_tags():
    quick_tags = L.choice(
        "quick-tags",
        question="In *Parts of speech and phrases*, `tags_per_word` gives each word in small letters the list of its tags, each tag once, in the order they first come. What does `tags_per_word([[(\"Book\", \"VERB\"), (\"it\", \"PRON\")], [(\"a\", \"DET\"), (\"book\", \"NOUN\")]])` give?",
        options={
            "upper": "`{'Book': ['VERB'], 'it': ['PRON'], 'a': ['DET'], 'book': ['NOUN']}`",
            "last": "`{'book': ['NOUN'], 'it': ['PRON'], 'a': ['DET']}`",
            "right": "`{'book': ['VERB', 'NOUN'], 'it': ['PRON'], 'a': ['DET']}`",
        },
        correct="right",
        why={
            "right": "*Book* and *book* share the key `'book'`, and its list keeps both tags, in the order they came. The same letters, two parts of speech: the words around them decide.",
            "upper": "`tags_per_word` makes each word small with `lower()`, so *Book* and *book* share one key. See Exercise 2 in [Parts of speech and phrases](/course/open/nlp-word-classes?lang=en).",
            "last": "The list keeps every tag of a word once, in the order they came, so `'book'` gets both `VERB` and `NOUN`. See Exercise 2 in [Parts of speech and phrases](/course/open/nlp-word-classes?lang=en).",
        },
    )
    quick_tags
    return (quick_tags,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    *Parts of speech and phrases* gave every word a tag, and marked one noun phrase, *a fantastic time*. Phrases also sit inside larger phrases. In *the train left the city after midnight*, *the city* is a noun phrase, and *left the city* is a larger group around the verb. A program that reads the sentence has to know which words go together. Jurafsky and Martin name uses such as grammar checking (a sentence that cannot be given such a structure may contain an error) and working out what a sentence means.

    The study of grammar is old: Jurafsky and Martin open their chapter on it with Pāṇini, who described the grammar of Sanskrit sometime between the 7th and the 4th century B.C. This lesson writes a grammar as a few rules in Python, and lets the rules build sentences. A handful of rules turns out to make an endless number of sentences, which no list could hold.

    **An everyday example.** A folder on a computer can hold files and other folders, which can hold files and folders again. One simple rule, and nothing in it says how deep the folders may go.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="The Penn Treebank, from *Parts of speech and phrases*, has a tree for every sentence, and you can read rules off those trees. One rule says that a verb phrase (VP), a verb with the words that belong to it, can be a verb followed by a noun phrase, as in *left the city*. Jurafsky and Martin give the number of different rules for a VP in the grammar of the Penn Treebank. About how many are there?",
        options={
            "few": "Fewer than 50",
            "some": "About 500",
            "right": "About 4,500",
        },
        reveal="**About 4,500.** That grammar is very flat, Jurafsky and Martin write: it has separate rules for any number of prepositional phrases (PP) in a row, such as *after midnight*, and for every arrangement of the words around the verb. So it has VP → VBD PP, VP → VBD PP PP, VP → VBD PP PP PP and so on, where VBD is the Penn Treebank's tag for a verb in the past tense. The course adds that two rules could make all of these: VP → VBD PP, and VP → VP PP, which can be used again on what it made itself. This lesson shows how such rules work.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Constituents

    In *the late train left the city after midnight*, some words belong together more closely than others. *the late train* can make way for one word, *it*: *it left the city after midnight*. For *train left*, no single word fits, and the two words cannot move together either. A group of words that acts as one unit in a sentence is a **constituent**. Jurafsky and Martin put it this way: "groups of words can behave as single units, or constituents".

    How do you tell? The textbook *Essentials of Linguistics* (Catherine Anderson and others, 2nd edition, 2022) gives four kinds of test:

    - **Replace** the group with one word. A noun phrase can become *it* or *they*, and a group that says when, such as *after midnight*, can become *then*.
    - **Move** the group: *After midnight, the late train left the city.*
    - **Cleft** the sentence, with the group in *it was ... that*: *It was the city that the late train left.*
    - **Answer** a question with the group alone: *What did the late train leave? The city.*

    Many textbooks add a test with *and*, but the authors call it not always reliable. And a test only tells you about the sentence you test, they write: the same words can be a constituent in one sentence and not in another.

    Dutch school grammar has a test of its own. P.C. Paardekooper's *Kleine ABN-syntaxis* (1961) puts it like this: the largest group that can stand before the *persoonsvorm*, the form of the verb that changes with the subject (*ik verliet*, *wij verlieten*: I left, we left), is one *zinsdeel*, a part of the sentence. In *Na middernacht verliet de late trein de stad*, the group *na middernacht* (after midnight) stands before *verliet*: one part. The test has exceptions, for verbs and for a few short words such as *niet* (not).

    ### Phrases inside phrases

    Constituents sit inside each other. In the grammar of this lesson, *the late train* holds a smaller group, *late train*, and that holds *train*. Draw the groups with the whole sentence at the top and the words at the bottom, and you get a tree: a **parse tree**, as in the picture below. The Penn Treebank has such a tree for every sentence; Universal Dependencies draws another kind of tree, which the next lesson shows.

    ### Rules that build sentences

    A **context-free grammar** describes constituents with rules. Each rule says that the symbol on the left of the arrow may be rewritten as the row of symbols on the right, its **right-hand side**. These are the rules of this lesson:

    ```
    S   → NP VP
    NP  → DET NOM  |  PROPN
    NOM → NOUN  |  ADJ NOM
    VP  → VERB NP  |  VERB NP PP
    PP  → ADP NP
    ```

    Read the arrow, Jurafsky and Martin write, as "rewrite the symbol on the left with the string of symbols on the right". The bar `|` separates choices: an NP is a DET and a NOM, or a PROPN. S stands for a sentence, NP for a noun phrase, VP for a verb phrase, and PP for a prepositional phrase, such as *after midnight*. NOM is the part of a noun phrase after the determiner: a noun, with any adjectives before it. The symbols right above the words are the tags of *Parts of speech and phrases*. A second set of rules, the **lexicon**, gives each tag its words: DET → *the* | *a*, NOUN → *train* | *bridge* | *city*, and so on.

    The symbols for groups, such as NP, are **non-terminals**. The words are **terminals**: there, the rewriting ends. Every sentence starts from one non-terminal, the **start symbol**, here S. In the formal definition of Jurafsky and Martin, a context-free grammar has these four parts: the non-terminals, the terminals, the rules and the start symbol. It is called context-free because a rule rewrites one symbol, whatever stands around it. The rules are the course's own, after the small grammar L0 of Jurafsky and Martin, with words of the course's own.

    ### From S to a sentence

    Start with S, and rewrite one non-terminal at a time, until only words are left. This lesson always rewrites the first non-terminal from the left:

    ```
    S
    NP VP
    DET NOM VP
    the NOM VP
    the ADJ NOM VP
    the late NOM VP
    ...
    the late train left the city
    ```

    The row of steps from S to the sentence is a **derivation**. At many steps there is a choice: NOM can become NOUN, or ADJ NOM. Number the right-hand sides of each symbol 0, 1, 2 and so on, and a list of numbers says which rule to take at each step: this sentence takes 13 steps. The picture shows the same derivation as a tree: each rewrite gives a symbol its children.

    ### Endless sentences

    Look at NOM → ADJ NOM. NOM stands on both sides of the arrow, so the rule can be used again on what it just made: NOM, ADJ NOM, ADJ ADJ NOM, and so on, as in *the late long late train*. There is no longest sentence: one more adjective always makes a longer one. A rule can also come back to its own symbol through other rules. Add VP → VERB *that* S, and a sentence holds a verb phrase that holds a sentence again: *you know that I think that the train left*. That is how a grammar with a few rules and a few words makes an endless number of sentences.

    In Dutch, that last rule does not work as it stands: after *dat* (that), the verb comes at the end, as in *ik weet dat de trein de stad verliet* (I know that the train left the city). A grammar of Dutch needs rules of its own for such a clause.

    ### Who thought of it

    The idea that a sentence breaks up into constituents goes back to the psychologist Wilhelm Wundt (1900), Jurafsky and Martin write. Noam Chomsky made it formal in 1956, as a phrase-structure grammar. Shortly after, by the account of Jurafsky and Martin, John Backus (1959) and, independently, Peter Naur and others (1960) came to the same kind of grammar, to describe the programming language ALGOL. The way they wrote their rules is known as Backus-Naur form, or BNF, and Jurafsky and Martin call it equivalent to a context-free grammar. They also call context-free grammars the backbone of many formal models of the syntax of human languages, and of computer languages too.
    """)
    return


@app.cell(hide_code=True)
def tree_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **constituent** | a group of words that acts as one unit in a sentence, such as *the late train* |
        | **parse tree** | the constituents of a sentence drawn as a tree, with S at the top and the words at the bottom |
        | **context-free grammar** | rules that each rewrite one symbol as a row of symbols, whatever stands around it; abbreviated CFG |
        | **right-hand side** | the row of symbols on the right of the arrow of a rule |
        | **non-terminal**, **terminal** | a symbol for a group, such as NP, and a word, where the rewriting ends |
        | **start symbol** | the non-terminal every sentence starts from, here S |
        | **lexicon** | the rules that give each tag its words |
        | **derivation** | the steps from S to a sentence, each step rewriting one symbol |
        | `GRAMMAR` | the grammar of the lesson as a dict: each non-terminal is a key, with the list of its right-hand sides, each a list of symbols |
        | `a + b` | with two lists: a new list with the items of `a`, then those of `b` |
        | `return` in a loop | ends the function at once, in the middle of the loop |
        """),
        L.watch_out("*Words and their parts* used *derivation* for making a new word from another, such as *careful* from *care*. In a grammar, the word means the steps from S to a sentence: the same word, another meaning."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Write the grammar as a dict: each non-terminal is a key, with the list of its right-hand sides.
    2. To take a step, find the first non-terminal, take one of its right-hand sides, and put that in the place of the symbol.
    3. Start from `["S"]`, take a step for each choice in a list, and keep every row.
    4. Draw the rows as a tree, and look for a rule that comes back to its own symbol.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first cell is the grammar of the lesson as a dict. The second defines `first_open`, which gives back the place of the first non-terminal, or `None` when there is none; then it rewrites one step by hand.
    """)
    return


@app.cell
def grammar_cell():
    GRAMMAR = {
        "S": [["NP", "VP"]],
        "NP": [["DET", "NOM"], ["PROPN"]],
        "NOM": [["NOUN"], ["ADJ", "NOM"]],
        "VP": [["VERB", "NP"], ["VERB", "NP", "PP"]],
        "PP": [["ADP", "NP"]],
        "DET": [["the"], ["a"]],
        "ADJ": [["late"], ["long"]],
        "NOUN": [["train"], ["bridge"], ["city"]],
        "PROPN": [["Boston"], ["Chicago"]],
        "VERB": [["passed"], ["left"]],
        "ADP": [["near"], ["after"]],
    }
    return (GRAMMAR,)


@app.cell
def example_first(GRAMMAR):
    def first_open(symbols, grammar):
        for i in range(len(symbols)):
            if symbols[i] in grammar:
                return i
        return None

    print(first_open(["the", "NOM", "VP"], GRAMMAR))
    print(first_open(["the", "late", "train"], GRAMMAR))
    print(["the"] + GRAMMAR["NOM"][1] + ["VP"])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    *the* is a word, so the first non-terminal of `["the", "NOM", "VP"]` is at place 1. In `["the", "late", "train"]` every symbol is a word: nothing is left to rewrite, and `first_open` gives back `None`. The `return` inside the loop stops the function at the first non-terminal it finds. The last line put the second right-hand side of NOM, `["ADJ", "NOM"]`, in the place of NOM, by hand. Exercise 1 does that at any place.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_rewrite():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. rewrite_first: one step

        Write `rewrite_first(symbols, grammar, k)`: find the place `i` of the first non-terminal in `symbols` with `first_open`, take its right-hand side number `k` from `grammar`, and give back a new list: the symbols before place `i`, then that right-hand side, then the symbols after place `i`. `symbols` always has a non-terminal here. The check gives your function `first_open`, the function of the worked example. Replace the `...`. The check calls `rewrite_first(["NP", "VP"], GRAMMAR, 0)`; then cases of its own, also with small grammars of letters.
        """),
        L.task("Put right-hand side number `k` in the place of the first non-terminal."),
    ])
    return


@app.cell(hide_code=True)
def exercise_rewrite():
    box_rewrite = L.code_box("rewrite_first", lines=4)
    box_rewrite
    return (box_rewrite,)


@app.cell(hide_code=True)
def check_rewrite(box_rewrite):
    result_rewrite = L.check("rewrite_first", box_rewrite.value)
    result_rewrite
    return (result_rewrite,)


@app.cell(hide_code=True)
def hints_rewrite():
    L.hints(
        "rewrite_first",
        [
            "Three lines: find the place with `first_open`, look up the right-hand side in `grammar`, and glue three lists together with `+`.",
            "Which symbol must not come back in the new list? And which part of `symbols` comes after the right-hand side?",
            "`    return symbols[:i] + right + ___`",
        ],
        why="`i = first_open(symbols, grammar)`, then `right = grammar[symbols[i]][k]`, and `return symbols[:i] + right + symbols[i + 1:]`. `rewrite_first([\"NP\", \"VP\"], GRAMMAR, 0)` gives `['DET', 'NOM', 'VP']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_rewrite():
    mo.md(r"""
    In the picture under the check, your function takes the 13 steps of the sentence of *The idea*. The words appear from left to right, because each step rewrites the first non-terminal: *the late train* is done before the verb phrase begins.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_derive():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. derive: from S to a sentence

        Write `derive(grammar, choices)`: start with `symbols`, the list `["S"]`, and a list `steps` that holds this first row. For each number `k` in `choices`, rewrite `symbols` with `rewrite_first`, and append the new row to `steps`. Give back `steps` at the end. The check gives your function the course's `rewrite_first`. Replace the `...`. The check calls `derive({"S": [["a", "S", "b"], ["c"]]}, [0, 0, 1])` first: a grammar with a rule that comes back to S, which makes *c*, *a c b*, *a a c b b* and so on. Then it tries a grammar for sums, such as *x + ( y )*, no choices at all, and the grammar of the lesson.
        """),
        L.task("Rewrite `[\"S\"]` once for each choice, and keep every row."),
    ])
    return


@app.cell(hide_code=True)
def exercise_derive():
    box_derive = L.code_box("derive", lines=7)
    box_derive
    return (box_derive,)


@app.cell(hide_code=True)
def check_derive(box_derive):
    result_derive = L.check("derive", box_derive.value)
    result_derive
    return (result_derive,)


@app.cell(hide_code=True)
def hints_derive():
    L.hints(
        "derive",
        [
            "Make the two lists before the loop. Inside the loop, two lines: one rewrites, one appends.",
            "What should `derive(grammar, [])` give back, with no choices at all?",
            "`    steps = ___`",
        ],
        why="`symbols = [\"S\"]` and `steps = [symbols]`, then `for k in choices:` with `symbols = rewrite_first(symbols, grammar, k)` and `steps.append(symbols)` under it; after the loop `return steps`. `derive({\"S\": [[\"a\", \"S\", \"b\"], [\"c\"]]}, [0, 0, 1])` gives `[['S'], ['a', 'S', 'b'], ['a', 'a', 'S', 'b', 'b'], ['a', 'a', 'c', 'b', 'b']]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_derive():
    mo.md(r"""
    The tree under the check belongs to a longer sentence of the grammar, with 17 choices. A noun phrase sits inside the verb phrase, and another inside the prepositional phrase *near Chicago*: phrases inside phrases. The grammar of letters did the same with S itself: *a a c b b* has an S inside an S inside an S.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: endless sentences

    The table counts the sentences that the grammar of the lesson makes, by their number of words. Switch the rule NOM → ADJ NOM off, and see the counts stop. You can also change `GRAMMAR` in the cell of the worked example, such as by adding a word to the lexicon; the table follows.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_rule = mo.ui.checkbox(value=True, label="With the rule NOM → ADJ NOM")
    demo_words = mo.ui.slider(3, 14, value=10, label="Words, at most", show_value=True)
    demo_new = mo.ui.button(value=0, on_click=lambda n: n + 1, label="Other examples")
    mo.hstack([demo_rule, demo_words, demo_new], justify="start", gap=1.5, wrap=True)
    return demo_new, demo_rule, demo_words


@app.cell(hide_code=True)
def demo_look(GRAMMAR, demo_new, demo_rule, demo_words):
    mo.Html(C.demo_picture(GRAMMAR, demo_rule.value, demo_words.value, demo_new.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_constituents():
    mo.md(r"""
    ### Exercise 3. Which groups are constituents?

    Take the sentence *The old ferry left the harbor before dawn.* Put each group of words under constituent or not a constituent. Try the tests of *The idea* on each group.
    """)
    return


@app.cell(hide_code=True)
def exercise_constituents():
    sort_constituents = L.sort("constituents")
    sort_constituents
    return (sort_constituents,)


@app.cell(hide_code=True)
def check_constituents(sort_constituents):
    result_constituents = L.check("constituents", sort_constituents)
    result_constituents
    return (result_constituents,)


@app.cell(hide_code=True)
def text_exercise_endless():
    mo.md(r"""
    ### Exercise 4. One rule more

    Take the grammar of the lesson without NOM → ADJ NOM. Then no rule comes back to its own symbol, and the grammar makes a limited number of sentences, as the demo showed.
    """)
    return


@app.cell(hide_code=True)
def choice_endless_cell():
    choice_endless = L.choice(
        "endless",
        question="Which one rule, added to that grammar, makes the number of sentences endless again?",
        options={
            "right": "NP → DET NOM PP",
            "vp": "VP → VERB NP PP PP",
            "pp": "PP → ADP DET NOUN",
            "s": "S → PROPN VP",
        },
        correct="right",
        why={
            "right": "A noun phrase now holds a PP, and PP → ADP NP holds a noun phrase again: NP comes back to NP through PP, as in *the bridge near the city near the bridge near ...*. A loop through two rules is as endless as a loop in one rule.",
            "vp": "VP → VERB NP PP PP makes longer sentences, but neither NP nor PP leads back to VP, so there is still a longest sentence. Look for a symbol that comes back to itself. See *Endless sentences* in *The idea*.",
            "pp": "PP → ADP DET NOUN ends in tags that each become one word, so no symbol comes back to itself through it. See *Endless sentences* in *The idea*.",
            "s": "PROPN and VP do not lead back to S, so S → PROPN VP makes new sentences, but not endless ones. See *Endless sentences* in *The idea*.",
        },
    )
    choice_endless
    return (choice_endless,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Parts of speech and phrases](/course/open/nlp-word-classes?lang=en) gave every word a tag, and found a first phrase. Here the tags became the symbols right above the words of a tree. Constituents are groups of words that act as one unit, as tests of replacing, moving and answering show; they sit inside each other, and a parse tree draws them. A context-free grammar describes them with rules that rewrite one symbol at a time, from S to a sentence. You wrote one step and a whole derivation, and saw that a rule that comes back to its own symbol, directly or through other rules, makes no end of sentences. The next lesson draws another kind of tree, in which each word hangs from the word it depends on.

    **An everyday example.** Your ancestors are your parents, and the ancestors of your parents: one rule that uses itself, and it reaches back as far as you like.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_endless, result_constituents, result_derive, result_rewrite):
    L.wrap_up(result_rewrite, result_derive, result_constituents, choice_endless)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
