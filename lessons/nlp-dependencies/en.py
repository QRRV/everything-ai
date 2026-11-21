# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-dependencies. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-dependencies", "en")
    C = L.checks   # the course's sentences, the treebanks, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Phrase structure and endless sentences](/course/open/nlp-phrase-structure?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rewrite():
    quick_rewrite = L.choice(
        "quick-rewrite",
        question="In *Phrase structure and endless sentences*, `rewrite_first(symbols, grammar, k)` puts right-hand side number `k` in the place of the first non-terminal. The grammar of that lesson has DET → *the* | *a*. What does `rewrite_first([\"DET\", \"NOM\", \"VP\"], GRAMMAR, 1)` give?",
        options={
            "right": "`['a', 'NOM', 'VP']`",
            "first": "`['the', 'NOM', 'VP']`",
            "kept": "`['DET', 'a', 'NOM', 'VP']`",
        },
        correct="right",
        why={
            "right": "The right-hand sides count from 0, so number 1 of DET is `['a']`, and it takes the place of DET. In this lesson the words of a sentence get numbers too, and those count from 1.",
            "first": "The right-hand sides count from 0: number 0 of DET is `['the']`, and number 1 is `['a']`. See Exercise 1 in [Phrase structure and endless sentences](/course/open/nlp-phrase-structure?lang=en).",
            "kept": "The right-hand side takes the place of DET, so DET itself is gone from the new list. See Exercise 1 in [Phrase structure and endless sentences](/course/open/nlp-phrase-structure?lang=en).",
        },
    )
    quick_rewrite
    return (quick_rewrite,)


@app.cell(hide_code=True)
def quick_check_vp():
    quick_vp = L.choice(
        "quick-vp",
        question="The grammar of *Phrase structure and endless sentences* has S → NP VP and VP → VERB NP. Which words does the VP cover in *the late train left the city*?",
        options={
            "right": "*left the city*",
            "np": "*the city*",
            "subject": "*the late train left*",
        },
        correct="right",
        why={
            "right": "The VP is the verb with the noun phrase after it: *left the city*. In this lesson, the verb *left* becomes the center of the whole sentence.",
            "np": "*the city* is the NP inside the VP, and VP → VERB NP puts the verb before it: *left the city*. See *Phrases inside phrases* in [Phrase structure and endless sentences](/course/open/nlp-phrase-structure?lang=en).",
            "subject": "*the late train* is the NP of S → NP VP, which comes before the VP. The VP is the verb with the noun phrase after it. See *Phrases inside phrases* in [Phrase structure and endless sentences](/course/open/nlp-phrase-structure?lang=en).",
        },
    )
    quick_vp
    return (quick_vp,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    *Phrase structure and endless sentences* put the words of a sentence into phrases inside phrases. There is another way to draw the structure of a sentence: link each word to the word it depends on. In *the late train left the city after midnight*, *late* says what kind of train, *train* is what left, and *city* is what it left. Such links say who did what: Jurafsky and Martin call them a good proxy for how a verb and the words that go with it are linked in meaning. That is an important reason, they write, why grammars of such links are currently more common in natural language processing than grammars of phrases. They also cope better with a free order of words, as in Czech: one kind of link can cover a relation wherever the word stands, where a grammar of phrases needs a separate rule for each place.

    Universal Dependencies draws these trees by the same guidelines for many languages. *Parts of speech and phrases* used two of its treebanks, one for English and one for Dutch, for their tags. This lesson reads their trees.

    **An everyday example.** An organization chart: everyone has one manager, except the person at the top. From anyone, you can follow the managers up, step by step, to the top.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Take every word that the English treebank of Universal Dependencies marks as the object of a verb, such as *city* in *left the city*. About 3 in 100 of those objects come before their verb. In the Dutch treebank of this lesson, about how many?",
        options={
            "same": "About 3 in 100, as in English",
            "some": "About 25 in 100",
            "right": "About 63 in 100",
        },
        reveal="**About 63 in 100**: 4,289 of the 6,818 objects of a verb in the Dutch treebank come before it, against 325 of 9,451 in the English one, by the course's own count. Dutch often puts the verb late. In *de trein heeft de stad verlaten* (the train has left the city), the object *stad* comes before the verb *verlaten*; and after *dat* (that), the verb comes at the end, as *Phrase structure and endless sentences* showed. Whatever the order, *stad* is the object of *verlaten*: the link between the two words stays the same.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Heads and dependents

    Draw an arrow from *train* to *late*: *late* depends on *train*, since it says what kind of train. The word the arrow comes from is the **head**, and the word it points at is the **dependent**. Jurafsky and Martin put it this way: "The head plays the role of the central organizing word, and the dependent as a kind of modifier." Each arrow gets a label that says what the dependent is to its head: *late* is an adjective that modifies *train*, label `amod`. A link from a head to a dependent, with its label, is a **dependency relation**. People write it as the label with the head and the dependent: `amod(train, late)`.

    ### A tree of words

    In a **dependency tree**, every word has one head, except one: the **root**, the center of the sentence, often its main verb. The root gets an arrow from above, labeled `root`, from ROOT: a point above the sentence that is not a word. Jurafsky and Martin give three conditions, with ROOT counted as a point of the tree: no arrow goes into ROOT; every word has exactly one arrow coming in; and from ROOT there is one path to each word. So from any word you can follow the heads up, step by step, and you reach the root.

    The parse tree of the previous lesson had symbols for groups, such as NP and VP. A dependency tree has none: only the words, and the links between them. The picture shows the tree of the sentence, as Universal Dependencies draws it.
    """)
    return


@app.cell(hide_code=True)
def tree_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_ud():
    mo.md(r"""
    ### Universal Dependencies

    Which word is the head, and which labels exist? **Universal Dependencies** (UD) fixes that with guidelines that are the same for every language: version 2 has 37 universal relations. The first guidelines came out in October 2014, and the first release, on 15 January 2015, had 10 treebanks in 10 languages. Release 2.19, of 15 November 2026, has 402 treebanks in 226 languages, by the download page of UD; its home page still says over 200 treebanks in over 150 languages. UD grew out of older work, including the Stanford dependencies and the universal tags of Google; the tags of *Parts of speech and phrases* are the tags of UD. These are the relations of the picture:

    | Label | The dependent is | In the sentence |
    |---|---|---|
    | `nsubj` | the subject: often who or what does something | `nsubj(left, train)` |
    | `obj` | the object: often what the action is done to | `obj(left, city)` |
    | `det` | a determiner of a noun | `det(train, the)` |
    | `amod` | an adjective that modifies a noun | `amod(train, late)` |
    | `obl` | a noun phrase that belongs to a verb and is neither subject nor object, such as one with a preposition | `obl(left, midnight)` |
    | `case` | a preposition, which hangs from its noun | `case(midnight, after)` |
    | `root` | the root of the sentence | `root(ROOT, left)` |

    ### Prepositions hang from their noun

    One choice of UD surprises many people. In *after midnight*, *midnight* is the head, and the preposition *after* the dependent. The guidelines say: "Case-marking elements are treated as dependents of the noun they attach to or introduce." So *midnight* hangs from *left*, with `obl`, and *after* hangs from *midnight*, with `case`.

    ### The files: CoNLL-U

    The treebanks of UD are text files in the format **CoNLL-U**: one word per line, with 10 columns separated by tabs, and an empty line between two sentences. A column without a value holds `_`. These are the columns this lesson uses:

    - ID, the number of the word, counted from 1 in each sentence;
    - FORM, the word itself, and UPOS, its tag;
    - HEAD, the number of its head, with 0 for the root;
    - DEPREL, the relation to its head.

    The sentence of the picture, in those five columns:

    ```
    ID  FORM      UPOS  HEAD  DEPREL
    1   the       DET   3     det
    2   late      ADJ   3     amod
    3   train     NOUN  4     nsubj
    4   left      VERB  0     root
    5   the       DET   6     det
    6   city      NOUN  4     obj
    7   after     ADP   8     case
    8   midnight  NOUN  4     obl
    ```

    In this lesson, a sentence is a list, and each word in it a tuple `(form, head, relation)`. Word number `i` stands at place `i - 1` of the list: the words count from 1, and the places in a list from 0. The relations and heads of the sentences of this lesson are the course's own, by the guidelines of UD.

    ### Who thought of it

    Dependency grammar is old. Jurafsky and Martin trace it back to Pāṇini, sometime between the 7th and the 4th century B.C., and to the grammarians of ancient Greece. Modern dependency grammar builds on the French linguist Lucien Tesnière (1893–1954), whose book *Éléments de syntaxe structurale* came out in 1959, after the author's death. Jurafsky and Martin open their chapter with a line from it, in their translation: "Between each word in a sentence and its neighbors, the mind perceives connections. These connections together form the scaffolding of the sentence."
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **head**, **dependent** | the two words of a link: the central word, and the word that says something about it |
        | **dependency relation** | a link from a head to a dependent, with a label such as `nsubj` |
        | **dependency tree** | the links of a sentence: every word has one head, except the root |
        | **root** | the one word without a head, often the main verb; in CoNLL-U its HEAD is 0 |
        | **Universal Dependencies** | guidelines for tags and trees that are the same for many languages, with treebanks that follow them; abbreviated UD |
        | **CoNLL-U** | the file format of UD: a word per line, with 10 columns |
        | `(form, head, relation)` | a word of a sentence in this lesson: the word, the number of its head, and the label |
        | `form, head, relation = word` | gives each of the three parts of a tuple its own name, as with a pair |
        | `for step in range(n)` | a loop that runs at most `n` times; a `return` inside it can end it sooner |
        """),
        L.watch_out("*Words and their parts* used *root* for the morpheme with the main meaning of a word, such as *work* in *worked*. In a dependency tree, the root is the word of a sentence that has no head: the same word, another meaning."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Write the sentence as a list of tuples `(form, head, relation)`: word number `i` at place `i - 1`, and head 0 for the root.
    2. To find the dependents of a word, go over all the words, and keep those whose head is that word.
    3. To go up from a word, look up its head, then the head of that one, until you reach the word with head 0.
    4. Read the relations off the labels: `nsubj` and `obj` of the verb say who did what.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The first cell is the sentence of the picture as a list. The second defines `head_of`, which gives back the word that is the head of word number `i`, or `"ROOT"` for the root.
    """)
    return


@app.cell
def sentence_cell():
    SENTENCE = [
        ("the", 3, "det"),
        ("late", 3, "amod"),
        ("train", 4, "nsubj"),
        ("left", 0, "root"),
        ("the", 6, "det"),
        ("city", 4, "obj"),
        ("after", 8, "case"),
        ("midnight", 4, "obl"),
    ]
    return (SENTENCE,)


@app.cell
def example_head(SENTENCE):
    def head_of(sentence, i):
        form, head, relation = sentence[i - 1]
        if head == 0:
            return "ROOT"
        return sentence[head - 1][0]

    print(head_of(SENTENCE, 3))
    print(head_of(SENTENCE, 7))
    print(head_of(SENTENCE, 4))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Word 3, *train*, has head 4, and word number 4 stands at place 3: *left*. Word 7, *after*, hangs from *midnight*, as the guidelines of UD want. Word 4, *left*, has head 0: it is the root. In the last line of `head_of`, `sentence[head - 1]` is the tuple of the head, and the `[0]` after it takes its first part, the word itself. Exercise 1 goes the other way: from a head to its dependents.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_dependents():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. dependents: who hangs from this word?

        Write `dependents(sentence, h)`: give back a list with the numbers of the words whose head is `h`, from left to right. Go over the word numbers `i`, from 1 up to and including the number of words in the sentence; word number `i` stands at place `i - 1`. With `h` equal to 0, it gives the number of the root, in a list. Replace the `...`. The check calls `dependents(SENTENCE, 4)`; then cases of its own, also on `ROOF`, the sentence *the roof of the house of the baker leaked*, and on a sentence of one word.
        """),
        L.task("Collect the numbers of the words whose head is `h`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_dependents():
    box_dependents = L.code_box("dependents", lines=7)
    box_dependents
    return (box_dependents,)


@app.cell(hide_code=True)
def check_dependents(box_dependents):
    result_dependents = L.check("dependents", box_dependents.value)
    result_dependents
    return (result_dependents,)


@app.cell(hide_code=True)
def hints_dependents():
    L.hints(
        "dependents",
        [
            "Make an empty list before the loop. In the loop, take the word with number `i` apart into its three parts, and append `i` when its head is `h`.",
            "What should `dependents(SENTENCE, 4)` give back? And does your loop get to word 8, *midnight*?",
            "`    for i in range(1, ___):`",
        ],
        why="`found = []`, then `for i in range(1, len(sentence) + 1):` with `form, head, relation = sentence[i - 1]` under it, and `if head == h:` with `found.append(i)`; after the loop `return found`. `dependents(SENTENCE, 4)` gives `[3, 6, 8]`: *train*, *city* and *midnight* hang from *left*.",
    )
    return


@app.cell(hide_code=True)
def text_after_dependents():
    mo.md(r"""
    The picture under the check is drawn from your function: the check asked it for the dependents of every head from 0 to 8, and each one became an arc. Each word gets exactly one arc coming in, and *left* gets its arc from above. A word with two arcs in, or none, would break the tree.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_path():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. path_to_root: up to the root

        Write `path_to_root(sentence, i)`: give back the numbers of the words from word `i` up to the root. `path_to_root(SENTENCE, 1)` gives `[1, 3, 4]`: word 1, its head, and the head of that one. Look up the head of word `i`: if it is 0, word `i` is the root, and the path is done; if not, append the head to `path`, and go on from there, with the head as the new `i`. A path never has more words than the sentence, so a loop `for step in range(len(sentence))` is long enough, and a `return` inside it ends the loop at the root. Replace the `...`. The check calls `path_to_root(SENTENCE, 1)`; then cases of its own, also on `ROOF`.
        """),
        L.task("Follow the heads from word `i` until a word has head 0."),
    ])
    return


@app.cell(hide_code=True)
def exercise_path():
    box_path = L.code_box("path_to_root", lines=9)
    box_path
    return (box_path,)


@app.cell(hide_code=True)
def check_path(box_path):
    result_path = L.check("path_to_root", box_path.value)
    result_path
    return (result_path,)


@app.cell(hide_code=True)
def hints_path():
    L.hints(
        "path_to_root",
        [
            "Make the list before the loop. In the loop, look up the head of word `i`: at head 0, give back the path; otherwise append the head, and make it the new `i`.",
            "What should `path_to_root(SENTENCE, 4)` give back, for the root itself?",
            "`    path = ___`",
        ],
        why="`path = [i]`, then `for step in range(len(sentence)):` with `form, head, relation = sentence[i - 1]` under it; `if head == 0:` gives back `path`; otherwise `path.append(head)` and `i = head`. After the loop, `return path`. `path_to_root(SENTENCE, 1)` gives `[1, 3, 4]`: *the*, *train*, *left*.",
    )
    return


@app.cell(hide_code=True)
def text_after_path():
    mo.md(r"""
    The picture under the check takes `ROOF`, the longer sentence *the roof of the house of the baker leaked*. *house* hangs from *roof* with `nmod`, a noun phrase that belongs to a noun: it says which roof. From the *the* before *baker*, the path climbs four steps: *baker*, *house*, *roof*, *leaked*. A program that wants to know what leaked finds the subject of *leaked*, *roof*, however many words stand in between.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: trees from two treebanks

    Pick a treebank, and draw a sentence from it, with at most the number of words you choose. Under the tree: who did what, read off the relations, the columns of CoNLL-U that the arcs come from, and the names of the labels, also those this lesson does not explain. The English and the Dutch trees use the same labels; in the Dutch ones, look for an object before its verb.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_bank = mo.ui.dropdown(options={name: key for key, name in C.TREEBANK_NAMES.items()}, value=C.TREEBANK_NAMES[L.lang], label="Treebank")
    demo_words = mo.ui.slider(5, 20, value=10, label="Words, at most", show_value=True)
    demo_new = mo.ui.button(value=0, on_click=lambda n: n + 1, label="Other sentence")
    mo.hstack([demo_bank, demo_words, demo_new], justify="start", gap=1.5, wrap=True)
    return demo_bank, demo_new, demo_words


@app.cell(hide_code=True)
def demo_look(demo_bank, demo_new, demo_words):
    mo.Html(C.demo_picture(demo_bank.value, demo_words.value, demo_new.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_relations():
    mo.md(r"""
    ### Exercise 3. Name the relations

    Take the sentence *A tired guard opened the heavy gate.* Each bold word hangs from another word of the sentence. Put each one under the relation it has with its head.
    """)
    return


@app.cell(hide_code=True)
def exercise_relations():
    sort_relations = L.sort("relations")
    sort_relations
    return (sort_relations,)


@app.cell(hide_code=True)
def check_relations(sort_relations):
    result_relations = L.check("relations", sort_relations)
    result_relations
    return (result_relations,)


@app.cell(hide_code=True)
def text_exercise_case():
    mo.md(r"""
    ### Exercise 4. Where does the preposition hang?

    Take the sentence *a ship sailed to the harbor*, with the guidelines of UD.
    """)
    return


@app.cell(hide_code=True)
def choice_case_cell():
    choice_case = L.choice(
        "case",
        question="Which word does UD make the head of *to*?",
        options={
            "right": "*harbor*",
            "verb": "*sailed*, the verb of the sentence",
            "next": "*the*, the word right after it",
            "none": "None: *to* is the root",
        },
        correct="right",
        why={
            "right": "In UD, a preposition hangs from its noun, with the label `case`: `case(harbor, to)`. *harbor* itself hangs from *sailed*, with `obl`, as *midnight* hangs from *left*.",
            "verb": "*harbor* hangs from *sailed*, with `obl`. But *to* hangs from its noun, *harbor*: UD treats a preposition as a dependent of its noun. See *Prepositions hang from their noun* in *The idea*.",
            "next": "*the* is a determiner: it hangs from *harbor*, with `det`, and nothing hangs from it. See *Prepositions hang from their noun* in *The idea*.",
            "none": "The root is *sailed*, the main verb; every other word has a head. See *A tree of words* in *The idea*.",
        },
    )
    choice_case
    return (choice_case,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Phrase structure and endless sentences](/course/open/nlp-phrase-structure?lang=en) put words into phrases inside phrases. A dependency tree links the words themselves: each word hangs from its head, with a label for the relation, and one word, the root, hangs from nothing. You found the dependents of a word, followed the heads up to the root, and named the relations of a sentence. Universal Dependencies draws such trees by the same guidelines for hundreds of treebanks, so the English and the Dutch trees of this lesson share their labels. The next lesson shows sentences that can get more than one tree.

    **An everyday example.** A river system: every stream flows into one larger stream, and all the water reaches the sea. Follow any stream down, and you come to the same mouth, as every word leads to the root.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_case, result_dependents, result_path, result_relations):
    L.wrap_up(result_dependents, result_path, result_relations, choice_case)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
