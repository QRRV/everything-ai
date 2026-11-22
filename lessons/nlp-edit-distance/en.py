# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-edit-distance. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai.lesson import lesson

    L = lesson("nlp-edit-distance", "en")
    C = L.checks   # the course's words, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    A question about [From lists to arrays: shape and dtype](/course/open/np-arrays?lang=en), and one about [Normalizing: case, accents and stop words](/course/open/nlp-normalize?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_shape():
    quick_shape = L.choice(
        "quick-shape",
        question="What is the shape of `np.array([[0, 1, 2, 3], [1, 1, 2, 3]])`?",
        options={
            "flip": "`(4, 2)`",
            "right": "`(2, 4)`",
            "flat": "`(8,)`: one long row",
        },
        correct="right",
        why={
            "right": "Two inner lists make two rows, and four numbers in each make four columns: the shape gives the rows first. This lesson keeps its table of distances in such an array, with a row for each first part of one word and a column for each first part of the other.",
            "flip": "The shape gives the rows first, then the columns: two inner lists are two rows, with four numbers each. See [From lists to arrays: shape and dtype](/course/open/np-arrays?lang=en).",
            "flat": "A list of lists becomes a grid with rows and columns, not one long row. See [From lists to arrays: shape and dtype](/course/open/np-arrays?lang=en).",
        },
    )
    quick_shape
    return (quick_shape,)


@app.cell(hide_code=True)
def quick_check_case():
    quick_case = L.choice(
        "quick-case",
        question="Which pair does `lower()` alone make into one string?",
        options={
            "accent": "*Résumé* and *resume*",
            "right": "*Paris* and *paris*",
            "plural": "*Books* and *book*",
        },
        correct="right",
        why={
            "right": "`lower()` makes *Paris* into *paris*. This lesson compares letters exactly as they are: without that step, *Paris* and *paris* differ in one letter.",
            "accent": "`lower()` makes *Résumé* into *résumé*: the accents stay, so it is still not *resume*. Taking the accents off is a step of its own; see *Accents* in [Normalizing: case, accents and stop words](/course/open/nlp-normalize?lang=en).",
            "plural": "`lower()` changes letters, not words: *books* keeps its *s*, so it is still not *book*. See [Normalizing: case, accents and stop words](/course/open/nlp-normalize?lang=en).",
        },
    )
    quick_case
    return (quick_case,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    People make typos: they type *resturant* and mean *restaurant*. A program that compares whole strings sees two different words. To suggest the word that was meant, it needs a number that says how far apart two strings are. Daniel Jurafsky and James Martin give one in the chapter *Words and Tokens* of *Speech and Language Processing*: "the minimum edit distance between two strings is defined as the minimum number of editing operations (operations like insertion, deletion, substitution) needed to transform one string into another." From *resturant* to *restaurant* is one edit: an *a* comes in. Speech recognition is measured with the same count, over words instead of letters: how many words of what a program heard must change to get what was said. This lesson computes the **edit distance** with a table in which each cell is built from the cells next to it, a way of working called **dynamic programming**.

    **An everyday example.** A search box that asks *Did you mean restaurant?* when you type *resturant* has found a word one edit away from what you typed.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_guess = L.guess(
        "guess",
        question="Line up two words of nine letters each, such as *intention* and *execution*: at each step, pair a letter of one word with a letter of the other, or pair a letter of one word with a gap. How many different ways are there to line them up?",
        options={
            "few": "About 1,000",
            "some": "About 40,000",
            "right": "About 1,500,000",
            "lots": "About 300,000,000",
        },
        reveal="**1,462,563**: about one and a half million. For *phone* and *power*, five letters each, there are already 1,683, and for two words of ten letters 8,097,453. Jurafsky and Martin: \"The space of all possible edits is enormous, so we can’t search naively.\" The table of this lesson needs only 10 × 10 = 100 cells for *intention* and *execution*, one for each pair of first parts, from no letters to all nine, and it finds the cheapest of all those ways.",
    )
    guess_guess
    return (guess_guess,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### Three kinds of edit

    An **edit** changes a string by one letter: an insertion puts a letter in, a deletion takes one out, and a substitution puts one letter in the place of another. *phone* becomes *power* in three edits: delete the *h*, substitute *w* for *n*, and insert *r* at the end. Christopher Manning, Prabhakar Raghavan and Hinrich Schütze give an example in the chapter *Dictionaries and tolerant retrieval* of *Introduction to Information Retrieval*: "the edit distance between cat and dog is three." Each letter of *cat* is replaced by one of *dog*.

    When each edit costs 1, the edit distance is also called the **Levenshtein distance**, after Vladimir Levenshtein, who defined it in 1965 in a paper about binary codes: strings of 0s and 1s sent over a line that can lose a symbol, add one, or turn a 0 into a 1 or a 1 into a 0. Not every book counts the same way. Jurafsky and Martin first let every edit cost 1: "The Levenshtein distance between intention and execution is 5." They go on: Levenshtein also proposed another version, in which "each insertion or deletion has a cost of 1 and substitutions are not allowed." A substitution then takes a deletion and an insertion, so it costs 2: "Using this version, the Levenshtein distance between intention and execution is 8." This lesson gives every edit the cost 1; the demo lets you try 2.

    ### Lining up two words

    An **alignment** writes one word above the other, so that you can see which letter goes with which. Jurafsky and Martin: "Given two sequences, an alignment is a correspondence between substrings of the two sequences." The lesson writes a gap as `-`:

    ```
    p h o n e -
    p - o w e r
    ```

    Where both lines have the same letter, nothing changes. A `-` in the bottom line is a deletion: the *h* goes. A `-` in the top line is an insertion: the *r* comes in. Two different letters are a substitution: *w* for *n*. Every alignment of two words has its number of edits, and the edit distance is the smallest of those numbers. Deleting all five letters of *phone* and typing *power* is an alignment too, with ten edits.
    """)
    return


@app.cell(hide_code=True)
def idea_picture():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_table():
    mo.md(r"""
    ### A table of smaller problems

    Trying every alignment takes far too long, as the guess showed. The way out is to solve smaller problems first and keep their answers. Write $D[i, j]$ for the edit distance between the first $i$ letters of one word and the first $j$ letters of the other: for *phone* and *power*, $D[2, 3]$ is the distance from *ph* to *pow*, and $D[5, 5]$ is the answer. The smallest problems are easy. Jurafsky and Martin: "In the base case, with a source substring of length i but an empty target string, going from i characters to 0 requires i deletes." And from no letters to $j$ letters takes $j$ insertions.

    Every other cell follows from three cells next to it. Look at the last place of the cheapest alignment of the first $i$ letters with the first $j$ letters. It holds one of three things:

    - the $i$-th letter of the first word above a gap: a deletion, so the cell above plus 1, $D[i-1, j] + 1$;
    - a gap above the $j$-th letter of the second word: an insertion, so the cell to the left plus 1, $D[i, j-1] + 1$;
    - the two letters one above the other: the cell up and to the left, $D[i-1, j-1]$, plus 0 when the letters are the same and plus 1 when they differ.

    The cell takes the smallest of the three. Fill the table row by row, from the top left, and every cell you fill needs only cells that are already filled. The cell at the bottom right holds the edit distance.

    Jurafsky and Martin call this way of working dynamic programming: "Dynamic programming is the name for a class of algorithms, first introduced by Bellman (1957), that apply a table-driven method to solve problems by combining solutions to subproblems." Richard Bellman later wrote, in the autobiography *Eye of the Hurricane* (1984), that the name had to get past a Secretary of Defense who "had a pathological fear and hatred of the word, research." The word *dynamic*, Bellman wrote, cannot be used in a bad sense, and: "Thus, I thought dynamic programming was a good name. It was something not even a Congressman could object to." Stuart Dreyfus quotes these lines in *Richard Bellman on the birth of dynamic programming* (2002).

    ### Following the path back

    The table gives the distance, not the edits. To find them, start in the last cell and go back the way its smallest value came: to the cell above, to the left, or up and to the left. Jurafsky and Martin call this a **backtrace**: "In a backtrace, we start from the last cell (at the final row and column), and follow the pointers back through the dynamic programming matrix." A pointer says which cell a value came from. A step up is a deletion, a step to the left an insertion, and a step up and to the left keeps a letter or substitutes it. "Each complete path between the final cell and the initial cell is a minimum distance alignment." Two words can have more than one cheapest alignment: *phone* and *power* have one, *intention* and *execution* seven, with every edit at cost 1.
    """)
    return


@app.cell(hide_code=True)
def table_picture():
    mo.Html(C.table_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_more():
    mo.md(r"""
    ### One idea, found many times

    Jurafsky and Martin write that the algorithm "was named by Wagner and Fischer (1974) but independently discovered by many people". Manning and colleagues write that their edit distance "is due to Levenshtein (1965) and the algorithm in Figure 3.5 is due to Wagner and Fischer (1974)". Robert Wagner and Michael Fischer wrote in the summary of their paper *The String-to-String Correction Problem*: "An algorithm is presented which solves this problem in time proportional to the product of the lengths of the two strings." That product is about the size of the table. In a survey of 2001, Gonzalo Navarro lists others who found the same method, among them Vintsyuk in 1968, for speech, and Needleman and Wunsch in 1970, who compared the amino acid sequences of proteins.

    ### Weights and swaps

    Not every edit is equally likely. Manning and colleagues: "a higher weight may be placed on replacing the character s by the character p, than on replacing it by the character a (the latter being closer to s on the keyboard)." The table works the same with other costs: only what each step adds changes.

    People also swap two letters, as in *acress* for *caress*. With insertions, deletions and substitutions, a swap costs two edits. Fred Damerau's method, in *A technique for computer detection and correction of spelling errors* (1964), assumes "that a word which cannot be found in a dictionary has at most one error, which might be a wrong, missing or extra letter or a single transposition." A **transposition** swaps two neighboring letters, and an edit distance that counts it as one edit is called the Damerau-Levenshtein distance. Jurafsky and Martin write in the appendix *Spelling Correction and the Noisy Channel* that "the majority of spelling errors consist of a single-letter change". Their example *acress* is one edit away from *actress*, *across* and *acres*, and with a transposition one edit from *caress* too.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **edit** | a change of one letter: an insertion, a deletion or a substitution |
        | **edit distance** | the smallest number of edits that turn one string into another: *phone* and *power* are 3 apart |
        | **Levenshtein distance** | the edit distance when every edit costs 1; some books give a substitution the cost 2 |
        | **alignment** | two strings written one above the other, with `-` for a gap, so that you can see which letter goes with which |
        | **dynamic programming** | solving a big problem by solving smaller ones first and keeping their answers in a table |
        | **backtrace** | going back through the table from the last cell to the first, to find the edits of a cheapest alignment |
        | **transposition** | swapping two neighboring letters, as in *thier* for *their*; the Damerau-Levenshtein distance counts it as one edit |
        | **word error rate** | the edits between what was said and what a program heard, counted in words, as a share of the words said |
        | `np.zeros((3, 4), dtype=int)` | an array of 3 rows and 4 columns of whole numbers, all 0 |
        | `D[i, j] = v` | puts the value `v` in row `i` and column `j` of the array `D` |
        | `min(a, b, c)` | the smallest of the values between the parentheses: `min(3, 1, 2)` is `1` |
        """),
        L.watch_out("In other lessons of the course, *alignment* means something else: getting an AI system to pursue the goals that people intend. Here an alignment is two strings written one above the other. And when you give an edit distance, say what a substitution costs: Jurafsky and Martin's table uses 2, this lesson 1."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. Make a table with a row for each first part of the first word, from no letters to all of them, and a column for each first part of the second word.
    2. Fill the first column with 0, 1, 2, ... (deletions) and the first row with 0, 1, 2, ... (insertions).
    3. Fill every other cell, row by row, with the smallest of three: the cell above plus 1, the cell to the left plus 1, and the cell up and to the left plus 0 for the same letter or 1 for a different one.
    4. Read the edit distance in the last cell, and go back along the path to see the edits.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The code below computes one cell of the table of *phone* and *power*: row 4 and column 3, the distance from *phon* to *pow*. The three cells around it are read from the table above.
    """)
    return


@app.cell
def example_cell():
    a = "phone"
    b = "power"
    print(a[:4], b[:3])
    print(a[3], b[2], a[3] == b[2])
    up = 2         # row 3, column 3: from pho to pow
    left = 2       # row 4, column 2: from phon to po
    diagonal = 1   # row 3, column 2: from pho to po
    print(min(up + 1, left + 1, diagonal + 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The fourth letter of *phone* is `a[3]`, since Python counts from 0, and the third letter of *power* is `b[2]`. *n* and *w* differ, so the step up and to the left adds 1 too. The cheapest of the three ways is the substitution: 1 + 1 = 2, the number in the table. Exercise 3 does this for every cell; Exercises 1 and 2 come first.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_count():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. count_edits: the edits of an alignment

        An alignment comes as two strings of the same length, `top` and `bottom`, with `"-"` for a gap. Write `count_edits(top, bottom)`: go over the places, and count an insertion where `top` has a `"-"`, a deletion where `bottom` has one, and a substitution where two different letters stand one above the other. Give back a dictionary with the three counts under the keys `"insert"`, `"delete"` and `"substitute"`. For `count_edits("phone-", "p-ower")`, the result is `{"insert": 1, "delete": 1, "substitute": 1}`. Replace the `...`. The check calls that, then cases of its own: the alignment of *intention* and *execution* that Jurafsky and Martin show, one without edits, one with only an insertion, two letters swapped, a gap on each side, two empty strings, *kitten* and *sitting*, and an alignment of words, given as lists.
        """),
        L.task("Count the insertions, deletions and substitutions in an alignment with `-` for a gap."),
    ])
    return


@app.cell(hide_code=True)
def exercise_count():
    box_count = L.code_box("count-edits", lines=10)
    box_count
    return (box_count,)


@app.cell(hide_code=True)
def check_count(box_count):
    result_count = L.check("count-edits", box_count.value)
    result_count
    return (result_count,)


@app.cell(hide_code=True)
def hints_count():
    L.hints(
        "count-edits",
        [
            r'Start with `counts = {"insert": 0, "delete": 0, "substitute": 0}` and go over the places with `for i in range(len(top))`. At each place, ask with `if`, `elif` and `elif` which kind of edit it is, add one to that count, as in `counts["insert"] = counts["insert"] + 1`, and give back `counts` after the loop.',
            r'Ask about the gaps before you ask whether the letters differ: a gap and a letter differ too. At place `i`, how do you get what the top line holds there?',
            r'`        if ___ == "-":`',
        ],
        why=r'`counts` starts at 0 for each key; in the loop, `if top[i] == "-":` counts an insertion, `elif bottom[i] == "-":` a deletion, and `elif top[i] != bottom[i]:` a substitution; after the loop, `return counts`.',
    )
    return


@app.cell(hide_code=True)
def text_after_count():
    mo.md(r"""
    Your count_edits works on lists of words just as well: the last alignment in the picture holds what someone said, above, and what a program heard, below. Counted in words, the edits give the word error rate of speech recognition. Jurafsky and Martin: "The standard evaluation metric for speech recognition systems is the word error rate." The rate is all insertions, substitutions and deletions, as a share of the words that were said: 4 edits on 7 words is 57.1%. Their own example, from a recorded phone conversation, has six substitutions, three insertions and one deletion on 13 words: 76.9%. Since the insertions count too, they note, "the error rate can be greater than 100%".
    """)
    return


@app.cell(hide_code=True)
def text_exercise_start():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. start_table: the first row and column

        Write `start_table(a, b)`: make a table `D` of whole numbers with `np.zeros`, with a row for each first part of `a`, from no letters to all of them, and a column for each first part of `b`. Fill the first column with 0, 1, 2, ..., since `D[i, 0]` is the number of deletions that turn the first `i` letters of `a` into nothing, and fill the first row the same way, with the insertions. Give back `D`. For `start_table("phone", "power")`, the result has the shape `(6, 6)`, with 0 to 5 down the first column and along the first row, and zeros everywhere else. Replace the `...`. The check calls that, then cases of its own: words of different lengths, both ways round, an empty word on each side, two empty words, and lists of words.
        """),
        L.task("Make a table of zeros, and fill its first column and first row with the distances from and to the empty start."),
    ])
    return


@app.cell(hide_code=True)
def exercise_start():
    box_start = L.code_box("start-table", lines=7)
    box_start
    return (box_start,)


@app.cell(hide_code=True)
def check_start(box_start):
    result_start = L.check("start-table", box_start.value)
    result_start
    return (result_start,)


@app.cell(hide_code=True)
def hints_start():
    L.hints(
        "start-table",
        [
            "Make the table with `D = np.zeros((rows, columns), dtype=int)`, with the right numbers of rows and columns. Then one `for` over the rows sets `D[i, 0] = i`, and one over the columns sets `D[0, j] = j`. Give back `D`.",
            "*phone* has five letters. How many first parts does it have, when no letters at all counts as one of them?",
            r'`    D = np.zeros((___, len(b) + 1), dtype=int)`',
        ],
        why=r'`D = np.zeros((len(a) + 1, len(b) + 1), dtype=int)`, then `for i in range(len(a) + 1):` with `D[i, 0] = i`, `for j in range(len(b) + 1):` with `D[0, j] = j`, and `return D`.',
    )
    return


@app.cell(hide_code=True)
def text_after_start():
    mo.md(r"""
    The start alone holds part of the answer already: from *phone* to no letters takes five deletions, the bottom of the first column, and from no letters to *power* five insertions, the end of the first row. $D[0, 0]$ is 0, since no letters need no edits. The grey cells are for Exercise 3.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_distance():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. edit_distance: fill the table

        Write `edit_distance(a, b)`: start with `D = start_table(a, b)`, the course's version, which the check gives your function. Then fill every other cell, row by row: `i` goes from 1 up to and including the length of `a`, and for each `i`, `j` goes from 1 up to and including the length of `b`. Each cell takes the smallest of the three ways into it, as in the recipe; the letters of row `i` and column `j` are the `i`-th letter of `a` and the `j`-th letter of `b`. Give back the last cell, `D[len(a), len(b)]`. For `edit_distance("phone", "power")`, the result is `3`. Replace the `...`. The check calls that, then cases of its own: *intention* and *execution*, *cats* and *fast*, a missing letter, two letters swapped, the same word, an empty word on each side, two empty words, two different letters, and lists of words, for which the same code works.
        """),
        L.task("Fill every cell of the table with the cheapest of the three ways into it, and give back the last cell."),
    ])
    return


@app.cell
def start_cell():
    start_table = C.start_table
    print(start_table("pho", "po"))
    return


@app.cell(hide_code=True)
def exercise_distance():
    box_distance = L.code_box("edit-distance", lines=9)
    box_distance
    return (box_distance,)


@app.cell(hide_code=True)
def check_distance(box_distance):
    result_distance = L.check("edit-distance", box_distance.value)
    result_distance
    return (result_distance,)


@app.cell(hide_code=True)
def hints_distance():
    L.hints(
        "edit-distance",
        [
            "Two `for` loops, one inside the other: `for i in range(1, len(a) + 1)` and `for j in range(1, len(b) + 1)`. In the inner loop, set `cost` to 0 when the letters `a[i - 1]` and `b[j - 1]` are the same and to 1 when they differ, then fill `D[i, j]` with `min` of the three ways. After both loops, give back `D[len(a), len(b)]`.",
            "Two of the three ways add 1. Which cell does the third way come from, and what does it add?",
            r'`            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, ___)`',
        ],
        why=r'`D = start_table(a, b)`; `for i in range(1, len(a) + 1):` and inside it `for j in range(1, len(b) + 1):`; `cost = 1`, and `cost = 0` when `a[i - 1] == b[j - 1]`; `D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + cost)`; after both loops, `return D[len(a), len(b)]`.',
    )
    return


@app.cell(hide_code=True)
def text_after_distance():
    mo.md(r"""
    Your edit_distance on every first part of *phone* and *power* makes the same table as the one above, and on lists of words it gives the 4 edits of the word error rate. The edit distance compares letters, not meanings: *car* and *cat* are one edit apart, *car* and *automobile* ten. And it takes time: the table has a cell for each pair of first parts, so it grows with the product of the two lengths, as Wagner and Fischer wrote. Comparing one word with every word of a long word list adds up. Manning and colleagues call it "inordinately expensive" to compute the edit distance from a search query to every term a search engine knows, and give tricks to try fewer words, such as only the terms that start with the same letter as the query: "the hope is that spelling errors do not occur in the first character of the query."
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: the table of your own words

    Type two words and choose what a substitution costs. The picture fills the table, colors the path back from the last cell, and shows the alignment that the path gives. Try *intention* and *execution* with both costs, a typo of your own, or two words that differ only in a capital.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_a = mo.ui.text(value="intention", label="First word")
    demo_b = mo.ui.text(value="execution", label="Second word")
    demo_cost = mo.ui.dropdown(options={"1": "1", "2": "2"}, value="1", label="A substitution costs")
    mo.vstack([demo_a, demo_b, demo_cost])
    return demo_a, demo_b, demo_cost


@app.cell(hide_code=True)
def demo_look(demo_a, demo_b, demo_cost):
    mo.Html(C.demo_picture(demo_a.value, demo_b.value, demo_cost.value, L.lang))
    return


@app.cell(hide_code=True)
def text_after_demo():
    mo.md(r"""
    When a substitution costs 1, *intention* and *execution* are 5 apart, and seven alignments reach that distance. When it costs 2, the distance is 8, as in the table of Jurafsky and Martin, and 134 alignments reach it: a substitution then costs as much as a deletion plus an insertion, so the table often has several cheapest ways into a cell. The path in the picture takes one of them.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Exercise 4. How many edits?

    Each bold pair is two strings. Work out their edit distance in your head, with every edit at cost 1, and put the pair under it.
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_edits = L.sort("how-many-edits")
    sort_edits
    return (sort_edits,)


@app.cell(hide_code=True)
def check_sort(sort_edits):
    result_sort = L.check("how-many-edits", sort_edits)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Normalizing: case, accents and stop words](/course/open/nlp-normalize?lang=en) made *Café* and *cafe* one string; this lesson measured how far apart two strings are that are not the same. You counted the edits of an alignment, made the start of the table in a NumPy array, as in [From lists to arrays: shape and dtype](/course/open/np-arrays?lang=en), and filled the rest with dynamic programming: each cell is the cheapest of three ways in, built from cells that are already filled. The same table works for the words of a sentence, which gives the word error rate of speech recognition. The next lesson of the domain, *A spelling corrector*, makes every string within one or two edits of a typo, keeps the real words among them, and picks the one that is most common in a collection of books.

    **An everyday example.** A phone keyboard that changes *teh* into *the* has picked a word close to what you typed: two edits away, or one when a swap of two letters counts as one edit.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_count, result_distance, result_sort, result_start):
    L.wrap_up(result_count, result_start, result_distance, result_sort)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
