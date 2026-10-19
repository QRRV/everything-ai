# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-quality-filter. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-quality-filter", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Where pretraining data comes from](/course/open/llm-pretraining-data?lang=en) and [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_web():
    quick_web = L.choice(
        "quick-web",
        question="FineWeb's license, ODC-By 1.0, lets anyone use and share the collection, if they name its makers. What does it say about the web pages in it?",
        options={
            "right": "Nothing, as each page keeps its own rights",
            "free": "That anyone may copy and share each page in it",
            "public": "That the pages are in the public domain",
        },
        correct="right",
        why={
            "free": "The license covers the collection as a whole, not the texts in it. See [Where pretraining data comes from](/course/open/llm-pretraining-data?lang=en).",
            "public": "That anyone can read a page on the web does not put it in the public domain: it keeps the rights of its maker. See [Where pretraining data comes from](/course/open/llm-pretraining-data?lang=en).",
            "right": "So the course hosts none of the raw web pages of this lesson: you download them yourself.",
        },
    )
    quick_web
    return (quick_web,)


@app.cell(hide_code=True)
def quick_check_probability():
    quick_probability = L.choice(
        "quick-probability",
        question="A logistic regression gives a page the number 0.8. What does that number say?",
        options={
            "share": "That 80% of the page is good",
            "class": "That the page is in class 1 for sure",
            "right": "The probability, by the model, that the page is in class 1",
        },
        correct="right",
        why={
            "share": "It says nothing about parts of the page: it is one probability for the whole of it. See [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en).",
            "class": "The sigmoid never reaches 1: 0.8 leaves a probability of 0.2 for class 0. See [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en).",
            "right": "The sigmoid of `w · x + b`. In this lesson, class 1 is text that reads like a good example.",
        },
    )
    quick_probability
    return (quick_probability,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    A web page as Common Crawl saves it is not only its text. It is also the menu, the buttons, the footer, the notice that you should turn on JavaScript, a list of tags, and sometimes a page of nothing but keywords to fool search engines. A model learns from all of it, token by token, so every token of a menu is a token it does not spend on a sentence. Before pretraining, every open corpus throws much of a crawl away, with rules that anyone can read and, more and more, with a small classifier.

    **An everyday example.** Sorting your post: the flyers go straight into the paper bin, unread, and you open the letters. A few rules sort most of it; for the rest, you look at an envelope and judge.
    """)
    return


@app.cell(hide_code=True)
def predict_lines():
    _m = C.MEASURED
    guess_lines = L.guess(
        "lines",
        question=f"The first 20 MB of a shard of C4 before its cleaning hold {C.number(_m['pages'], 0, L.lang)} raw web pages, {C.number(_m['chars'] / 1e6, 1, L.lang)} million characters. C4 keeps only the lines that end a sentence and have at least 5 words. How much of the text do you think is left?",
        options={
            "most": "About 90%",
            "two": "About two thirds",
            "right": "About a third",
        },
        reveal=f"**About a third: {C.number(_m['chars_lines'] / 1e6, 1, L.lang)} million characters, {C.number(100 * _m['chars_lines'] / _m['chars'], 0, L.lang)}%.** The other two thirds are lines without the end of a sentence, or with fewer than 5 words: menus, buttons, titles, lists, dates and links.",
    )
    guess_lines
    return (guess_lines,)


@app.cell(hide_code=True)
def text_idea():
    _m = C.MEASURED
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Rules per line.** C4, a corpus that Google made from Common Crawl in 2019 to train T5, keeps a line only when it ends a sentence, with `.`, `!`, `?` or `"`, has at least 5 words, and does not mention JavaScript. Its makers also threw out every page with fewer than 3 sentences, with "lorem ipsum" or a curly bracket `{` in it, and every page with a word from a list of bad words.

        **Rules per page.** Gopher, a model DeepMind made in 2021, threw out, among other rules, a page with fewer than 50 or more than 100,000 words, with an average word length below 3 or above 10, with fewer than two of the stop words "the", "be", "to", "of", "and", "that", "have" and "with", or with fewer than 80% of words that have a letter. Such rules are cheap and anyone can read them, but they are blunt: a good table of numbers fails them, and a page of spam in good sentences passes.

        **A classifier.** So corpora also train a classifier: a few numbers per page, its features, go into a model that gives the probability that the page reads like good text. To learn that, it needs examples of good text. The classifier of GPT-3 was a logistic regression on counts of words, with WebText, books and Wikipedia as its good examples and raw web pages as the others. FineWeb-Edu let a large language model grade some 450,000 pages for how much they teach, trained a classifier on those grades, and kept the pages it gave at least 3 of 5: a threshold. The classifier learns what its good examples look like, and nothing else: that is its strength and its weakness.

        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **quality filter** | a rule or a small classifier that removes low-quality text, such as spam, boilerplate or broken pages, before training | `keep_page(text)` |
        | **stop word** | a very common word that carries little meaning on its own, such as "the", "of" or "and"; ordinary sentences are full of them, lists and spam are not | `{"the", "of", "and"}` |
        | **ends with** | whether a text ends with a piece of text, or with one of a tuple of them | `line.endswith((".", "!"))` |
        | **is a letter** | whether a character is a letter, of any alphabet; `isspace` asks the same of spaces, tabs and new lines | `c.isalpha()` |
        """),
        L.watch_out("A filter removes text, so it decides what a model never sees. C4's list of bad words also threw out many pages by and about some groups of people more than others, such as pages in African American English and pages that mention gay or lesbian people (Dodge and others, 2021). Look at what a filter throws out, not only at what it keeps."),
        mo.md(r"""
        The picture: the characters of the 8,690 raw pages of this lesson's download, after each step, with the course's own versions of the exercises.
        """),
        mo.Html(C.steps_picture([("raw web pages", round(_m["chars"] / 1e6, 1)), ("after the line rules of C4", round(_m["chars_lines"] / 1e6, 1)),
                    ("after the page rules of Gopher", round(_m["chars_kept"] / 1e6, 1))],
                                {"title": "Millions of characters, step by step", "lang": L.lang,
                                 "note": f"{C.number(_m['kept'], 0, L.lang)} of the {C.number(_m['pages'], 0, L.lang)} pages pass the page rules",
                                 "alt": f"Bars of millions of characters: the raw web pages {C.number(_m['chars'] / 1e6, 1, L.lang)}; after the line rules of C4 {C.number(_m['chars_lines'] / 1e6, 1, L.lang)}; after the page rules of Gopher {C.number(_m['chars_kept'] / 1e6, 1, L.lang)}. {C.number(_m['kept'], 0, L.lang)} of the {C.number(_m['pages'], 0, L.lang)} pages pass the page rules."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To filter raw web pages for pretraining:

    1. Clean every page line by line: keep the lines that read as sentences.
    2. Throw out the pages that break a page rule: too short, too long, odd words, no stop words, too few letters.
    3. Describe every page that is left with a few numbers, its features.
    4. Train a classifier on the features of good examples and of ordinary pages, and score every page.
    5. Keep the pages above a threshold, and read a sample of what you threw out.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 1 for three lines, one rule at a time: does the line end a sentence, and how many words does it have?
    """)
    return


@app.cell
def example_lines():
    _end = (".", "!", "?", '"')
    _lines = ["  Read more  ", "Our shop is open every day from nine to five.", "Free shipping!"]
    for _line in _lines:
        # 1. the line without spaces at its ends, then the two rules
        _line = _line.strip()
        print(f'"{_line}"', "ends a sentence:", _line.endswith(_end), "words:", len(_line.split()))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    "Read more" is a button: no end of a sentence, 2 words. "Free shipping!" ends with `!`, but with 2 words it is a slogan, not a sentence. Only the opening hours pass both rules. `strip()` matters: without it, the spaces after "Read more" would be part of the line.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_lines():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. The line rules of C4

        Complete `clean_lines(page)`: step 1 of the recipe. It gives back the lines of `page` that end with one of `END`, have at least 5 words and do not mention javascript, in capitals or not, each without spaces at its ends, joined by new lines. The check calls it with a page of six lines, from "Home" to "We walked along the beach until the sun went down.", and with others.
        """),
        L.task("Make `clean_lines` keep only the lines that pass all three rules."),
    ])
    return


@app.cell(hide_code=True)
def exercise_lines():
    box_lines = L.code_box("clean-lines", lines=10)
    box_lines
    return (box_lines,)


@app.cell(hide_code=True)
def check_lines(box_lines):
    result_lines = L.check("clean-lines", box_lines.value)
    result_lines
    return (result_lines,)


@app.cell(hide_code=True)
def hints_lines():
    L.hints(
        "clean-lines",
        [
            "The worked example shows the first two rules for one line. Add the third, and keep the line only when all three hold.",
            "`line.lower()` gives the line in small letters, so one test with `in` finds JavaScript in any spelling.",
        ],
        why="A line that ends a sentence, with at least 5 words, is almost always part of the text a person wrote; the menus, buttons and notices of a page are not. Joining the kept lines with new lines keeps the page as a page.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_page():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. The page rules of Gopher

        Complete `keep_page(text)`: step 2 of the recipe, for a page that `clean_lines` already cleaned. It gives `False` when the page breaks one of the four rules of Gopher in the idea, and else `True`. The check calls it with a page of 60 words, "the cat sat on a warm mat and looked at the birds with care" over and over, and with others, such as a page of 49 words.
        """),
        L.task("Make `keep_page` throw out a page that breaks one of the four rules."),
    ])
    return


@app.cell(hide_code=True)
def exercise_page():
    box_page = L.code_box("keep-page", lines=18)
    box_page
    return (box_page,)


@app.cell(hide_code=True)
def check_page(box_page):
    result_page = L.check("keep-page", box_page.value)
    result_page
    return (result_page,)


@app.cell(hide_code=True)
def hints_page():
    L.hints(
        "keep-page",
        [
            "Each rule is an `if` that ends with `return False`; a page that gets past all four ends at `return True`. The average word length is the total of `len(w)` divided by the number of words.",
            "For rule 3, collect the stop words of the page in a set: then \"the\" three times counts once. For rule 4, count the words with at least one character for which `c.isalpha()` is true.",
        ],
        why="Each rule catches another kind of page that is not ordinary text: too little of it, words that are not words, no sentences, or mostly numbers and symbols. The order does not matter for the outcome; cheap rules first only save time.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_features():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. Four numbers per page

        Complete `features(page)`: step 3 of the recipe. It gives back four numbers between 0 and 1: the share of the lines that end a sentence, the share of the words that are one of `WORDS` in lower case, the share of the characters that are letters or spaces, and the number of words per line divided by 20, at most 1. Lines that are empty after `strip()` do not count. The check calls it with a page of five lines, from "Home" to an e-mail address, and with others.
        """),
        L.task("Make `features` give the four numbers of a page."),
    ])
    return


@app.cell(hide_code=True)
def exercise_features():
    box_features = L.code_box("features", lines=26)
    box_features
    return (box_features,)


@app.cell(hide_code=True)
def check_features(box_features):
    result_features = L.check("features", box_features.value)
    result_features
    return (result_features,)


@app.cell(hide_code=True)
def hints_features():
    L.hints(
        "features",
        [
            "Each number is a count divided by a total: lines, words or characters. Count with a loop and an `if`, as in exercise 2.",
            "Feature 4 is `len(words) / len(lines) / 20`, and `min(..., 1.0)` keeps it at most 1.",
        ],
        why="Shares instead of counts let a short and a long page be compared, and four numbers between 0 and 1 suit a logistic regression without scaling.",
    )
    return


@app.cell(hide_code=True)
def text_demo(result_lines, result_page, result_features):
    result_lines, result_page, result_features  # the note changes when an exercise is done
    mo.vstack([
        mo.md(f"""
        ### What your functions do with raw web pages

        The button below downloads the first {round(C.hub.PINNED[C.C4_SAMPLE]['size'] / 1e6)} MB of a shard of C4 before its cleaning, once, from Hugging Face: {C.number(C.MEASURED['pages'], 0, L.lang)} raw web pages. The course hosts none of them: their texts keep their own rights, so you download them for yourself, and the course shares no model trained on them. Then your three functions run on every page, and your `fit` of [Training logistic regression](/course/open/ml-train-logistic?lang=en) trains the classifier: {C.number(C.N_TRAIN, 0, L.lang)} short stories of TinyStories as its good examples, class 1, and {C.number(C.N_TRAIN, 0, L.lang)} of the pages that pass your rules as the others, class 0. It takes a few seconds.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def download_button_cell():
    download_button = mo.ui.run_button(label="Download the web pages")
    download_button
    return (download_button,)


@app.cell(hide_code=True)
def demo_pipeline(download_button, box_lines, box_page, box_features, result_lines, result_page, result_features):
    _ready = result_lines.state == "correct" and result_page.state == "correct" and result_features.state == "correct"
    mo.stop(not _ready, mo.md("*Finish exercises 1 to 3 first: the demo runs your three functions.*"))
    C.sample_or_stop(download_button, {
        "ask": "*Click the button to download the web pages, 20 MB.*",
        "busy": "**Downloading** · {done} of {total} MB",
        "no_net": "*There is no connection to Hugging Face. Try again when you are online.*",
        "not_pinned": "*The file from Hugging Face is not the one the course pinned, so it was thrown away. Report it on GitHub, with the link at the bottom of the lesson.*",
        "no_room": "*There is no room left on the disk for the download. Make room, and click again.*",
        "failed": "*The download did not work: {why}.*",
    })
    _fns = {}
    for _box, _name in ((box_lines, "clean_lines"), (box_page, "keep_page"), (box_features, "features")):
        _fns[_name] = run_code(_box.value).names[_name]
    _r = C.pipeline(_fns["clean_lines"], _fns["keep_page"], _fns["features"])
    if "error" in _r:
        _out = mo.md("*Your `fit` in `work/projects/nn/logistic.py` stops with an error. Check its exercise again, or delete the file to use the course's version.*")
    else:
        _above = sum(1 for _s in _r["scores"] if _s > 0.5)
        _rows = []
        for _page, _what in ((3715, "a page of keywords for T-shirts"), (22, "a page that offers films to download"),
                             (368, "a help page about exporting search results"), (7149, "a question and its answer on a forum")):
            _s = C.score_of(_r, _page)
            _rows.append(f"| {_what} | {'thrown out by your rules' if _s is None else ('below 0.001' if _s < 0.001 else C.number(_s, 3, L.lang))} |")
        _who = "" if not _r["stand_ins"] else " The course's version stood in for " + ", ".join(f"`{_m}`" for _m in _r["stand_ins"]) + "."
        _out = mo.vstack([
            mo.md(f"""Of {C.number(_r['pages'], 0, L.lang)} pages, {C.number(_r['chars'] / 1e6, 1, L.lang)} million characters, your line rules keep {C.number(_r['chars_lines'] / 1e6, 1, L.lang)} million, {C.number(100 * _r['chars_lines'] / _r['chars'], 0, L.lang)}%. Your page rules keep {C.number(_r['kept'], 0, L.lang)} pages, with {C.number(_r['chars_kept'] / 1e6, 1, L.lang)} million characters. Your classifier gives {C.number(_above, 0, L.lang)} of those pages a score above 0.5. Your first number is 1 on nearly every page here: your line rules kept only lines that end a sentence. A feature that a filter already settled tells the classifier nothing, so it leans on the other three.{_who}"""),
            mo.Html(C.histogram_picture(_r["scores"], {"title": "The scores of the pages your rules keep", "x": "score of your classifier", "lang": L.lang,
                                                       "alt": f"A histogram of the scores of the {C.number(_r['kept'], 0, L.lang)} pages, from 0 to 1, with a line at 0.5; {C.number(_above, 0, L.lang)} pages score above it."})),
            mo.md("Four of the pages, and the score your classifier gives each:\n\n| page | score |\n|---|---|\n" + "\n".join(_rows) +
                  "\n\nThe keywords and the film page pass the rules, as their lines are long enough and end in a mark, but the classifier sees that they do not read like sentences. The help page and the forum answer do."),
        ])
    _out
    return


@app.cell(hide_code=True)
def text_exercise_bias():
    mo.md(r"""
    ### Exercise 4. What does the classifier learn?
    """)
    return


@app.cell(hide_code=True)
def choice_bias_cell():
    choice_bias = L.choice(
        "bias",
        question="The classifier of the demo learned from short stories for children what good text looks like. Which good page is it most likely to score low?",
        options={
            "spam": "A blog post about a day at the beach: long, plain sentences",
            "right": "A clear recipe: short lines with numbers",
            "none": "None: a classifier scores every good page well",
        },
        correct="right",
        why={
            "spam": "That one reads much like its examples: long sentences, full of stop words and letters. It gets a high score.",
            "none": "A classifier knows only what its good examples look like. A good page in another style looks to it like the pages it learned to reject.",
            "right": "Short lines, few stop words, many numbers: unlike a story, so a low score, however good the recipe. Every classifier prefers what its good examples look like: FineWeb-Edu's prefers pages that read like school texts.",
        },
    )
    choice_bias
    return (choice_bias,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The web pages of [Where pretraining data comes from](/course/open/llm-pretraining-data?lang=en), now cleaned; the logistic regression of [Logistic regression and the decision boundary](/course/open/ml-logistic-regression?lang=en), trained with your `fit` of [Training logistic regression](/course/open/ml-train-logistic?lang=en), now as a quality filter. The next lesson finds the pages that are almost the same as another, which no filter here catches.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_lines, result_page, result_features, choice_bias):
    L.wrap_up(result_lines, result_page, result_features, choice_bias)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
