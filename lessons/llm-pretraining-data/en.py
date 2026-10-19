# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson llm-pretraining-data. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("llm-pretraining-data", "en")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two short questions about [Where does this data come from?](/course/open/data-provenance?lang=en) and [Project: your GPT on your own tokens](/course/open/gpt-train-bpe?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_license():
    quick_license = L.choice(
        "quick-license",
        question="A website offers a large collection of texts, free to download, without any license. What do you assume?",
        options={
            "free": "That you may do anything with it: it is free, and nobody claims it",
            "name": "That you may use it, as long as you name the website",
            "right": "That you may not copy or share it: ask the maker",
        },
        correct="right",
        why={
            "free": "Free to download is not free to use. Without a license, the maker keeps their rights: see [Where does this data come from?](/course/open/data-provenance?lang=en).",
            "name": "Naming the maker is what CC BY asks, and only a license such as CC BY gives you that right in return. Without a license there is no such right: see [Where does this data come from?](/course/open/data-provenance?lang=en).",
            "right": "This lesson asks the same about the trillions of tokens a large language model learns from.",
        },
    )
    quick_license
    return (quick_license,)


@app.cell(hide_code=True)
def quick_check_stories():
    quick_stories = L.choice(
        "quick-stories",
        question="Your GPT on tokens trained only on short stories for children. You start it with \"The capital of France is\". What do you expect it to write?",
        options={
            "right": "Words from a children's story: it never read such facts",
            "paris": "Paris: the tokenizer knows the word, so the model can write it",
            "nothing": "Nothing: it only writes after \"Once upon a time\"",
        },
        correct="right",
        why={
            "paris": "The tokenizer cuts text into pieces; it knows nothing of what they mean. Your GPT learned only what its stories hold: see [Project: your GPT on your own tokens](/course/open/gpt-train-bpe?lang=en).",
            "nothing": "It continues any start you give it, with what it learned: the words and turns of children's stories. See [Project: your GPT on your own tokens](/course/open/gpt-train-bpe?lang=en).",
            "right": "A model knows what it read. This lesson looks at what large models read.",
        },
    )
    quick_stories
    return (quick_stories,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Your GPT on tokens writes children's stories, because that is all it read: 5 million characters of TinyStories. A large language model also knows what it read, only it read trillions of tokens: web pages, code, papers, books. What goes into that pile decides what the model knows, which languages it speaks and whether it can write code. And it decides something else: whether you may share the model. The course had to decide that itself: your GPT trains on TinyStories, and the GPT on tokens that the course shares is trained on other stories, from Cosmopedia. This lesson shows why.

    **An everyday example.** Someone who has only ever read cookbooks can tell you how to make soup, not how an engine works. And a cookbook copied from someone else's shelf without asking is not theirs to hand out.
    """)
    return


@app.cell(hide_code=True)
def predict_web():
    guess_web = L.guess(
        "web",
        question="Dolma, an open corpus of 3 trillion tokens, holds web pages, code, posts on social media, scientific papers, books and an encyclopedia. How much of it do you think is web pages?",
        options={
            "quarter": "About 25%",
            "half": "About 50%",
            "most": "About 80%",
        },
        reveal=f"**About 80%: {C.number(C.DOLMA['web pages (Common Crawl)'], 0, L.lang)} of its {C.number(C.DOLMA_TOTAL, 0, L.lang)} billion tokens are web pages, {C.number(C.share(C.DOLMA['web pages (Common Crawl)']), 0, L.lang)}%.** Code is {C.number(C.share(C.DOLMA['code (GitHub)']), 0, L.lang)}%, and the books and the encyclopedia together are less than half a percent. No other source has nearly as much text as the web.",
    )
    guess_web
    return (guess_web,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## The idea

        **Many sources, one pile.** The text a model pretrains on comes from many sources, joined into one corpus:

        - **Web pages**, by far the most. Common Crawl, a non-profit, has collected billions of web pages since 2008 and publishes them for free; corpora such as C4, FineWeb and Dolma clean them up for training.
        - **Code**, from public repositories such as those on GitHub.
        - **Scientific papers**, such as those on arXiv; **books**, such as the public-domain books of Project Gutenberg; **encyclopedias**, such as Wikipedia; and **forums**, such as Reddit and Stack Exchange.
        - **Text written by models**, such as the stories of TinyStories and Cosmopedia: synthetic data.

        **What the model knows follows from the pile.** A model can only learn what is in it. A corpus without code gives a model that cannot write code; a corpus in English, one that writes little Dutch; a corpus that ends in 2024, one with a knowledge cutoff in 2024: it knows nothing of what came after.

        **What you may do with it follows from the sources.** The course asks four questions of every text that goes into a model it shares, and keeps the answers in `data/README.md` and `course/route/bronnen.yaml`:

        1. **Is the text itself open?** Text in the public domain, or under a license such as CC BY, CC BY-SA, Apache-2.0 or MIT, is. With SA, the course gives a model it shares the same license, to be safe: whether SA reaches a model trained on the text is not settled.
        2. **Who wrote it?** Text written by a model can carry that model's terms. OpenAI's terms forbid using its output to build models that compete with OpenAI. The course does not want to argue about what competes, so it shares no model trained on such text; the license of the model Mixtral, Apache-2.0, sets no such condition.
        3. **Was it copied lawfully?** Never from a pirate source, not even to train on yourself: the exception of question 4 covers only text you may lawfully read. Books3, nearly 200,000 books from a pirate site, was part of a well-known open corpus, the Pile, until a request by the Danish Rights Alliance took it offline.
        4. **Did the maker opt out?** In the EU, text and data mining of text you may lawfully read is allowed, also for training AI, unless the rightsholder reserved it; for text online, in a machine-readable way, such as rules in robots.txt, the file in which a website tells crawlers what they may fetch (the EU's copyright directive of 2019, article 4).

        **The model has a license too.** Many models come with open weights, but open weights alone are not open source. The weights of Llama, the language model of Meta, can be downloaded by anyone, but its license limits who may use them and how. By the definition of the Open Source Initiative (2024), open-source AI comes with its weights and its code, and with enough information about its training data to build a similar model, all under terms that let anyone use, study, change and share it.

        ### New words and symbols

        | Word | What it means | In code |
        |---|---|---|
        | **pretraining corpus** | all the text a model learns from while pretraining, joined from many sources, often trillions of tokens | `{"web pages": 2479, "code": 411}` |
        | **Common Crawl** | a non-profit project that has collected billions of web pages since 2008 and publishes them for free: the main source of web text for pretraining | |
        | **synthetic data** | data made by a program instead of collected from the world; here, text written by a language model | |
        | **text and data mining** | analyzing large amounts of text and data by computer to find patterns, as training a model does | |
        | **opt-out** | a way for the owner of content to say that it may not be used, for example for training AI: through rules in robots.txt, metadata or a request | |
        | **open weights** | a model's trained parameters, published so that anyone can download and run them; its data and code may stay private, and its license may still set conditions | |
        | **open-source AI** | a model that anyone may use, study, change and share: with its weights, its code and enough information about its data to build a similar one | |
        """),
        L.watch_out("The license of a corpus can cover the collection and not the texts in it. FineWeb's license, ODC-By 1.0, lets you use and share the collection, if you name its makers; the web pages in it keep their own rights, so the course trains no model it shares on it."),
        mo.md(r"""
        The picture: the six sources of Dolma, in billions of tokens of the tokenizer of Llama, with the share of each.
        """),
        mo.Html(C.sources_picture({"title": "What Dolma is made of", "note": "billions of tokens · share of all 3,059",
                                   "alt": "Bars for the six sources of Dolma: web pages from Common Crawl 2,479 billion tokens, 81%; code from GitHub 411 billion, 13.4%; social media from Reddit 89 billion, 2.9%; scientific papers from Semantic Scholar 70 billion, 2.3%; books from Project Gutenberg 6 billion, 0.2%; the encyclopedia, Wikipedia and Wikibooks, 4.3 billion, 0.1%."}, L.lang)),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Before you pretrain a model on a corpus, and share it:

    1. List every source, with how many tokens it adds: that says what the model will know.
    2. For every source, find the license, and check what it covers: the collection, or the texts themselves.
    3. Find out who wrote the texts: people, or a model whose terms come along.
    4. Leave out pirate sources, and texts whose makers opted out.
    5. Write down the sources, their shares and their licenses next to the model, and license the model to match.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Step 1 for Dolma: its sources, in billions of tokens, as a dictionary, and the share of each, from the table in its paper.
    """)
    return


@app.cell
def example_shares():
    # 1. every source of Dolma, with its tokens in billions
    _dolma = {"web pages": 2479, "code": 411, "social media": 89, "scientific papers": 70, "books": 6.0, "encyclopedia": 4.3}
    _total = 0
    for _source in _dolma:
        _total = _total + _dolma[_source]
    print("all tokens:", round(_total, 1), "billion")
    # the share of every source, in percent
    for _source in _dolma:
        print(_source, round(100 * _dolma[_source] / _total, 1))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The six sources add up to 3,059.3 billion tokens; the paper rounds that to 3,059. Web pages are 81.0% and code 13.4%: a model trained on Dolma learns mostly from the web, and a fair amount of code. Books are 0.2% of it. Steps 2 to 4 need more than numbers: Dolma's web pages come from Common Crawl, and Dolma's license, ODC-By like FineWeb's, covers the collection, not the pages.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_where():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. Where may it go?

        Six text datasets, as the course meets them. Put each where the four questions of the idea put it: it may go into the models the course shares; it is only for you to train on, and the course shares no model trained on it; or not at all.
        """),
        L.task("Give every dataset its place."),
    ])
    return


@app.cell(hide_code=True)
def exercise_where():
    sort_where = L.sort("where")
    sort_where
    return (sort_where,)


@app.cell(hide_code=True)
def check_where(sort_where):
    result_where = L.check("where", sort_where)
    result_where
    return (result_where,)


@app.cell(hide_code=True)
def hints_where():
    L.hints(
        "where",
        [
            "Only a text that passes all four questions may go into the models the course shares. A text that is not open, or whose maker's terms rule out a shared model, may still be fine to learn from on your own computer; a text that was not copied lawfully is not.",
            "Two datasets are written by models, and one comes from a pirate site. For the others: does the license cover the texts, or only the collection?",
        ],
        why="Tiny Shakespeare is in the public domain, Wikipedia is CC BY-SA, so the course would give a model trained on it CC BY-SA too, to be safe, and Cosmopedia was written by an Apache-2.0 model: they may go into the models the course shares. TinyStories carries OpenAI's terms, and FineWeb's license covers only the collection: the course lets you train on them yourself, but shares no model trained on them. Books3 comes from a pirate site: not at all.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_license():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. How much is under which license?

        Step 2 of the recipe, in numbers. Complete `by_license(sources)`: every source is a dict with a `name`, its `tokens` and its `license`, and the function gives back a dict with every license and the share of all tokens its sources have, in percent, rounded to 1 decimal. The check calls `by_license` with an example corpus of five sources, in millions of tokens: Dutch Wikipedia, 900 under CC BY-SA 4.0; Project Gutenberg, 300 in the public domain; code on GitHub, 250 under MIT and 60 under Apache-2.0; and Dutch web pages, 3,500 under ODC-By 1.0. ODC-By 1.0 should come out at 69.9. Then it tries others.
        """),
        L.task("Make `by_license` add up the tokens of every license, and give each as a share of all."),
    ])
    return


@app.cell(hide_code=True)
def exercise_license():
    box_license = L.code_box("by-license", lines=14)
    box_license
    return (box_license,)


@app.cell(hide_code=True)
def check_license(box_license):
    result_license = L.check("by-license", box_license.value)
    result_license
    return (result_license,)


@app.cell(hide_code=True)
def hints_license():
    L.hints(
        "by-license",
        [
            "First the total, with a loop as in the worked example; then a second loop over `sources` that adds up the tokens per license, with `s[\"license\"]` as the key; then a loop over the licenses for their shares.",
            "A license you meet for the first time starts at the tokens of its source; a license you met before gets them added. Round only at the end, once per license.",
        ],
        why="Adding up per license first, and rounding once, gives each license exactly its part of the corpus. Rounding every source before adding them up would add up the roundings too.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_open():
    mo.md(r"""
    ### Exercise 3. Open weights or open source?
    """)
    return


@app.cell(hide_code=True)
def choice_open_cell():
    choice_open = L.choice(
        "open-weights",
        question="Anyone can download the weights of Llama 3. Its license forbids some uses, and companies with more than 700 million users a month must ask Meta first. Its training data is not published. Is Llama 3 open-source AI, by the definition of the Open Source Initiative?",
        options={
            "yes": "Yes: anyone can download its weights, run them, change them and build on them",
            "right": "No: open weights, but its terms limit its use and its data stays unknown",
            "free": "Yes, for everyone who has fewer than 700 million users a month",
        },
        correct="right",
        why={
            "yes": "That makes its weights open. Open-source AI asks more: terms that let anyone use, study, change and share it, its code, and enough about its data to build a similar model.",
            "free": "Terms that depend on who you are are not open-source terms: those let anyone use it, for any purpose.",
            "right": "Open weights, not open source. A model you build on Llama inherits its license; one you train yourself on open data, with your own code, can be open source.",
        },
    )
    choice_open
    return (choice_open,)


@app.cell(hide_code=True)
def text_exercise_optout():
    mo.md(r"""
    ### Exercise 4. An opt-out
    """)
    return


@app.cell(hide_code=True)
def choice_optout_cell():
    choice_optout = L.choice(
        "opt-out",
        question="A Dutch news site reserves text and data mining of its articles, in a machine-readable way in its robots.txt. A company in the EU crawls the site anyway, to train a model it sells. Does the EU's copyright directive allow that?",
        options={
            "right": "No: the site opted out of text and data mining",
            "public": "Yes: the articles are online, free for anyone to read",
            "learn": "Yes: a model learns from the text; it does not copy it",
        },
        correct="right",
        why={
            "public": "Free to read is not free to mine. Article 4 allows mining text you may lawfully read, unless the rightsholder reserved it, and this site did.",
            "learn": "Training starts with copying the pages to a computer, and that copy needs permission or an exception. The exception of article 4 falls away when the rightsholder opts out.",
            "right": "With a machine-readable reservation, article 4 no longer covers the company's mining. Whether a reservation counts, in plain words on a page or for some crawlers only, is for courts to decide, and they are still at it.",
        },
    )
    choice_optout
    return (choice_optout,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The datasheet of [Where does this data come from?](/course/open/data-provenance?lang=en), now for every source of a corpus of trillions of tokens; and the stories of [Project: your GPT on your own tokens](/course/open/gpt-train-bpe?lang=en), whose source decided which models the course may share. The next lessons clean such a corpus: throwing out bad text, then the same text twice, then the questions of tests.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_where, result_license, choice_open, choice_optout):
    L.wrap_up(result_where, result_license, choice_open, choice_optout)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
