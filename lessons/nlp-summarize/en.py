# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-summarize. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-summarize", "en")
    C = L.checks   # the course's six abstracts, Qwen3-0.6B's summaries, the idf, SciTLDR, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    One question about [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en), and one about [From data to text](/course/open/nlp-what-is-nlg?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_idf():
    quick_idf = L.choice(
        "quick-idf",
        question="In *TF-IDF: weighing the words*, the idf of a word is log10(n / df): n documents, of which df hold the word. Take the six abstracts of this lesson as the documents. What is the idf of *the*, which is in all six?",
        options={
            "one": "1, as the word is in every document",
            "right": "0",
            "six": "log10(6), about 0.78",
            "none": "Nothing: idf only exists for a word that some documents lack",
        },
        correct="right",
        why={
            "right": "log10(6 / 6) is log10(1), and that is 0. A word in every document does not tell the documents apart at all, so it weighs nothing: this lesson uses that, so that *the* and *we* add nothing to a score.",
            "one": "1 is n / df, before the log10. What is log10 of 1? See [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en).",
            "six": "log10(6) is the idf of a word in one of the six documents. What is df for a word in all six? See [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en).",
            "none": "The formula works for every word that is in at least one document. What does log10(6 / 6) give? See [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en).",
        },
    )
    quick_idf
    return (quick_idf,)


@app.cell(hide_code=True)
def quick_check_selection():
    quick_selection = L.choice(
        "quick-selection",
        question="In *From data to text*, content selection decides which facts of the data a text says. A summary is a shorter text made from a longer one. Which part of making a summary is content selection?",
        options={
            "words": "Choosing a word for each fact, such as *windy* for wind force 6",
            "right": "Choosing which sentences of the text the summary keeps",
            "commas": "Putting a comma between names, and *and* before the last",
            "file": "Reading the text from a file first, so that the program has all of its words",
        },
        correct="right",
        why={
            "right": "Content selection decides what to say, and a summary that keeps sentences of the text makes exactly that choice: which sentences stay. This lesson builds one.",
            "words": "Choosing words is microplanning, the middle step. Which step decides what is said at all? See [From data to text](/course/open/nlp-what-is-nlg?lang=en).",
            "commas": "Commas and *and* belong to surface realization, the last step. Which step decides what is said? See [From data to text](/course/open/nlp-what-is-nlg?lang=en).",
            "file": "Reading a file comes before the three steps and decides nothing about the text. Which step chooses what the text says? See [From data to text](/course/open/nlp-what-is-nlg?lang=en).",
        },
    )
    quick_selection
    return (quick_selection,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    In 1958, Hans Peter Luhn of IBM let an IBM 704 computer pick sentences from articles. Luhn wrote: "The purpose of abstracts in technical literature is to facilitate quick and accurate identification of the topic of published papers. The objective is to save a prospective reader time and effort in finding useful information in a given article or report." The program counted words: "Statistical information derived from word frequency and distribution is used by the machine to compute a relative measure of significance, first for individual words and then for sentences. Sentences scoring highest in significance are extracted and printed out to become the “auto-abstract.”"

    Today even the abstracts are too many to read. In 2020, Isabel Cachola, Kyle Lo, Arman Cohan and Daniel S. Weld of the Allen Institute for AI proposed a summary of one sentence for each paper, the *TLDR*, short for *too long; didn't read*: "An alternative to abstracts, TLDRs focus on the key aspects of the paper, such as its main contributions, eschewing nonessential background or methodological details."

    In this lesson you write a summarizer that picks sentences, measure it on 618 real abstracts, and measure what a summarizer that writes adds.

    **An everyday example.** Under each result, a search engine shows a few lines of the page, often with your search words in bold: a summary picked from the page itself.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_first = L.guess(
        "guess",
        question="SciTLDR, the set of Cachola and colleagues, holds 618 abstracts of papers in computer science with their TLDRs, and marks in each abstract the one sentence that comes closest to its TLDRs. In news, the first sentences usually say what matters most. How often is the closest sentence of an abstract its first sentence?",
        options={
            "most": "In about 3 of every 4 abstracts",
            "half": "In about half of them",
            "right": "In about 1 of every 5",
            "rarely": "In fewer than 1 of every 20",
        },
        reveal="In about 1 of every 5: in 118 of the 618 abstracts, 19.1%. A sentence picked at random would be the closest one in 14.8% of the abstracts, so the first sentence does only a little better than chance. For news, Abigail See and colleagues write that \"news articles tend to be structured with the most important information at the start; this partially explains the strength of the lead-3 baseline\": the summary of the first three sentences. Abstracts often start with background instead, as the abstract about FearNet does: *Incremental class learning involves sequentially learning classes in bursts of examples from the same class.* Its closest sentence is the fifth: *FearNet is a generative model that does not store previous examples, making it memory efficient.*",
    )
    guess_first
    return (guess_first,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    Abigail See, Peter Liu and Christopher Manning, in 2017: "Summarization is the task of condensing a piece of text to a shorter version that contains the main information from the original. There are two broad approaches to summarization: extractive and abstractive. Extractive methods assemble summaries exclusively from passages (usually whole sentences) taken directly from the source text, while abstractive methods may generate novel words and phrases not featured in the source text – as a human-written abstract usually does."

    1. An **extractive summary** picks sentences: it is content selection on a text, as in *From data to text*. A summarizer gives each sentence a **score** and keeps the highest. Luhn scored sentences by their frequent words. In 1969, H. P. Edmundson tried three more clues: cue words such as *significant*, the place of a sentence in the text, and the words of the title. Of the title, Edmundson wrote: "The Title method is based on the hypothesis that an author conceives the title as circumscribing the subject matter of the document." And of the results: "The results indicate that the three newly proposed components dominate the frequency component in the production of better extracts." This lesson builds the title method, with each title word weighed by its idf.
    2. An **abstractive summary** writes sentences of its own, as a language model does. It can join facts and drop details, but it can also say what the text does not. Joshua Maynez and colleagues found in 2020 that such summarizers "are highly prone to hallucinate content that is unfaithful to the input document." They name two kinds: an **intrinsic hallucination** twists what the text says, and an **extrinsic hallucination** adds what the text does not say.

    How abstractive is a summary? Cachola and colleagues: "we measure abstractiveness (or novelty) by percentage of words in the summary that do not appear in the source document."

    And how good is a pick? SciTLDR marks in each abstract the **oracle sentence**, the one sentence a method would best pick. Cachola and colleagues: "Given a paper and its multiple gold TLDRs, it selects the single sentence in the document with the highest Rouge overlap for each gold TLDR. Then it returns the single sentence that yields the maximum Rouge across all gold TLDRs." ROUGE is the subject of the next lesson; here it is enough that the oracle sentence shares the most words with the TLDRs. For the six abstracts that the course wrote, the course takes the sentence that holds the largest share of the different words of its TLDR.

    The six abstracts are made up in the style of computer science, each with a title and a TLDR that the course wrote. Qwen3-0.6B, the chat model of the course, a small language model trained further to answer in a chat, summarized each in one sentence; the course recorded its answers. The picture shows both kinds of summary on the abstract about bees.
    """)
    return


@app.cell(hide_code=True)
def idea_cell():
    mo.Html(C.idea_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_words():
    mo.md(r"""
    ### New words and symbols

    | Word | What it means |
    |---|---|
    | **summarization** | making a text shorter while keeping its main content |
    | **extractive summary** | a summary of sentences picked from the text, left as they are |
    | **abstractive summary** | a summary in sentences of its own, which may hold words that are not in the text |
    | **score** | a number for each sentence that says how much it belongs in the summary |
    | **title method** | scoring a sentence by the words it shares with the title, after Edmundson |
    | **TLDR** | a summary of one sentence; short for *too long; didn't read* |
    | **oracle sentence** | the sentence of a text closest to its reference summaries: the best one sentence a method could pick |
    | **novelty** | the share of the different words of a summary that are not in the text |
    | **hallucination** | what a summary says that the text does not support: intrinsic when it twists the text, extrinsic when it adds to it |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To summarize a text with the title method:

    1. Split the title and each sentence into words, in small letters.
    2. Score each sentence: for each different word that is also in the title, add its idf. A word in every document, such as *the*, adds 0.
    3. Pick the k sentences with the highest scores; of two equal scores, the earlier sentence.
    4. Give the picked sentences in the order of the text.

    To measure how abstractive a summary is: count its different words that are not in the text, and divide by the number of its different words.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Take sentence 2 of the abstract about bees, counted from 0. `C.ABSTRACTS` holds the six abstracts, each a dictionary with `"title"`, `"sentences"` and `"tldr"`. `C.words` splits a text into words in small letters, as in *TF-IDF: weighing the words*, and `C.IDF` gives the idf of each word over the six abstracts, with each abstract and its title as one document. The cell prints the words of the title, then each word of the sentence that is also in the title with its idf, and last the score:
    """)
    return


@app.cell
def example_cell():
    _abstract = C.ABSTRACTS["bees"]
    _title = C.words(_abstract["title"])
    print(_title)
    _sentence = C.words(_abstract["sentences"][2])
    _shared = [_w for _w in sorted(set(_sentence)) if _w in _title]
    _total = 0
    for _w in _shared:
        print(_w, round(C.IDF[_w], 3))
        _total = _total + C.IDF[_w]
    print("score:", round(_total, 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Nine of the eleven title words are in sentence 2. *the* and *a* are in all six abstracts and add 0, and *network*, in five of the six, adds only 0.079. *at* adds 0.301 and *small* 0.176, and *bees*, *entrance*, *hive* and *neural* are in this abstract alone and add 0.778 each. The score is 3.67, the highest of the seven sentences: the first sentence scores 1.56, with *bees* and *hive*, as the picture in *The idea* shows. *counting* in the title is not *counts* in the sentence: to the method, two forms of a word are two words. The lesson *Stems and lemmas* shows how to bring such forms together; the title method here does not.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_title():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. title_score: the title words of a sentence

        Write `title_score`, which gets `sentence` and `title`, each a list of words, and `idf`, a dictionary from each word to its idf, such as `C.IDF`. Give back the score of the sentence: the idf of each different word of the sentence that is also in the title, added up. In the start code, `score` starts at 0, and the loop goes over the different words of the sentence, each once. Replace the two `...`.

        The check calls `title_score(["bees", "count", "bees", "at", "the", "hive"], ["counting", "bees", "at", "the", "hive"], {"bees": 0.8, "count": 0.3, "at": 0.2, "the": 0.0, "hive": 0.8, "counting": 0.6})`, which gives `1.8`: *bees*, *at*, *the* and *hive* are in the title, and *bees* counts once. Then `title_score(["het", "netwerk", "telt", "bijen"], ["bijen", "tellen", "met", "een", "netwerk"], {"het": 0.0, "netwerk": 0.1, "telt": 0.8, "bijen": 0.8, "tellen": 0.8, "met": 0.0, "een": 0.0})`, with the words of the Dutch lesson, and cases of its own: a sentence without title words, a word that comes three times, and sentences of some of the six abstracts with `C.IDF`.
        """),
        L.task("Score a sentence by the idf of the words it shares with the title."),
    ])
    return


@app.cell(hide_code=True)
def exercise_title():
    box_title = L.code_box("title_score", lines=7)
    box_title
    return (box_title,)


@app.cell(hide_code=True)
def check_title(box_title):
    result_title = L.check("title_score", box_title.value)
    result_title
    return (result_title,)


@app.cell(hide_code=True)
def hints_title():
    L.hints(
        "title_score",
        [
            "`word` is one different word of the sentence. `in` asks whether a list holds a value, as in *Lists*, and `idf` is a dictionary: a key gives its value, as in *Dictionaries*. `score` grows one value at a time, as a running total does.",
            "The `if` asks whether the word is in the title. Inside it, the idf of that one word goes on top of what `score` holds: `score` on the left of `=`, and on the right `score` plus that value. Look up the word in the dictionary, not the whole sentence.",
            "`            score = score + ___`",
        ],
        why="`if word in title:` and `score = score + idf[word]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_title():
    mo.md(r"""
    In 3 of the six abstracts, the sentence with the highest score is the one closest to the TLDR: about bees, rivers and wind power. In the abstract about tomato leaves, the closest sentence is the one with the result, *the model finds 89% of the diseased leaves*, and the title method picks the one with the method. In the abstract about leaks, the title says *Leaks*, and the sentence that says what the paper does has *a leak*: the method picks the first sentence, about water companies, which has *leaks*, *water* and *pipes*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pick():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. pick: the best sentences, in the order of the text

        Write `pick`, which gets `scores`, the score of each sentence in the order of the text, and `k`, how many sentences to pick. Give back the places of the k highest scores, counted from 0, in the order of the text; of two equal scores, the earlier place goes first. In the start code, the first loop puts a tuple in `pairs` for each place `i`; `sorted(pairs)` puts the tuples in order, and the second loop takes the place out of each of the first k. `sorted` puts tuples in order by their first value, and tuples with the same first value by their second. Replace the two `...`.

        The check calls `pick([0.8, 2.6, 1.1, 2.6, 1.1], 3)`, which gives `[1, 2, 3]`: the two scores of 2.6, and of the two scores of 1.1 the earlier, at place 2. Then `pick([0.5, 0.0, 1.9], 1)`, which gives `[2]`, and cases of its own: k equal to the number of sentences, k equal to 0, scores that are all the same, and the scores of three of the six abstracts.
        """),
        L.task("Pick the places of the k highest scores, and give them in the order of the text."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pick():
    box_pick = L.code_box("pick", lines=9)
    box_pick
    return (box_pick,)


@app.cell(hide_code=True)
def check_pick(box_pick):
    result_pick = L.check("pick", box_pick.value)
    result_pick
    return (result_pick,)


@app.cell(hide_code=True)
def hints_pick():
    L.hints(
        "pick",
        [
            "`pairs.append` adds one tuple to `pairs` for place `i`, and `scores[i]` is the score at that place, as in *Lists*. The place `i` comes second in the tuple, as the second loop takes it out with `pair[1]`. `sorted` goes from small to large, as in *Tuples, sets and sorting*.",
            "For the highest score to come first in a sort from small to large, put the score in the tuple with a minus in front: 2.6 becomes -2.6, which is smaller than -1.1. On a tie, `sorted` looks at the place, so the earlier place comes first. The last line gives the chosen places from small to large, which is the order of the text.",
            "`        pairs.append((___, i))`",
        ],
        why="`pairs.append((-scores[i], i))` and `return sorted(chosen)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_pick():
    mo.md(r"""
    With k = 2, the summary of the abstract about bees holds sentences 0 and 2: sentences 0 and 1 tie at 1.56, and the earlier one comes along. The same rule decides the abstract about plastic with k = 1: sentences 1 and 2 tie at 2.03, so the title method picks sentence 1, about dark plastic, and not sentence 2, the closest one. Edmundson ran into ties too: in the first version of the weights of the Title method, "this caused many ties among sentence weights."

    ### Try it: 618 real abstracts

    SciTLDR holds 3,229 papers in computer science, with TLDRs of the authors of submissions on OpenReview and TLDRs that experts wrote from the peer reviews: what other researchers wrote about each paper when it was submitted. Its test set has 618 papers with 1,967 TLDRs, the TLDR of the authors first. The course uses the version with the abstract as the text, SciTLDR-A, under the license Apache-2.0. The demo picks one sentence from each abstract in three ways: by title words, as your `title_score` does, with the idf over the 618 abstracts; by word weights, after Luhn: the TF-IDF weights of the different words of a sentence, each with its count in the whole abstract, as in *TF-IDF: weighing the words*; and the first sentence. Choose a paper with the slider to see its sentences and its TLDRs.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label="Download SciTLDR, 1.2 MB")
    return (get_button,)


@app.cell(hide_code=True)
def scitldr_cell(get_button):
    sci_papers = C.scitldr_or_stop(get_button, {
        "ask": "*The test set of SciTLDR is not on your computer yet. The button downloads it once from its repository on GitHub, 1.2 MB, under the license Apache-2.0. The rest of the lesson works without it.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: GitHub could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: the file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: your disk has no room for 1.2 MB. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    sci_paper = mo.ui.slider(1, len(sci_papers), value=C.SCI_START, label="Paper", show_value=True)
    sci_paper
    return sci_paper, sci_papers


@app.cell(hide_code=True)
def scitldr_look(sci_paper, sci_papers):
    mo.Html(C.scitldr_picture(L.lang, sci_papers, sci_paper.value))
    return


@app.cell(hide_code=True)
def text_after_scitldr():
    mo.md(r"""
    The title method picks the closest sentence in 222 of the 618 abstracts, 35.9%: about twice as often as word weights, with 104, or 16.8%, which does worse than the first sentence. Edmundson saw the same order in 1969: "The Cue-Title-Location method is seen to have the highest mean coselection score, while the Key method in isolation is the lowest of the automatic methods." The Key method is the method of frequent words. Abstracts in computer science often say what the paper does with the words of its title: *We propose Aggregated Momentum (AggMo)* stands in the paper *Aggregated Momentum: Stability Through Passive Damping*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_novel():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. novel_share: how much of a summary is new

        Write `novel_share`, which gets `summary` and `source`, each a list of words; the summary has at least one word. Give back its novelty: the share of the different words of the summary that are not in the source, a number from 0 to 1. In the start code, `words` holds the different words of the summary, and `new` counts those that are not in the source. Replace the two `...`.

        The check calls `novel_share(["tiny", "network", "counts", "bees", "bees"], ["we", "present", "a", "small", "network", "that", "counts", "bees"])`, which gives `0.25`: of the four different words, only *tiny* is new. Then `novel_share(["het", "netwerk", "telt", "bijen"], ["we", "tellen", "bijen", "met", "een", "netwerk"])`, with the words of the Dutch lesson, which gives `0.5`, and cases of its own: a summary of copied words only, a summary of new words only, and summaries and TLDRs of some of the six abstracts.
        """),
        L.task("Give the share of the different words of a summary that are not in its source."),
    ])
    return


@app.cell(hide_code=True)
def exercise_novel():
    box_novel = L.code_box("novel_share", lines=7)
    box_novel
    return (box_novel,)


@app.cell(hide_code=True)
def check_novel(box_novel):
    result_novel = L.check("novel_share", box_novel.value)
    result_novel
    return (result_novel,)


@app.cell(hide_code=True)
def hints_novel():
    L.hints(
        "novel_share",
        [
            "`word` is one different word of the summary, and `new` counts the words that are not in the source. `not in` asks whether a list lacks a value. A share is a part divided by the whole, with `/`.",
            "The whole is the number of different words of the summary, not of all its words: *bees* comes twice in the first case, and `new` counts each different word once. `len` gives the number of values in a list or a set.",
            "`    return new / ___`",
        ],
        why="`if word not in source:` and `return new / len(words)`.",
    )
    return


@app.cell(hide_code=True)
def text_after_novel():
    mo.md(r"""
    Qwen3-0.6B's six summaries have 33.2% new words on average, and the six TLDRs that the course wrote 17.2%. Qwen3 adds words such as *improving*, *accuracy* and *outperforming*, and those words carry the claims that the abstracts do not support: in four of the six summaries, in the course's reading. The summary about tomato leaves calls the model that does poorly a phone camera model, and leaves out that the network of the abstract, trained further on phone photos, finds 89% of the diseased leaves; the model that does poorly is the one trained on lab photos: an intrinsic hallucination. The summary about leaks says that the network outperforms traditional methods, while the abstract compares it with no other method: an extrinsic one. New words are not wrong by themselves: *reducing manual labor* says in other words that counting by hand takes hours.

    On SciTLDR, the TLDRs of the authors have 23.3% new words, measured against the title and the abstract. Cachola and colleagues measure against the whole paper and find 9.6%, as a TLDR may also take words from the rest of the paper. Maynez and colleagues had people check summaries of one sentence of BBC articles, made by the summarizers of 2020: intrinsic and extrinsic hallucinations "happen frequently – in more than 70% of single-sentence summaries."

    ### Try it: pick or write

    Choose an abstract, a way to score its sentences, and how many sentences to pick. Under the picked sentences stand Qwen3-0.6B's summary and the TLDR that the course wrote. Try the abstract about leaks with word weights: even four sentences leave out the closest one. Then try the abstract about tomato leaves with title words: how many sentences does it take before the closest one comes along?
    """)
    return


@app.cell(hide_code=True)
def try_controls():
    try_abstract = mo.ui.dropdown(options={C.ABSTRACT_NAMES[L.lang][_key]: _key for _key in C.ABSTRACTS},
                                  value=C.ABSTRACT_NAMES[L.lang]["leaks"], label="Abstract")
    try_method = mo.ui.dropdown(options={C.METHOD_NAMES[L.lang][_m]: _m for _m in C.METHODS},
                                value=C.METHOD_NAMES[L.lang]["title"], label="Score by")
    try_k = mo.ui.slider(1, 4, step=1, value=1, label="Sentences", show_value=True)
    mo.hstack([try_abstract, try_method, try_k], justify="start", gap=1.5, wrap=True)
    return try_abstract, try_k, try_method


@app.cell(hide_code=True)
def try_look(try_abstract, try_k, try_method):
    mo.Html(C.try_picture(L.lang, try_abstract.value, try_method.value, try_k.value))
    return


@app.cell(hide_code=True)
def text_exercise_risks():
    mo.md(r"""
    ### Exercise 4. Which summary runs this risk?

    Each problem below makes a summary worse. Put each with the kind of summary that runs it.
    """)
    return


@app.cell(hide_code=True)
def exercise_risks():
    sort_risks = L.sort("risks")
    sort_risks
    return (sort_risks,)


@app.cell(hide_code=True)
def check_risks(sort_risks):
    result_risks = L.check("risks", sort_risks)
    result_risks
    return (result_risks,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [TF-IDF: weighing the words](/course/open/nlp-tf-idf?lang=en) gave each word a weight by how few documents hold it; here that weight lets the words of the title count and *the* not. [From data to text](/course/open/nlp-what-is-nlg?lang=en) took three steps from data to text; an extractive summary takes only the first, content selection, on a text, and leaves the sentences as they are. A summarizer that writes, as a language model does in [A language model predicts the next token](/course/open/lm-what-is-a-language-model?lang=en), reads better and can say what the text does not. The next lesson, *Scoring a summary: ROUGE*, builds the measure that SciTLDR used to find the closest sentence, and scores a summary against summaries that people wrote.

    **An everyday example.** The preview of an email in an inbox shows its first line: the simplest extractive summary, and, as this lesson found for abstracts, not always the line that matters most.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_title, result_pick, result_novel, result_risks):
    L.wrap_up(result_title, result_pick, result_novel, result_risks)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
