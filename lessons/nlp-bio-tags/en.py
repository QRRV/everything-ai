# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-bio-tags. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-bio-tags", "en")
    C = L.checks   # the sentences of Universal NER, the pictures and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions, about [Parts of speech and phrases](/course/open/nlp-word-classes?lang=en) and [Cutting text into sentences and words](/course/open/nlp-tokenize?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_book():
    quick_book = L.choice(
        "quick-book",
        question="In *Parts of speech and phrases*, *Book* was a `VERB` in *Book that flight.*, and *book* a `NOUN` in *Hand me that book.* What decided the tag each time?",
        options={
            "capital": "The capital letter: *Book* with a capital is a verb",
            "right": "The words around it",
            "first": "The place in the sentence: the first word is always a verb",
            "list": "A dictionary that gives every word one tag",
        },
        correct="right",
        why={
            "right": "Only the words around it tell *book* the verb from *book* the noun. This lesson gives every word a label, as the tags of parts of speech do.",
            "capital": "*Book* has a capital because it is the first word of the sentence. What makes it a verb is *that flight* after it. See [Parts of speech and phrases](/course/open/nlp-word-classes?lang=en).",
            "first": "Many sentences start with a noun or a pronoun, such as *We had a fantastic time.* Here *that flight* after *Book* makes it a verb. See [Parts of speech and phrases](/course/open/nlp-word-classes?lang=en).",
            "list": "Then *book* would have the same tag in both sentences. The same letters had two tags, and the words around them decided which. See [Parts of speech and phrases](/course/open/nlp-word-classes?lang=en).",
        },
    )
    quick_book
    return (quick_book,)


@app.cell(hide_code=True)
def quick_check_tokens():
    quick_tokens = L.choice(
        "quick-tokens",
        question="A program gives one label to every token of *We don't know.*, cut into tokens the way the Penn Treebank does it. How many labels does it give?",
        options={
            "spaces": "3: one for each piece between the spaces",
            "no_nt": "4: *n't* is not a word, so it gets no label",
            "right": "5: *don't* gives two tokens",
            "whole": "4: *We*, *don't*, *know* and the period",
        },
        correct="right",
        why={
            "right": "*We*, *do*, *n't*, *know* and the period. The Penn Treebank splits *don't* into *do* and *n't*, and the treebank of this lesson does the same. One label per token means as many labels as tokens, so the standard that cuts the tokens also decides how many labels there are.",
            "spaces": "Cutting at the spaces gives *We*, *don't* and *know.*, with the period stuck to *know*. The Penn Treebank splits off the period, and splits *don't* into *do* and *n't*. See [Cutting text into sentences and words](/course/open/nlp-tokenize?lang=en).",
            "no_nt": "*n't* is a token of its own, and every token gets a label; when the labels mark names, *n't* gets the label for no name. See [Cutting text into sentences and words](/course/open/nlp-tokenize?lang=en).",
            "whole": "The Penn Treebank splits *don't* into two tokens, *do* and *n't*, and each token gets its own label. See [Cutting text into sentences and words](/course/open/nlp-tokenize?lang=en).",
        },
    )
    quick_tokens
    return (quick_tokens,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Many questions about a text are about its pieces, not about the text as a whole. Which words are names of people, places and organizations? In *wake me up at seven tomorrow*, which words say when? In a line of an address, which words are the street and which the town? A spam filter gives a whole message one label; these tasks give every word a label of its own. The tools for parts of speech already give one label per word, but a name can be several words long. This lesson shows how a stretch of words becomes one label per word, and back again.

    **An everyday example.** You mark the names in a letter with a highlighter. When two names stand side by side, such as a town and the state after it, one long yellow stroke makes them look like one name. A small gap at the start of the second stroke shows where it begins.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_one = L.guess(
        "guess",
        question="Universal NER is a dataset in which people marked every name of a person, a place or an organization. This lesson uses 11,722 of its English sentences, from the web: reviews, questions with their answers, blogs and newsgroups. How many of every 100 tokens there are outside every name?",
        options={
            "seventy": "About 70: the web is full of names",
            "eighty-five": "About 85",
            "right": "About 95",
            "ninety-nine": "About 99",
        },
        reveal="**About 95:** 187,363 of the 197,019 tokens are outside every name, 95.1 in 100. The other 9,656 tokens make up 6,429 names, and 8,089 of the 11,722 sentences have no name at all. Almost every token gets the same label; the few labels that matter mark where a name begins and what kind of name it is.",
    )
    guess_one
    return (guess_one,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    ### One label per token: sequence labeling

    In [Parts of speech and phrases](/course/open/nlp-word-classes?lang=en), every word of a treebank had a tag, so a sentence of ten tokens had ten tags. Jurafsky and Martin give that kind of task a name: "Such tasks in which we assign, to each word $x_i$ in an input word sequence, a label $y_i$, so that the output sequence $Y$ has the same length as the input sequence $X$ are called sequence labeling tasks." Tagging parts of speech is **sequence labeling**, and so is finding names, once each name is written as labels on its tokens. A label belongs to a token, as [Cutting text into sentences and words](/course/open/nlp-tokenize?lang=en) cut them: *don't* is two tokens, *do* and *n't*, with a label each.

    ### Names are spans

    A name can be one token, such as *India*, or more, such as *San Diego* or *Byron York*. A **span** is a stretch of tokens that stand next to each other in a sentence. **Universal NER** (Stephen Mayhew and others, 2024) marks names in 19 datasets of 13 languages, all on sentences of Universal Dependencies. For English, it marks among others the sentences of EWT, the treebank of the lesson about parts of speech, in an earlier release. This lesson uses those sentences without the emails, which hold the names and addresses of real people. Each span has one of three **types**. In the words of Mayhew and others: "Universal NER's annotation schema emphasizes three coarse-grained entity types: Person (PER), Organization (ORG), and Location (LOC)." Titles stay outside a name: "We never annotate titles as a name or part of a name", say its guidelines, so in *Mr. Robinson* only *Robinson* is a person.

    Of the 6,429 names in the 11,722 sentences, 2,222 are more than one token, about 1 in 3; the longest has 16. In Python, the course writes a span as a tuple `(start, end, kind)`. `start` is the position of its first token, counted from 0, and `end` the position just after its last, as in a slice: `words[7:9]` is `["San", "Diego"]`, the span `(7, 9, "LOC")`. In a text, a span is often written between brackets with its type, as in [LOC San Diego].
    """)
    return


@app.cell(hide_code=True)
def spans_cell():
    mo.Html(C.spans_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_bio():
    mo.md(r"""
    ### BIO: a span as labels

    To write spans as one label per token, every token gets one of three kinds of label: **B** on the first token of a span (begin), **I** on the tokens after it in the span (inside), and **O** on every token outside all spans (outside). B and I carry the type, as in `B-LOC` and `I-LOC`. In the words of Jurafsky and Martin: "In BIO tagging we label any token that begins a span of interest with the label B, tokens that occur inside a span are tagged with an I, and any tokens outside of any span of interest are labeled O." With three types, that makes 2 × 3 + 1 = 7 labels. This is **BIO tagging**.

    ### Why B? Two names that touch

    Simpler is **IO tagging**: `I-` on every token of a span, and `O` on the rest. Jurafsky and Martin note that it "loses some information by eliminating the B tag". What it loses: two spans of the same type that touch. *Miramar California* becomes `I-LOC I-LOC`, which reads back as one place of two tokens, as *San Diego* does. The guidelines of Universal NER ask for two spans here: "When two locations are consecutive, we annotate separately". With BIO, `B-LOC B-LOC` says just that.
    """)
    return


@app.cell(hide_code=True)
def bio_cell():
    mo.Html(C.bio_picture(L.lang))
    return


@app.cell(hide_code=True)
def text_idea_back():
    mo.md(r"""
    In the 11,722 sentences, a `B-` comes right after a span of the same type 33 times: 24 times two names, mostly a place right before the bigger place it lies in, such as [LOC san francisco] [LOC california], and 9 times what looks like a slip, such as *George Bush* tagged as two persons, *George* and *Bush*. People who label make mistakes too.

    ### Other schemes

    Lance Ramshaw and Mitchell Marcus brought in the three tags in 1995, with a B only where two spans touch. Erik Tjong Kim Sang and Jorn Veenstra (1999) called that scheme IOB1, and the scheme with a B on every first token IOB2; Universal NER uses IOB2, and so does this lesson. **BIOES** adds E for the last token of a span and S for a span of one token. All of them hold the same spans, and a program can turn one into the other.

    ### From labels back to spans

    Reading tags back goes from left to right. A `B-` starts a span. An `I-` right after a span of the same type makes that span one token longer. An `O` belongs to no span. A **tagger**, a program that gives the labels, can also give an `I-` that continues nothing: after an `O`, or after a span of another type. The evaluation script of the CoNLL shared tasks reads such an `I-` as the start of a new span, and so does this lesson.
    """)
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | Meaning |
        |---|---|
        | **sequence labeling** | a task that gives every token of a sequence one label, so there are as many labels as tokens |
        | **span** | a stretch of tokens that stand next to each other, such as *San Diego*; in the course a tuple `(start, end, kind)` |
        | **Universal NER** | a dataset of names in 13 languages, on sentences of Universal Dependencies |
        | **type** | the kind of name a span is: `PER` (a person), `LOC` (a place) or `ORG` (an organization) |
        | **BIO tagging** | labels for spans: `B-` and the type on the first token of a span, `I-` and the type on the tokens after it, `O` on all other tokens |
        | **IO tagging** | labels with only `I-` and `O`: two spans of the same type that touch become one |
        | **IOB1**, **IOB2**, **BIOES** | other schemes: a B only where two spans touch; a B on every first token, as here; and E and S for the last token and for a span of one token |
        | **tagger** | a program that gives every token a label |
        | `C.UNER` | the 11,722 English sentences of Universal NER from EWT that this lesson uses: a tuple of sentences, each a tuple of pairs `(word, tag)` |
        | `C.EXAMPLE` | the sentence of the pictures, *Miramar California is a bit north of San Diego.*, as pairs `(word, tag)` |
        | `["O"] * n` | a list with `n` times `"O"`: `["O"] * 3` is `["O", "O", "O"]` |
        | `spans[-1]` | the last item of the list `spans`; `spans[-1] = x` puts `x` in its place |
        """),
        L.watch_out("In other lessons of the course, about following what an AI application does step by step, a *span* is one step of a trace, such as a single call to a model. Here a span is a stretch of tokens in a sentence: the same word for something else."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    1. From spans to tags: start with `O` for every token. For each span `(start, end, kind)`, the token at `start` gets `B-` and the type, and the tokens after it, up to `end` and without `end`, get `I-` and the type.
    2. From tags to spans: go over the tags from left to right. At a `B-`, start a new span of one token. At an `I-`, look at the last span: when it ends right before this token and has the same type, make it one token longer; otherwise start a new span. At an `O`, do nothing.
    3. Check both ways at once: the spans you read back from your tags are the spans you started with.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    The cell makes IO tags for the spans of `C.EXAMPLE`, and prints each word with its tag in Universal NER and its IO tag.
    """)
    return


@app.cell
def example_io():
    def io_tags(n, spans):
        tags = ["O"] * n
        for start, end, kind in spans:
            for i in range(start, end):
                tags[i] = "I-" + kind
        return tags


    example_spans = [(0, 1, "LOC"), (1, 2, "LOC"), (7, 9, "LOC")]
    tags_io = io_tags(len(C.EXAMPLE), example_spans)
    for i in range(len(C.EXAMPLE)):
        word, tag = C.EXAMPLE[i]
        print(word, tag, tags_io[i])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Universal NER has `B-LOC` on *Miramar* and on *California*: two places. The IO tags give both `I-LOC`, just as *San* and *Diego*, which are one place. Exercise 1 makes the BIO tags.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_tags():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. to_tags: from spans to BIO tags

        Write `to_tags(n, spans)`: give back the list of `n` tags of a sentence of `n` tokens with these `spans`, each a tuple `(start, end, kind)`. Start with `n` times `"O"`. For each span, the token at `start` gets `"B-"` and the type; the tokens after it, up to `end` and without `end`, get `"I-"` and the type. Replace the `...`. The check calls `to_tags(10, [(0, 1, "LOC"), (1, 2, "LOC"), (7, 9, "LOC")])`, the spans of the example, which gives `['B-LOC', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'O']`; then cases of its own: two more sentences of Universal NER, no spans, a span of three tokens, a span up to the last token, spans out of order, and one token.
        """),
        L.task("Give the first token of each span a B-, the other tokens of the span an I-, and every other token an O."),
    ])
    return


@app.cell(hide_code=True)
def exercise_tags():
    box_tags = L.code_box("to_tags", lines=7)
    box_tags
    return (box_tags,)


@app.cell(hide_code=True)
def check_tags(box_tags):
    result_tags = L.check("to_tags", box_tags.value)
    result_tags
    return (result_tags,)


@app.cell(hide_code=True)
def hints_tags():
    L.hints(
        "to_tags",
        [
            "Begin as the worked example does, with `[\"O\"] * n` and a loop over the spans. Inside it, the token at `start` gets a line of its own, and a second loop over `i` in a `range` that stops at `end` does the tokens after it.",
            "At which position does the second loop start: at `start`, or further on? Which token has its tag already?",
            "`        for i in range(___, end):`",
        ],
        why="`tags = [\"O\"] * n`, then `for start, end, kind in spans:` with `tags[start] = \"B-\" + kind` under it, and `for i in range(start + 1, end):` with `tags[i] = \"I-\" + kind` under that; after the loop `return tags`. `to_tags(10, [(0, 1, \"LOC\"), (1, 2, \"LOC\"), (7, 9, \"LOC\")])` gives `['B-LOC', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'O']`.",
    )
    return


@app.cell(hide_code=True)
def text_after_tags():
    mo.md(r"""
    The picture under the check runs your function on all 11,722 sentences of Universal NER, with the spans that the course reads from their tags. When every sentence comes out right, the spans and the tags hold the same information: nothing gets lost on the way. Exercise 2 goes the other way.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_spans():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. to_spans: from tags back to spans

        Write `to_spans(tags)`: give back the list of the spans `(start, end, kind)` that `tags` marks, from left to right. Start with an empty list `spans`, and go over the positions `i` of `tags`. At a tag that starts with `B`, add a new span of one token, with the type `tag[2:]`: the tag without its first two characters. At a tag that starts with `I`, look at the last span, `spans[-1]`: when there is one that ends right before this token and has the same type, put a span in its place that is one token longer; otherwise add a new span of one token, as the CoNLL script does. At an `O`, nothing happens. Replace the `...`. The check calls `to_spans(["B-LOC", "B-LOC", "O", "O", "O", "O", "O", "B-LOC", "I-LOC", "O"])`, the tags of the example, which gives `[(0, 1, 'LOC'), (1, 2, 'LOC'), (7, 9, 'LOC')]`; then cases of its own: two more sentences, the tags of a tagger with an `I-` after an `O`, an `I-` of another type, no names, no tokens, an `I-` at the start, a span of four tokens, and two persons that touch.
        """),
        L.task("Read the tags from left to right, and give back the spans they mark."),
    ])
    return


@app.cell(hide_code=True)
def exercise_spans():
    box_spans = L.code_box("to_spans", lines=11)
    box_spans
    return (box_spans,)


@app.cell(hide_code=True)
def check_spans(box_spans):
    result_spans = L.check("to_spans", box_spans.value)
    result_spans
    return (result_spans,)


@app.cell(hide_code=True)
def hints_spans():
    L.hints(
        "to_spans",
        [
            "One loop over the positions `i`, with `if` and `elif` on the first letter of the tag. In the `I` branch, a second `if` asks three things about the last span, joined with `and`: is there one (`len(spans) > 0`), where does it end (`spans[-1][1]`), and which type does it have (`spans[-1][2]`)?",
            "A span `(start, end, kind)` that ends right before token `i`: which number is its `end`?",
            "`            if len(spans) > 0 and spans[-1][1] == ___ and spans[-1][2] == tag[2:]:`",
        ],
        why="`spans = []`, then `for i in range(len(tags)):` with `tag = tags[i]`; under `if tag[0] == \"B\":` comes `spans.append((i, i + 1, tag[2:]))`, and under `elif tag[0] == \"I\":` comes `if len(spans) > 0 and spans[-1][1] == i and spans[-1][2] == tag[2:]:` with `spans[-1] = (spans[-1][0], i + 1, tag[2:])` under it, and `else:` with `spans.append((i, i + 1, tag[2:]))`; after the loop `return spans`. `to_spans` of the example gives `[(0, 1, 'LOC'), (1, 2, 'LOC'), (7, 9, 'LOC')]`.",
    )
    return


@app.cell(hide_code=True)
def text_after_spans():
    mo.md(r"""
    In the picture, your function reads the tags of all 11,722 sentences: 6,429 names, 2,222 of them more than one token. On the tags of the tagger, *Miramar California* comes back as one place, and *Diego* as a place of its own: the tagger made two mistakes, and the spans show both. A later lesson scores a tagger by counting such spans.
    """)
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Try it: one word, more than one type

    Type a word, and see how often it is part of a person, a place or an organization in Universal NER, with examples. Try *Washington*, *US*, *Kennedy*, *Apple* or *Chicago*.
    """)
    return


@app.cell(hide_code=True)
def demo_controls():
    demo_word = mo.ui.text(value="Washington", label="A word")
    demo_word
    return (demo_word,)


@app.cell(hide_code=True)
def demo_look(demo_word):
    mo.Html(C.demo_picture(demo_word.value, L.lang))
    return


@app.cell(hide_code=True)
def text_exercise_sort():
    mo.md(r"""
    ### Exercise 3. When does IO lose a span?

    Each sentence below comes from Universal NER, with its spans between brackets. Think of the IO tags of each sentence, and read them back: do you get the same spans?
    """)
    return


@app.cell(hide_code=True)
def exercise_sort():
    sort_io = L.sort("io-or-bio")
    sort_io
    return (sort_io,)


@app.cell(hide_code=True)
def check_sort(sort_io):
    result_sort = L.check("io-or-bio", sort_io)
    result_sort
    return (result_sort,)


@app.cell(hide_code=True)
def text_exercise_kinds():
    mo.md(r"""
    ### Exercise 4. One name, more than one type

    In the demo, *Washington* is inside the span of a place 15 times, and inside the span of an organization 5 times, mostly a newspaper such as the *Washington Post*.
    """)
    return


@app.cell(hide_code=True)
def choice_kinds_cell():
    choice_kinds = L.choice(
        "kinds",
        question="What does that mean for a program that has to find names?",
        options={
            "list": "A list of names, each with its type, is enough, as long as it is long",
            "right": "The words around a name have to help decide its type",
            "rare": "Such names are rare, so a program can leave them out",
            "slips": "The people who labeled the sentences made mistakes, since a name always has the same type",
        },
        correct="right",
        why={
            "right": "As with *book* in *Parts of speech and phrases*, the same letters can stand for different things, and the words around them decide: *in Washington* is a place, *the Washington Post* a newspaper. A list of names helps, but it cannot choose the type by itself. Later lessons build programs that learn from the words around a name.",
            "list": "A list would have to give *Washington* one type, and in these sentences it is a place 15 times and part of an organization 5 times. *US* is a place 161 times and an organization 22 times. Which word in *the Washington Post* tells you it is not the city?",
            "rare": "*US* alone is a place 161 times and an organization 22 times, and *Chicago*, *Apple* and *Kennedy* have more than one type too. Try them in the demo: what do the words around them tell you?",
            "slips": "Some labels are slips, such as *George Bush* tagged as two persons. But the city of *Washington* and the newspaper the *Washington Post* are different things that share a word. What tells them apart in a sentence?",
        },
    )
    choice_kinds
    return (choice_kinds,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Parts of speech and phrases](/course/open/nlp-word-classes?lang=en) gave every token a tag: that is sequence labeling. A name can be several tokens long, so the course wrote names as spans `(start, end, kind)`, and BIO tagging turns spans into one label per token: `B-` on the first token, `I-` on the tokens after it, `O` outside. You wrote both ways, and on all 11,722 sentences of Universal NER they gave back what they started with. IO tagging is simpler, but it turns two spans of the same type that touch into one, as in *Miramar California*. The next lesson, *Tagging word classes*, builds a program that gives every word its part of speech, and shows that the most frequent tag of each word already gets most words right.

    **An everyday example.** In a theater, every seat is either taken or free. Two families who sit right next to each other look like one big family, unless each booking marks its first seat.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_kinds, result_sort, result_spans, result_tags):
    L.wrap_up(result_tags, result_spans, result_sort, choice_kinds)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
