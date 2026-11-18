# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson rai-bias-in-embeddings. Route: course/route/lessen/verantwoord.yaml. Checks and the course's run of Granite: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    from everything_ai.lesson import lesson

    L = lesson("rai-bias-in-embeddings", "en")
    C = L.checks   # the word lists of three WEAT tests, and Granite's embedding of every word


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Three short questions about [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en), [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en) and [Bias in language models](/course/open/rai-bias-in-llms?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_cosine():
    quick_cosine = L.choice(
        "quick-cosine",
        question="Two vectors point exactly the same way, and one is twice as long as the other. What is their cosine similarity?",
        options={
            "two": "2, since one of them is twice as long",
            "right": "1: the direction is the same",
            "half": "0.5, the ratio of the two lengths",
        },
        correct="right",
        why={
            "two": "Cosine similarity divides the dot product by both lengths, so the lengths drop out and only the direction counts. See [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en).",
            "right": "Cosine similarity divides the dot product by both lengths, so only the direction counts. This lesson measures how close two words are in this way.",
            "half": "Cosine similarity divides the dot product by both lengths, so the lengths drop out and only the direction counts. See [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en).",
        },
    )
    quick_cosine
    return (quick_cosine,)


@app.cell(hide_code=True)
def quick_check_close():
    quick_close = L.choice(
        "quick-close",
        question="In an embedding, two words get vectors that point almost the same way. What does that say?",
        options={
            "same": "That the two words mean exactly the same",
            "right": "That the model treats them as similar",
            "letters": "That they are spelled almost alike",
        },
        correct="right",
        why={
            "same": "Close is not the same: an embedding puts similar things close together, and similar is not equal. See *The idea* in [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en).",
            "right": "An embedding puts things the model treats as similar close together. For words, a model learns that from how words are used in text: this lesson asks what else it learns that way.",
            "letters": "An embedding is about meaning, not spelling: it puts things the model treats as similar close together. See *The idea* in [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en).",
        },
    )
    quick_close
    return (quick_close,)


@app.cell(hide_code=True)
def quick_check_score():
    quick_score = L.choice(
        "quick-score",
        question="Qwen3-0.6B had a bias score of 0.008 on the course's questions in the form of BBQ. What does that show?",
        options={
            "none": "That Qwen3 has no gender bias and no age bias",
            "right": "How Qwen3 did on those questions, nothing more",
            "small": "That small models have less bias than large ones",
        },
        correct="right",
        why={
            "none": "A low score finds no problem; it does not prove there is none. With the facts, Qwen3 was right more often when they fit the stereotype. See [Bias in language models](/course/open/rai-bias-in-llms?lang=en).",
            "right": "A score holds for the questions, the language and the prompt that were tested. This lesson's test, WEAT, has the same limit, and a few more.",
            "small": "The course tested one small model, so it cannot compare sizes. And a low score finds no problem; it does not prove there is none. See [Bias in language models](/course/open/rai-bias-in-llms?lang=en).",
        },
    )
    quick_score
    return (quick_score,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Before a model can work with words, it turns them into vectors: embeddings. Search by meaning, recommendations and language models all start there. An embedding model learns which words belong together from the texts it reads, and those texts hold the stereotypes of the people who wrote them. Aylin Caliskan and colleagues (2017) showed that a word embedding learned such links: how much closer the embeddings of 50 occupation words lay to female words than to male words went together with the share of women in those occupations in the US: a correlation of 0.90, on a scale where 1 means that the two rise together perfectly and 0 that they have nothing to do with each other. In this lesson you measure how close words lie to each other in Granite, the course's embedding model, with WEAT, a test that does for embeddings what a bias benchmark does for language models. You also take the gender direction out of words, as a well-known method does, and measure what is left.

    **An everyday example.** In a word game, someone says "salary", and you hear "office" or "boss" in your head before you think. What comes up first is what you most often heard together. An embedding model learned the same way, from which words appear together in texts: also that some names more often stand next to "career" and others next to "family".
    """)
    return


@app.cell(hide_code=True)
def predict_glove():
    guess_glove = L.guess(
        "glove",
        question="Caliskan and colleagues found an effect size of 1.81 in GloVe, the word embedding they tested, for men's and women's names with words about career and about family. On a scale from −2 to 2, 0 means no difference. Granite came out in December 2024; the model it was built on learned from text that was filtered to remove hate, abuse and profanity. What do you expect for Granite?",
        options={
            "zero": "Close to 0, since the filter took it out",
            "right": "Still large, above 1",
            "flip": "Below 0, the other way round",
        },
        reveal="**Still large**: 1.49, with each word embedded on its own. The filter took out hate, abuse and profanity; that men's names lie closer to words about career than women's names do is none of those. The rest of the lesson shows how that number is made, and what it does and does not mean.",
    )
    guess_glove
    return (guess_glove,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Word embeddings.** A **word embedding** is an embedding of one word, learned from the words it tends to occur near, so that words used in similar ways get vectors that point in similar directions. GloVe and word2vec give one fixed vector per word. Granite is made for whole texts: it embeds a word as a text of one word, as a vector of 384 numbers, of length 1, so that the dot product of two embeddings is their cosine similarity.

    **WEAT.** Caliskan and colleagues built the **WEAT**, the Word Embedding Association Test, after a test for people, the Implicit Association Test. WEAT takes two lists of **target words** of equal size, X and Y, such as eight men's names and eight women's names, and two lists of **attribute words**, A and B, such as eight words about career and eight about family. First the **association** of one word w: its mean cosine similarity with the words of A, minus that with the words of B.

    $$s(w, A, B) = \text{mean}_{a \in A} \cos(w, a) - \text{mean}_{b \in B} \cos(w, b)$$

    A positive association means: closer to A than to B. Then the effect size: the mean association of X minus that of Y, divided by the standard deviation of the associations of all target words, X and Y together.

    $$d = \frac{\text{mean}_{x \in X}\, s(x, A, B) - \text{mean}_{y \in Y}\, s(y, A, B)}{\text{std of } s(w, A, B) \text{ over all target words}}$$

    At 0, the two lists lie equally close to A and B. A positive d means that X lies closer to A, and Y closer to B. With X and Y of equal size, it cannot be more than 2: the standard deviation is then at least half the gap between the two means. Caliskan and colleagues add that their numbers "do not have the same interpretation as the IAT because the “subjects” in our experiments are words, not people". In GloVe they found 1.81 for men's and women's names with career and family, 1.06 for math and arts words with male and female words, and 1.24 for science and arts words with male and female words. They also give a p-value, from how often the target words, split into two lists at random, give a difference as large.

    **The gender direction.** Tolga Bolukbasi and colleagues (2016) found in word2vec that "man is to computer programmer as woman is to x" gives homemaker. Their finding: gender bias is captured by a direction in the embedding. The differences she − he, her − his, woman − man and seven more pairs point roughly the same way; this lesson takes their mean, scaled to length 1, as the **gender direction** g. Bolukbasi and colleagues compute it another way, with principal components, which a later lesson covers; on Granite, their direction points almost the same way: the cosine similarity of the two is 0.998. Their method, hard debiasing, then has a step **Neutralize**: for every word that should be gender-neutral, such as an occupation, it takes out the part along g and scales the rest back to length 1. With g of length 1, that part is (w · g) g. A second step, Equalize, moves pairs such as grandmother and grandfather so that each neutral word lies equally far from both. Of the top 150 analogies the embedding gave, crowdworkers judged 19% to be stereotyped before, and only 6% afterwards.

    **Hidden, not removed.** Hila Gonen and Yoav Goldberg (2019) looked again: "the actual effect is mostly hiding the bias, not removing it." After hard debiasing, words that had been the most male or female still fell into two groups that matched their old gender for 92.5% of them, against 99.9% before. In their words: "the gender-direction provides a way to measure the gender-association of a word, but does not determine it."

    **What WEAT cannot show.** Chandler May and colleagues (2019) built SEAT, WEAT for models of whole sentences such as BERT: each word goes into sentences such as `This is <word>.`. Their warning: "while SEAT can confirm the existence of bias, negative results do not indicate the model is bias-free." For BERT, with each word on its own, the career and family test gave 0.21, where a simpler embedding, CBoW, gave 1.81. And Kawin Ethayarajh and colleagues (2019) showed that WEAT "systematically overestimates bias", that with one word in each target list the effect size "is necessarily 2 in one direction and −2 in the other", and that the attribute words can be chosen so that almost any word comes out as more male than another. Granite's model card says that the model it was built on learned from text filtered to remove hate, abuse and profanity, and reports no test for social bias.

    **The course's run.** The course embedded every word of this lesson with Granite, each as a text of one word, and each word of the three tests also in the sentence `This is <word>.`. The exercises read those embeddings; with the download below you embed your own words.

    **Try it.** Below, Granite embeds four lists of words of your own, and the demo gives each target word's association and WEAT's effect size. It starts with one word in each list, as Ethayarajh and colleagues did: door and curtain, with masculine and feminine. Then try man and woman as A and B. Or write a test of your own, with several words in each list.
    """)
    return


@app.cell(hide_code=True)
def download_cell():
    get_button = mo.ui.run_button(label="Download Granite Embedding (223 MB)")
    return (get_button,)


@app.cell(hide_code=True)
def demo_download(get_button):
    C.download_or_stop(get_button, {
        "ask": "*Granite Embedding is not on your computer yet. The button downloads it once from Hugging Face, at a fixed version: `ibm-granite/granite-embedding-107m-multilingual`, 223 MB, under the Apache-2.0 license. The rest of the lesson works without it: it reads the embeddings the course recorded.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: a file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: there is not enough room on your disk. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    mo.md("*Granite Embedding is downloaded.*")
    return


@app.cell(hide_code=True)
def words_box_cell():
    x_box = mo.ui.text(value="door", label="Target words X", full_width=True)
    y_box = mo.ui.text(value="curtain", label="Target words Y", full_width=True)
    a_box = mo.ui.text(value="masculine", label="Attribute words A", full_width=True)
    b_box = mo.ui.text(value="feminine", label="Attribute words B", full_width=True)
    measure_button = mo.ui.run_button(label="Measure with Granite")
    mo.vstack([mo.md("*Words separated by commas, at most 12 in each list.*"), x_box, y_box, a_box, b_box, measure_button])
    return a_box, b_box, measure_button, x_box, y_box


@app.cell(hide_code=True)
def demo_measure(a_box, b_box, measure_button, x_box, y_box):
    _boxes = {"X": x_box.value, "Y": y_box.value, "A": a_box.value, "B": b_box.value}
    if not C.downloaded():
        _out = mo.md("*Download Granite Embedding first, with the button above.*")
    elif not measure_button.value:
        _out = mo.md("*Press the button to measure.*")
    elif not C.filled(_boxes):
        _out = mo.md("*Write at least one word in each of the four boxes.*")
    else:
        mo.output.replace(mo.md("*Granite is loading and embedding your words …*"))
        _words = {"title": "Your test on Granite", "xlabel": "association s(word, A, B)",
                  "rows": {"X": "target words X", "Y": "target words Y"},
                  "every": "Each word with its association: {words}.",
                  "effect": "WEAT's effect size: {d}.",
                  "no_spread": "All target words have the same association, so the effect size cannot be worked out: there is nothing to divide by."}
        _out = mo.Html(C.demo(_boxes, _words, L.lang))
    _out
    return


@app.cell(hide_code=True)
def text_idea_after():
    mo.vstack([
        mo.md(r"""
        ### New words and symbols

        | Word | What it means |
        |---|---|
        | **word embedding** | an embedding of one word, learned from the words it tends to occur near |
        | **WEAT** | the Word Embedding Association Test: whether two lists of target words differ in how close they lie to two lists of attribute words, as an effect size from −2 to 2 |
        | **target words, attribute words** | the words WEAT compares (X and Y, such as names), and the words it compares them with (A and B, such as career and family) |
        | **association** s(w, A, B) | the mean cosine similarity of w with A, minus that with B |
        | **gender direction** g | the direction from he to she in an embedding, of length 1 |
        | **Neutralize** | the step of hard debiasing that takes the part along g out of a gender-neutral word: w − (w · g) g, scaled to length 1 |
        """),
        L.watch_out("A WEAT number holds for the word lists you chose, in the way you embedded them. A large number shows an association in the embedding; a small number does not show there is none. And with few words, other lists for the same idea can give another number: choose the lists before you look."),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    Testing an embedding with WEAT:

    1. Choose the target words X and Y and the attribute words A and B, with a reason for each word, before you look at any result.
    2. Embed every word with the model you test, the way it is used: as a word on its own, or in a sentence.
    3. Work out the association of every target word with A and B. That is `association`.
    4. Work out the effect size. That is `effect_size`. Try other lists for the same idea too.
    5. To test a step against bias, such as Neutralize, measure afterwards with more than the measure it was built on: other lists, names, and which words still lie close together.
    6. Write down the lists, the model, how you embedded the words, and that a small number does not mean no bias.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    Four target words and two attribute words, in two dimensions, each of length 1: A has one word (1, 0), B has one word (0, 1). X has x₁ = (0.8, 0.6) and x₂ = (1, 0); Y has y₁ = (0.6, 0.8) and y₂ = (0, 1). With vectors of length 1, the cosine similarity is the dot product.

    | Word | cos(w, a) | cos(w, b) | s(w, A, B) |
    |---|---|---|---|
    | x₁ = (0.8, 0.6) | 0.8 | 0.6 | 0.2 |
    | x₂ = (1, 0) | 1 | 0 | 1 |
    | y₁ = (0.6, 0.8) | 0.6 | 0.8 | −0.2 |
    | y₂ = (0, 1) | 0 | 1 | −1 |

    The cell works out the effect size from the four associations.
    """)
    return


@app.cell
def example_effect():
    _s = [0.2, 1.0, -0.2, -1.0]   # the associations of x1, x2, y1 and y2

    print("mean of X:", np.mean(_s[:2]))
    print("mean of Y:", np.mean(_s[2:]))
    print("standard deviation:", round(np.std(_s), 4))
    print("effect size:", round((np.mean(_s[:2]) - np.mean(_s[2:])) / np.std(_s), 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The means are 0.6 and −0.6, a gap of 1.2; the standard deviation of all four is 0.7211, so the effect size is 1.2 / 0.7211 = 1.66. Now Neutralize, in three dimensions: w = (0.6, 0, 0.8) and g = (0, 0, 1). The part along g is w · g = 0.8 times g, so w − 0.8 g = (0.6, 0, 0). That has length 0.6; scaled back to length 1, it is (1, 0, 0). Along g, the word now lies at 0; in the other directions, it still points where it pointed.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_association():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. association: closer to A or to B

        `association(w, A, B)` gets one embedding `w` and two lists of embeddings, `A` and `B`. The box has `cosine(u, v)` from [Length, distance and cosine similarity](/course/open/math-norms-cosine?lang=en), and the loop that puts the cosine similarity of `w` with each word of `A` in the list `to_a`. Replace the `...` with the same for `B`, in a list `to_b`, and give back the mean of `to_a` minus the mean of `to_b`. The check calls `association(vec["John"], CAREER, FAMILY)`, with `vec["John"]` Granite's embedding of John and `CAREER` and `FAMILY` the embeddings of the eight words about career and about family; then five other cases. The picture under the check then places all sixteen names of the test with your function.
        """),
        L.task("Give back the mean cosine similarity of `w` with `A`, minus that with `B`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_association():
    box_association = L.code_box("association", lines=16)
    box_association
    return (box_association,)


@app.cell(hide_code=True)
def check_association(box_association):
    result_association = L.check("association", box_association.value)
    result_association
    return (result_association,)


@app.cell(hide_code=True)
def hints_association():
    L.hints(
        "association",
        [
            "The loop over `A` is already there. Write the same loop over `B`, then subtract the two means.",
            "`to_b = []`, then `for b in B:` with `to_b.append(cosine(w, b))` inside. After the loop, `np.mean` gives the mean of a list.",
            "`return np.mean(to_a) - ___`",
        ],
        why="`to_b = []`, `for b in B:` with `to_b.append(cosine(w, b))`, and `return np.mean(to_a) - np.mean(to_b)`. For John, that gives 0.0084: a little closer to the career words than to the family words.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_effect():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. effect_size: WEAT's number

        `effect_size(X, Y, A, B)` gets four lists of embeddings. The function already puts the association of every word of `X` in `s_x`, and of every word of `Y` in `s_y`, with the course's own `association`, so this exercise also works when exercise 1 is not done yet. Replace the `...` with one line, as in the worked example: the mean of `s_x` minus the mean of `s_y`, divided by the standard deviation of all associations. `s_x + s_y` puts the two lists one after the other, in one list. The check calls `effect_size(MEN, WOMEN, CAREER, FAMILY)`, then the lists of the two other tests, the same test with the names swapped, and the worked example.
        """),
        L.task("Give back WEAT's effect size of `X` and `Y` with `A` and `B`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_effect():
    box_effect = L.code_box("effect", lines=14)
    box_effect
    return (box_effect,)


@app.cell(hide_code=True)
def check_effect(box_effect):
    result_effect = L.check("effect", box_effect.value)
    result_effect
    return (result_effect,)


@app.cell(hide_code=True)
def hints_effect():
    L.hints(
        "effect",
        [
            "Three parts, as in the worked example: the mean of `s_x`, the mean of `s_y`, and the standard deviation of both lists together.",
            "`np.mean` gives the mean of a list and `np.std` its standard deviation; `s_x + s_y` is one list with both. Put the gap between the two means in brackets before you divide, or Python divides only the second mean.",
            "`return (np.mean(s_x) - np.mean(s_y)) / ___`",
        ],
        why="`return (np.mean(s_x) - np.mean(s_y)) / np.std(s_x + s_y)`. For the names with career and family, that gives 1.49 on Granite.",
    )
    return


@app.cell(hide_code=True)
def text_granite():
    mo.md(r"""
    ### What Granite's embeddings show

    With each word embedded on its own, Granite gives 1.49 for men's and women's names with career and family, 1.36 for math and arts words with male and female words, and 0.96 for science and arts words with male and female words. In GloVe, Caliskan and colleagues found 1.81, 1.06 and 1.24. In the sentence `This is <word>.`, Granite gives 1.28, 1.30 and 0.87: lower, but still the same way round. Every number rests on eight words a list, and on these eight; other names or other career words could give other numbers.

    Ethayarajh and colleagues' example, measured on Granite: door against curtain, with one attribute word a side. With masculine and feminine, the effect size is 2.00: door is the more male word. With man and woman, it is −2.00: now curtain is. The associations behind those numbers are door −0.0117 and curtain −0.0128 with masculine and feminine, and door −0.0043 and curtain −0.0016 with man and woman.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_door():
    mo.md(r"""
    ### Exercise 3. A door and a curtain
    """)
    return


@app.cell(hide_code=True)
def choice_door_cell():
    choice_door = L.choice(
        "door",
        question="On Granite, with masculine and feminine as attribute words, door against curtain gives an effect size of 2.00. With man and woman, it gives −2.00. What does that show?",
        options={
            "doors": "That Granite links doors with men",
            "right": "One word a side always gives 2 or −2",
            "broken": "That Granite's embeddings of single words are broken",
            "large": "A large effect, since 2 is the most WEAT can give",
        },
        correct="right",
        why={
            "doors": "With man and woman, curtain is the more male word. With one word a side, the words you choose decide which side wins, and the effect size is 2 or −2 either way. See *What Granite's embeddings show*.",
            "right": "The associations differ by about 0.001 with masculine and feminine, and by about 0.003 with man and woman. With one word in X and one in Y, the standard deviation is exactly half the gap, so the effect size is 2 or −2 however small the gap is. In Ethayarajh and colleagues' words: \"by contriving the attribute sets for WEAT, virtually any word can be classified as gender-biased relative to another.\"",
            "broken": "Nothing is broken: the two associations differ by about 0.001 and 0.003, and with one word a side, any gap gives 2 or −2. See *What Granite's embeddings show*.",
            "large": "2 is the most WEAT can give, and here it comes from gaps of about 0.001 and 0.003. With one word a side, the effect size is always 2 or −2, so it says nothing about how large the difference is.",
        },
    )
    choice_door
    return (choice_door,)


@app.cell(hide_code=True)
def text_exercise_neutralize():
    mo.vstack([
        mo.md(r"""
        ### Exercise 4. neutralize: take the gender direction out

        `neutralize(w, g)` gets one embedding `w` and the gender direction `g`, of length 1. The first line works out `along = np.dot(w, g)`, how far `w` lies along `g`. Replace the `...` with two lines, as in the worked example: take `along * g` out of `w`, and give `w` back scaled to length 1, divided by `np.linalg.norm(w)`. The check calls `neutralize(vec["nurse"], g)`, with `g` the gender direction of the ten pairs on Granite, and four other cases. The picture under the check then passes every gender-neutral word of the lesson through your function, and measures WEAT again.
        """),
        L.task("Give back `w` without its part along `g`, scaled to length 1."),
    ])
    return


@app.cell(hide_code=True)
def exercise_neutralize():
    box_neutralize = L.code_box("neutralize", lines=10)
    box_neutralize
    return (box_neutralize,)


@app.cell(hide_code=True)
def check_neutralize(box_neutralize):
    result_neutralize = L.check("neutralize", box_neutralize.value)
    result_neutralize
    return (result_neutralize,)


@app.cell(hide_code=True)
def hints_neutralize():
    L.hints(
        "neutralize",
        [
            "Two steps, as in the worked example: subtract the part along `g`, then scale to length 1.",
            "The part along `g` is `along * g`, a vector. `np.linalg.norm(w)` is the length of `w`; dividing by it gives length 1.",
            "`w = w - ___` and then `return w / np.linalg.norm(w)`.",
        ],
        why="`w = w - along * g` and `return w / np.linalg.norm(w)`. For w = (0.6, 0, 0.8) and g = (0, 0, 1), that gives (1, 0, 0).",
    )
    return


@app.cell(hide_code=True)
def text_after_neutralize():
    mo.md(r"""
    ### What Neutralize changed

    The course passed every gender-neutral word of the lesson through Neutralize: the words about career, family, math, science and the arts, and twenty occupations, the ten that Bolukbasi and colleagues found closest to *she* and the ten closest to *he* in word2vec. The men's and women's names, and words such as *he* or *sister*, stay as they are, since they are about gender. Afterwards, every neutral word lies at 0 along g. Math and arts with male and female words drops from 1.36 to −0.33, and science and arts from 0.96 to 0.13. But names with career and family only drops from 1.49 to 1.15: the names were not touched. The ten occupations on the *she* side against the ten on the *he* side, with male and female words, go from 1.11 to 0.96. And 15 of the 20 occupations still have, as their nearest occupation, one from their own side of the list, against 16 before.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_hidden():
    mo.md(r"""
    ### Exercise 5. Free of gender bias?
    """)
    return


@app.cell(hide_code=True)
def choice_hidden_cell():
    choice_hidden = L.choice(
        "hidden",
        question="A team passes every gender-neutral word of its embedding through Neutralize. WEAT with math and arts words then drops from 1.36 to −0.33, and the team writes: \"Our embedding is free of gender bias now.\" What would Gonen and Goldberg say?",
        options={
            "right": "That it hides the bias more than it removes it",
            "done": "That it works: −0.33 is close enough to 0",
            "worse": "That it made the bias worse, now against men",
            "more": "That they should take out two directions instead of one",
        },
        correct="right",
        why={
            "right": "In their words, \"the actual effect is mostly hiding the bias, not removing it\". On Granite too: names with career and family still give 1.15, the occupations still differ by 0.96 with male and female words, and 15 of the 20 occupations keep a nearest occupation from their old side. The direction measures the gender association of a word; it does not decide it.",
            "done": "On this one test, yes. But the same male and female words still separate the occupations, 1.11 → 0.96, names with career and family still give 1.15, and 15 of the 20 occupations keep a nearest occupation from their old side. See *What Neutralize changed*.",
            "worse": "A number below 0 on one test does not show the bias turned around. The same male and female words still separate the occupations, 1.11 → 0.96, and names with career and family still give 1.15. See *What Neutralize changed*.",
            "more": "Gonen and Goldberg's point is that the direction measures the gender association of a word but does not decide it: the words still group with the words of their old gender. On Granite, 15 of the 20 occupations keep a nearest occupation from their old side. See *What Neutralize changed*.",
        },
    )
    choice_hidden
    return (choice_hidden,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** [Embeddings: meaning as coordinates](/course/open/ml-embeddings-idea?lang=en) put similar things close together; here you saw that close can also mean: often named together in texts, stereotypes included. [Bias in language models](/course/open/rai-bias-in-llms?lang=en) tested what a model answers; WEAT tests what lies inside it, and has the same limit: a small number does not show there is no bias. The next lesson, *Bias in face analysis and image generators*, looks at models that see and make pictures.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_door, choice_hidden, result_association, result_effect, result_neutralize):
    L.wrap_up(result_association, result_effect, choice_door, result_neutralize, choice_hidden)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
