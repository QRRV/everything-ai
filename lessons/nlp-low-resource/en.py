# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-low-resource. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-low-resource", "en")
    C = L.checks   # the course's sentences in seven languages, the counts of NTREX-128, the tokenizers and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en) and [About 7,000 languages](/course/open/nlp-languages-of-the-world?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_tokens():
    quick_tokens = L.choice(
        "quick-tokens",
        question="GPT-2 cuts the English sentence *Tokenization is surprisingly important.* into 6 tokens, and the same sentence in Dutch, *Tokenisatie is verrassend belangrijk.*, into 15. Why the difference?",
        options={
            "longer": "the Dutch sentence has more words",
            "letters": "GPT-2 cuts every word of a language other than English into single letters",
            "right": "GPT-2 made its list of tokens from text that was almost all English",
            "random": "a tokenizer cuts at random places, so the count changes from run to run",
        },
        correct="right",
        why={
            "right": "GPT-2's tokens are the pieces that came together often in its training text, and that text was almost all English. So English words are often one token, and Dutch words fall into pieces: *verrassend* became four. This lesson measures how much more other languages cost, and what that means for languages with even less text.",
            "longer": "Both sentences have four words. What did GPT-2 learn its tokens from? See [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en).",
            "letters": "*verrassend* became four pieces, not ten letters, and *is* stayed whole. Where did GPT-2 get its pieces from? See [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en).",
            "random": "A tokenizer cuts the same text the same way every time, and `decode` gives it back exactly. What did GPT-2 learn its tokens from? See [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en).",
        },
    )
    quick_tokens
    return (quick_tokens,)


@app.cell(hide_code=True)
def quick_check_classes():
    quick_classes = L.choice(
        "quick-classes",
        question="Pratik Joshi and four others sorted 2,485 languages into six classes, by the resources that exist for them. How many of those languages are in the bottom class, *The Left-Behinds*?",
        options={
            "half": "about half of them",
            "seven": "7, as many as in the top class",
            "right": "2,191, almost nine in ten",
            "few": "a few hundred, because most languages have some text on the web",
        },
        correct="right",
        why={
            "right": "2,191 of the 2,485 languages are in class 0, with almost no text in digital form, labeled or not, and 7 are in class 5, *The Winners*. Dutch is in class 4. This lesson is about what a tokenizer and a model do with the languages lower down, and how such a language can still get a working model.",
            "half": "Far more than half: the classes are very uneven. How many languages did the bottom class hold? See [About 7,000 languages](/course/open/nlp-languages-of-the-world?lang=en).",
            "seven": "7 is the top class, *The Winners*, with English, Spanish, German, Japanese and French. How many are at the bottom? See [About 7,000 languages](/course/open/nlp-languages-of-the-world?lang=en).",
            "few": "Most languages have almost no text on the web. How many of the 2,485 are in the bottom class? See [About 7,000 languages](/course/open/nlp-languages-of-the-world?lang=en).",
        },
    )
    quick_classes
    return (quick_classes,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Joshi and others put English, Spanish, German, Japanese and French in their top class: "They are the quintessential rich-resource languages, reaping benefit from each state-of-the-art NLP breakthrough." Dutch is one class lower, in class 4, *The Underdogs*: "They have a large amount of unlabeled data, comparable to those possessed by the winners, and are only challenged by lesser amount of labeled data." Swahili, Zulu and Yoruba, each with millions of speakers, are in class 2.

    A model learns from the text it gets. Its tokenizer learns its pieces from that text too, and when the text is mostly English, every other language costs more tokens for the same message. A service that charges per token then charges more for it, the model fits less of it in its context, and it writes the answer more slowly, one token at a time. Orevaoghene Ahia and six others measured this for the API of OpenAI in 2023: "We show evidence that speakers of a large number of the supported languages are overcharged while obtaining poorer results." This lesson measures what a language costs a tokenizer, and looks at what helps a language with little data get a model that works.

    **An everyday example.** A text message on a phone holds 160 characters of the Latin alphabet. Put one Russian letter in it, and it holds only 70: the phone then stores every character in more room. The same news costs more messages in Russian than in English.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_tamil = L.guess(
        "guess",
        question="NTREX-128 is a set of 1,997 sentences of news, translated from English into 128 languages by professional translators. GPT-2's tokenizer makes 2,741 tokens of the first 100 English sentences. How many tokens does it make of the same 100 sentences in Tamil, a language of the south of India, Sri Lanka and Singapore?",
        options={
            "same": "about 2,700: the same news costs the same in every language",
            "dutch": "about 5,400, twice as many, like Dutch",
            "five": "about 14,000, five times as many",
            "right": "about 39,000, fourteen times as many",
        },
        reveal="**39,232**, 14.3 times as many as the English sentences. A Tamil letter is three bytes in UTF-8, and GPT-2 learned almost no pieces of Tamil: it makes 0.99 tokens of every byte. The tokenizer of Qwen3, whose models of 2025 learned from 119 languages and dialects, makes 15,772 tokens of the Tamil sentences: 5.7 times its 2,770 tokens of the English ones. Dutch costs GPT-2 5,414 tokens, 2.0 times as many, and Qwen3 4,419, 1.6 times.",
    )
    guess_tamil
    return (guess_tamil,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Low-resource language.** A language with little text in digital form, and even fewer labeled examples for a task, such as the names in a text. The line is not sharp. Joshi and others counted, for each language, the labeled data sets in two catalogs and its pages on Wikipedia, and drew their six classes from those counts. A language can have many speakers and still little data: class 0 holds 88% of the languages they sorted and 15% of all the speakers.

    **A tokenizer learns from its text.** GPT-2's tokenizer starts from the 256 possible values of a byte, so it can write any text, and makes tokens of pieces that came together often in its training text, as you saw in [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en). That text was almost all English. So English words are often one token, Dutch words two or three, and the letters of a language it rarely saw stay apart, or even in bytes. In UTF-8, a letter of the Latin alphabet without an accent is one byte, a Greek or Russian letter two bytes, and a letter of Tamil, Hindi or Thai, or a Chinese character, three.

    **The premium.** Aleksandar Petrov, Emanuele La Malfa, Philip Torr and Adel Bibi compared languages on the same text, translated: the **premium** of a language is its number of tokens divided by the number of tokens of the same sentences in English. A premium of 1 is **parity**. "Therefore, we make the case for multilingual tokenization parity: tokenizers should produce similar encoded lengths for the same content across languages." On 2,000 sentences from Wikipedia, translated into 200 languages, they found for GPT-2 a premium of 1.97 for Dutch. Other languages cost far more: with the tokenizer of ChatGPT and GPT-4, Shan, a language of Myanmar, costs 15 times as many tokens as English. "The same text translated into different languages can have drastically different tokenization lengths, with differences up to 15 times in some cases."

    **Tokens per byte.** The number of tokens divided by the number of bytes of the text in UTF-8. A token of GPT-2 holds at least one byte, so this is never more than 1, and close to 1 means that the tokenizer learned almost no pieces of the language. For English, GPT-2 makes 0.22 tokens of a byte: a token holds 4.6 bytes on average. Unlike the premium, tokens per byte needs no translation: one text is enough.

    **Tokenizer fertility and continued words.** Phillip Rust and four others measure a tokenizer on the words of a language, with its **tokenizer fertility**: "First, the subword fertility measures the average number of subwords produced per tokenized word. A minimum fertility of 1 means that the tokenizer’s vocabulary contains every single word in the text." That is your `tokens_per_word` of *Tokens*. Their second measure counts the **continued words**, the words that the tokenizer cuts into more than one piece: "Whereas the fertility is concerned with how aggressively a tokenizer splits, this metric measures how often it splits words." Both need words, and `split` finds words only where a language puts spaces between them. Chinese, Japanese, Thai, Lao, Khmer and Tibetan write their words without. Ahia and colleagues give this as a reason to measure tokens per sentence instead: "many languages we analyze are understudied and do not have word tokenizers available which are required to compute fertility."

    **What the tokenizer does to the model.** More tokens is more than a higher bill. Rust and colleagues compared models for one language with a multilingual model: "We find that while the pretraining data size is an important factor, a designated monolingual tokenizer plays an equally important role in the downstream performance." And: "We further find that replacing the original multilingual tokenizer with the specialized monolingual tokenizer improves the downstream performance of the multilingual model for almost every task and language."

    **The same 100 sentences in 48 languages.** The picture shows the premium of 47 languages against English, on the first 100 sentences of NTREX-128, for two tokenizers: GPT-2's, of 2019, with 50,257 tokens, and Qwen3's, with 151,669. Qwen3 uses the tokenizer that Qwen made in 2023 for its first models, and the Qwen3 models of 2025 learned from text in 119 languages and dialects. GPT-2's premium for Dutch is 1.98, close to the 1.97 of Petrov and colleagues on other sentences. Spanish, French, Dutch and German cost GPT-2 about twice as many tokens as English, and so do Samoan and Maori, in class 1, written in the Latin alphabet. Russian costs 5.55, Arabic 4.48, both in the top classes: the premium follows the script of a language more than its class. Qwen3 lowers every premium: Chinese from 3.21 to 1.11, almost parity, and Russian from 5.55 to 1.88. And Telugu, Tibetan, Tamil, Khmer, Lao and Armenian still cost Qwen3 more than five times as many tokens as English, and Bengali five times.

    **Cross-lingual transfer.** A language with few labeled examples can borrow them. A **multilingual model** learned from text in many languages at once: Multilingual BERT, mBERT, learned from Wikipedia in 104 languages, with one list of pieces for all of them. Telmo Pires, Eva Schlinger and Dan Garrette fine-tuned it in one language and tested it in another: "we fine-tune the model using task-specific supervised training data from one language, and evaluate that task in a different language". That is **cross-lingual transfer**: zero-shot, when the model sees no labeled example of the language it is tested on. Fine-tuned on the names in English news of CoNLL 2003, mBERT found the names in Dutch news with a span F1 of 0.774; fine-tuned on Dutch news, 0.899. Transfer works best between similar languages. For parts of speech, mBERT reached 81.55% accuracy on average from one language with the order subject, verb, object to another one with that order, and 66.52% to a language with the verb at the end.

    **A few examples of its own.** Anne Lauscher and three others added a few labeled sentences of the language itself after the fine-tuning in English: "For instance, we improve dependency parsing for some target languages up to 40 UAS points with as few as 10 target language sentences." UAS is the share of the words that get the right head in a dependency tree, in points out of 100. On average over their languages, 500 sentences of the language itself lifted mBERT's UAS from 52.96 to 80.47, and its accuracy on parts of speech from 67.2 to 91.16. For tasks about the meaning of a whole sentence, such as whether one sentence follows from another, the gain was small: there, 100 examples even lowered the score, by 1.00 point. And a model of its own on the little text there is does not help: Shijie Wu and Mark Dredze found that "mBERT does better than or comparable to baselines on high resource languages but does much worse for low resource languages. Furthermore, monolingual BERT models for these languages do even worse." So people build labeled data where it is missing. In MasakhaNER, a project of 61 researchers, people marked the names in texts in ten African languages, among them Swahili, Wolof and Yoruba, as "the first large, publicly available, high-quality dataset for named entity recognition (NER) in ten African languages."
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
    | **low-resource language** | a language with little text in digital form, and few labeled examples for tasks |
    | **premium** | the tokens of a text in a language divided by the tokens of the same text in English |
    | **parity** | a premium of about 1: the same message costs the same number of tokens in both languages |
    | **tokens per byte** | the tokens of a text divided by its bytes in UTF-8; close to 1, the tokenizer learned almost nothing of the language |
    | **tokenizer fertility** | the average number of tokens per word of a language; *fertility* for short |
    | **continued words** | the share of the words that the tokenizer cuts into more than one token |
    | **multilingual model** | a model that learned from text in many languages at once, such as mBERT |
    | **cross-lingual transfer** | fine-tuning a multilingual model in one language and using it in another |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To measure what a language costs a tokenizer:

    1. Take sentences that say the same in the language and in English: translations, sentence by sentence.
    2. Count the tokens of each sentence and add them up. The premium is the tokens of the language divided by the tokens of English.
    3. Count the bytes of each sentence in UTF-8. Tokens per byte says how much of the language the tokenizer learned.
    4. Where the language puts spaces between its words: count the words that the tokenizer cuts into more than one token, each word with a space before it, as it stands in a sentence.
    5. Do the same with another tokenizer, on the same sentences, and compare.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.OWN` holds the course's three sentences in seven languages: `C.OWN["nl"][0]` is the Dutch sentence about the library, and `C.OWN["el"][0]` the same sentence in Greek. `C.ENC` is GPT-2's tokenizer. Its `encode` gives the tokens of a text, and `C.ENC.decode_single_token_bytes(t)` the bytes of token `t`. For NTREX-128, `C.counts()` gives the course's counts of the first 100 sentences of each language: the words, the characters, the bytes, the tokens of GPT-2 and of Qwen3, and the continued words of both, by the code of the language in NTREX, such as `"tam"` for Tamil. `C.course_premium(code, tokenizer)`, `C.course_per_byte`, `C.course_continued` and `C.course_fertility` work out the measures from those counts:
    """)
    return


@app.cell
def example_cell():
    _sentence = C.OWN["nl"][0]
    _tokens = C.ENC.encode(_sentence)
    print(len(_tokens), [C.ENC.decode_single_token_bytes(_t) for _t in _tokens])
    print(len(_sentence), len(_sentence.encode("utf-8")))
    _greek = C.OWN["el"][0]
    print(len(_greek), len(_greek.encode("utf-8")), len(C.ENC.encode(_greek)))
    for _code in ("nld", "tam", "zho-CN"):
        print(_code, C.counts()[_code])
        print(round(C.course_premium(_code, "gpt2"), 2), round(C.course_premium(_code, "qwen3"), 2),
              round(C.course_per_byte(_code, "gpt2"), 3), round(C.course_continued(_code, "gpt2"), 3),
              round(C.course_fertility(_code, "gpt2"), 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    The Dutch sentence is 18 tokens. *De* and *om* are whole, *open* too, but *bibliotheek* falls into four pieces, *b*, *ibli*, *othe* and *ek*, and *zaterdag* into four as well. It has 47 characters and 47 bytes: every letter is one byte. The Greek sentence has 43 characters and 79 bytes, nearly two per character, and GPT-2 makes 47 tokens of it: more tokens than characters. 20 of those 47 tokens hold half a letter, one of its two bytes; three of them hold a space too.

    Then NTREX-128. The 100 Dutch sentences have 2,391 words and 15,200 bytes, and GPT-2 makes 5,414 tokens of them: a premium of 1.98, 0.356 tokens per byte, 64% of the words cut, and 2.26 tokens per word. Tamil: 39,232 tokens of 39,678 bytes, 0.989 per byte, a premium of 14.31 for GPT-2 and 5.69 for Qwen3; GPT-2 cuts 97% of the words, into 24.22 tokens per word on average. Chinese costs GPT-2 3.21 times as much as English, and Qwen3 1.11: almost parity. Its 27.15 tokens per "word" say nothing about the Chinese language: `split` found 324 words, only where the text has a space, around words in Latin letters, such as *AM*.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_premium():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. premium: the price of a language

        Write `premium`, which gets `texts`, a list of sentences in a language, `english`, the same sentences in English, and `enc`, a tokenizer. It gives back the premium: the tokens of all of `texts` divided by the tokens of all of `english`. In the start code, `tokens` counts the tokens of `texts`, with `for text in texts`, and `english_tokens` the tokens of `english`, with `for text in english`. The last line divides the two. Replace the two `...` with the number of tokens of `text`.

        The check calls `premium(["De trein naar Utrecht vertrekt over tien minuten."], ["The train to Utrecht leaves in ten minutes."], enc)` with GPT-2's tokenizer as `enc`, which gives `1.5454545454545454`: 17 tokens for the Dutch sentence, 11 for the English one. Then cases of its own: the course's three sentences in several languages, English against itself, and English against Russian.
        """),
        L.task("Add the number of tokens of every `text` to its count, in both loops."),
    ])
    return


@app.cell(hide_code=True)
def exercise_premium():
    box_premium = L.code_box("premium", lines=8)
    box_premium
    return (box_premium,)


@app.cell(hide_code=True)
def check_premium(box_premium):
    result_premium = L.check("premium", box_premium.value)
    result_premium
    return (result_premium,)


@app.cell(hide_code=True)
def hints_premium():
    L.hints(
        "premium",
        [
            "Each loop makes its count, `tokens` or `english_tokens`, grow by the number of tokens of `text`: the length, with `len`, of the list of tokens that `enc` gives for it.",
            "`enc.encode(\"Hello!\")` gives the tokens of *Hello!*, `[15496, 0]`, and `len` of that list gives `2`.",
            "`        tokens = tokens + len(___)`",
        ],
        why="`tokens = tokens + len(enc.encode(text))`, and in the second loop `english_tokens = english_tokens + len(enc.encode(text))`: each sentence adds its tokens, so a long sentence weighs more than a short one, as in the counts of NTREX.",
    )
    return


@app.cell(hide_code=True)
def text_after_premium():
    mo.md(r"""
    In the picture, your function works out the premium of the course's three sentences in seven languages. French costs 1.77 times as many tokens as English, Dutch 1.83 and German 2.07, all in the Latin alphabet. Japanese costs 3.17, Russian 5.67 and Greek 6.00. Three sentences make a rough measure: on the 100 sentences of NTREX-128, Dutch costs 1.98, Russian 5.55 and Greek 6.19. And English against English gives exactly 1: the premium compares a language with English, so English is parity by definition.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_per_byte():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. per_byte: what did the tokenizer learn of a language?

        Write `per_byte`, which gets `texts`, a list of sentences, and `enc`, a tokenizer, and gives back the tokens of all the sentences divided by their bytes in UTF-8. In the start code, `tokens` and `n_bytes` start at 0, and `for text in texts` goes through the sentences. Inside, `tokens = tokens + len(enc.encode(text))` counts the tokens, as in exercise 1. Replace the `...` with the number of bytes of `text` in UTF-8.

        The check calls `per_byte(["Το τρένο για την Ουτρέχτη φεύγει σε δέκα λεπτά."], enc)` with GPT-2's tokenizer, which gives `0.6470588235294118`: 55 tokens of 85 bytes. Then cases of its own: the course's three sentences in each of the seven languages, and a text with accents.
        """),
        L.task("Add the number of bytes of `text` in UTF-8 to `n_bytes`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_per_byte():
    box_per_byte = L.code_box("per_byte", lines=7)
    box_per_byte
    return (box_per_byte,)


@app.cell(hide_code=True)
def check_per_byte(box_per_byte):
    result_per_byte = L.check("per_byte", box_per_byte.value)
    result_per_byte
    return (result_per_byte,)


@app.cell(hide_code=True)
def hints_per_byte():
    L.hints(
        "per_byte",
        [
            "`n_bytes` grows by the number of bytes of `text`: the length, with `len`, of the bytes that a text gives with its method `encode`, in UTF-8.",
            "`\"één\".encode(\"utf-8\")` gives the bytes of *één*: two for each *é* and one for the *n*, so `len` of it gives `5`, where `len(\"één\")` counts 3 characters.",
            "`        n_bytes = n_bytes + len(___)`",
        ],
        why="`n_bytes = n_bytes + len(text.encode(\"utf-8\"))`: `len(text)` counts characters, and a Greek letter is one character but two bytes. `text.encode()` without an argument gives the same, because UTF-8 is what Python uses unless you say otherwise.",
    )
    return


@app.cell(hide_code=True)
def text_after_per_byte():
    mo.md(r"""
    In the picture, your function on the course's sentences: English 0.222 tokens per byte, French 0.356, Dutch 0.357, German 0.392, Japanese 0.528, Russian 0.603 and Greek 0.604. The Greek sentence about the library shows why: a dashed token holds part of a letter, and many of its other tokens are a single letter. On the 100 sentences of NTREX-128, GPT-2 makes 0.989 tokens per byte of Tamil, 0.992 of Telugu and 0.985 of Armenian: almost no pieces of those languages at all. Qwen3 makes 0.397 of Tamil.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_continued():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. continued: how often does the tokenizer cut a word?

        Write `continued`, which gets `texts`, a list of sentences, and `enc`, a tokenizer, and gives back the share of the words that the tokenizer cuts into more than one token. In the start code, `words` counts all the words and `count` the words that are cut. `for text in texts` goes through the sentences, and `for word in text.split()` through the words of each one. Inside, `words = words + 1` counts the word, and `count = count + 1` runs when the test after `if` holds. Replace the `...` with the test: does the tokenizer cut `word` into more than one token? Give the word a space before it, as it stands in a sentence: GPT-2 has other tokens for a word at the start of a text.

        The check calls `continued(["De trein naar Utrecht vertrekt over tien minuten."], enc)` with GPT-2's tokenizer, which gives `0.75`: six of the eight words are cut. Then cases of its own: the course's three sentences in each of the seven languages, and short words.
        """),
        L.task("Test whether `word`, with a space before it, becomes more than one token."),
    ])
    return


@app.cell(hide_code=True)
def exercise_continued():
    box_continued = L.code_box("continued", lines=9)
    box_continued
    return (box_continued,)


@app.cell(hide_code=True)
def check_continued(box_continued):
    result_continued = L.check("continued", box_continued.value)
    result_continued
    return (result_continued,)


@app.cell(hide_code=True)
def hints_continued():
    L.hints(
        "continued",
        [
            "Encode `word` with a space glued before it, and test whether `len` of the list of tokens that `enc` gives with `encode` is more than 1.",
            "`+` glues two texts together: `\"a\" + \"b\"` gives `\"ab\"`. And the space matters: `enc.encode(\"leaves\")` gives two tokens, *le* and *aves*, while `enc.encode(\" leaves\")` gives one.",
            "`            if len(enc.encode(___)) > 1:`",
        ],
        why="`if len(enc.encode(\" \" + word)) > 1:`: in a sentence, every word but the first has a space before it, and GPT-2 learned most words with that space. A word of exactly one token is whole; from two tokens on, it is cut.",
    )
    return


@app.cell(hide_code=True)
def text_after_continued():
    mo.md(r"""
    In the picture, your function on the course's sentences: GPT-2 cuts 16% of the English words, 62% of the French, 70% of the Dutch and 75% of the German, and every Russian and Greek word. Japanese also gets 100%, but that says nothing: `split` finds no space in a Japanese sentence, so each sentence is one "word". Under the table, the words of the Dutch sentence about the library, with a dot between their pieces. On NTREX-128, Qwen3 cuts 44% of the Dutch words, against GPT-2's 64%.

    ### Try it: one sentence in 48 languages

    The demo needs the first 100 sentences of NTREX-128 in 48 languages and Qwen3's tokenizer. Choose a language and a tokenizer: the picture cuts the first sentence of NTREX-128 into its tokens, in English and in that language, and the table gives the measures of this lesson over the first 100 sentences, for both tokenizers.

    Start with Tamil. Then try Vietnamese with both tokenizers, and look at the continued words; then Japanese, and Yoruba, which is written in the Latin alphabet, with accents.
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label=f"Download NTREX-128 and Qwen3's tokenizer, {C.download_size(C.FILES, L.lang)}")
    return (get_button,)


@app.cell(hide_code=True)
def ntrex_cell(get_button):
    try_languages = C.files_or_stop(get_button, {
        "ask": f"*The sentences of NTREX-128 and the tokenizer of Qwen3 are not on your computer yet. The button downloads them once, {C.download_size(C.FILES, L.lang)}: the sentences from the repository of NTREX on GitHub, under the license CC BY-SA 4.0, and the tokenizer from Hugging Face, under the license Apache-2.0. The rest of the lesson works without them.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: GitHub or Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: a file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: your disk has no room for it. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    }, L.lang)
    try_language = mo.ui.dropdown(options=try_languages, value=C.ntrex.name(C.TRY_START, L.lang), label="Language")
    try_tokenizer = mo.ui.radio(options={"GPT-2": "gpt2", "Qwen3": "qwen3"}, value="GPT-2", label="Tokenizer", inline=True)
    mo.hstack([try_language, try_tokenizer], justify="start", gap=1.5, wrap=True)
    return try_language, try_tokenizer


@app.cell(hide_code=True)
def try_look(try_language, try_tokenizer):
    mo.Html(C.try_picture(L.lang, try_language.value, try_tokenizer.value))
    return


@app.cell(hide_code=True)
def text_after_try():
    mo.md(r"""
    Tamil, with GPT-2: almost every token of the sentence is a dashed one, one byte of a letter. With Qwen3, many tokens are whole letters, or several, and the premium drops from 14.31 to 5.69. Vietnamese is written in the Latin alphabet, with many accents, one syllable at a time, with spaces between the syllables. Vietnamese costs GPT-2 4.55 times as much as English, and GPT-2 cuts 93% of the "words"; for Qwen3 it costs 1.45 times, and Qwen3 cuts 19%, about as often as it cuts English words. In the 100 Japanese sentences, `split` finds 125 "words", only where a space stands, so GPT-2's 57 tokens per word say nothing about Japanese; the premium, 2.62 for GPT-2 and 1.33 for Qwen3, compares fairly. Yoruba marks its tones with accents above the letters, and a few sounds with a dot below: an accented letter is two bytes or more, and GPT-2's premium is 5.38. Qwen3's is 3.24, more than for any other language in the Latin alphabet in the table.

    ### Exercise 4. Helps or falls short?

    A language has little text and almost no labeled data. Sort each way below: does it help such a language get a model that works, or does it fall short?
    """)
    return


@app.cell(hide_code=True)
def exercise_helps():
    sort_helps = L.sort("helps")
    sort_helps
    return (sort_helps,)


@app.cell(hide_code=True)
def check_helps(sort_helps):
    result_helps = L.check("helps", sort_helps)
    result_helps
    return (result_helps,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** The tokens are those of [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en), and the classes of languages those of [About 7,000 languages](/course/open/nlp-languages-of-the-world?lang=en). Cross-lingual transfer is the fine-tuning of [Fine-tuning an encoder for a task](/course/open/nlp-fine-tune-bert?lang=en), in one language for another. You measured the premium, tokens per byte and the continued words, and saw that a tokenizer for many languages lowers the price, but not to parity. The next lesson, *Language models for Dutch*, is about models made for one class 4 language: Dutch.

    **An everyday example.** A traveler in a country without a phrasebook of its own picks up the phrasebook of a neighboring language: much of it works, close enough. A few phrases learned from people there do the rest.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_premium, result_per_byte, result_continued, result_helps):
    L.wrap_up(result_premium, result_per_byte, result_continued, result_helps)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
