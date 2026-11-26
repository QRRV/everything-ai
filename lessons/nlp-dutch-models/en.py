# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Lesson nlp-dutch-models. Route: course/route/lessen/taal-en-zoeken.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("nlp-dutch-models", "en")
    C = L.checks   # the course's sentences, the tokenizers of GPT-2 and RobBERT, the counts of NTREX-128 and the checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Quick check

    Two questions about [Languages with little data](/course/open/nlp-low-resource?lang=en) and [BERT: filling in the blank](/course/open/nlp-bert?lang=en).
    """)
    return


@app.cell(hide_code=True)
def quick_check_premium():
    quick_premium = L.choice(
        "quick-premium",
        question="On the first 100 sentences of NTREX-128, GPT-2's tokenizer makes 5,414 tokens of the Dutch sentences and 2,741 of the English ones. What is the premium of Dutch for GPT-2?",
        options={
            "upside": "about 0.51, English divided by Dutch",
            "right": "about 1.98: Dutch costs almost twice as many tokens",
            "difference": "2,673, the extra tokens of Dutch",
            "both": "8,155, because a premium adds up the tokens of both languages",
        },
        correct="right",
        why={
            "right": "The premium is the tokens of a language divided by the tokens of the same text in English: 5,414 / 2,741 = 1.98. This lesson is about models made for Dutch, and what their tokenizers make of it.",
            "upside": "That divides the wrong way around: English comes below the line. Which number goes on top? See [Languages with little data](/course/open/nlp-low-resource?lang=en).",
            "difference": "The premium divides, it does not subtract: it says how many times as many tokens. What do you divide by what? See [Languages with little data](/course/open/nlp-low-resource?lang=en).",
            "both": "The premium compares the two languages, so it cannot be their total. What do you divide by what? See [Languages with little data](/course/open/nlp-low-resource?lang=en).",
        },
    )
    quick_premium
    return (quick_premium,)


@app.cell(hide_code=True)
def quick_check_mask():
    quick_mask = L.choice(
        "quick-mask",
        question="BERT gets the sentence *The train to Utrecht leaves in ten [MASK].* What does it do with the [MASK]?",
        options={
            "next": "it writes the words that come after the sentence, one token at a time",
            "skip": "it leaves the [MASK] out",
            "right": "it guesses the piece at the [MASK] from the words on both sides",
            "lookup": "it looks up the most common word of English and puts it there",
        },
        correct="right",
        why={
            "right": "BERT learned by masked language modeling: in an encoder every token looks at all tokens, so it guesses a hidden piece from both sides. BERTje and RobBERT, in this lesson, are BERTs that learned the same way from Dutch text.",
            "next": "Writing one token at a time is what a decoder does. What does an encoder do at a hidden place? See [BERT: filling in the blank](/course/open/nlp-bert?lang=en).",
            "skip": "The [MASK] is exactly the place BERT learned to work on. What does it do there? See [BERT: filling in the blank](/course/open/nlp-bert?lang=en).",
            "lookup": "BERT uses the words around the [MASK], not a list of common words. What does it guess from? See [BERT: filling in the blank](/course/open/nlp-bert?lang=en).",
        },
    )
    quick_mask
    return (quick_mask,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Why this matters

    Dutch is in class 4 of Joshi and others: a lot of text, little labeled data. The models most people use learned from text that was mostly English, and in [Languages with little data](/course/open/nlp-low-resource?lang=en) you saw what their tokenizers make of Dutch: GPT-2's needs almost twice as many tokens for the same news as in English. A Dutch organization that wants a model often wants more than a model that also understands Dutch: one that writes good Dutch and knows the Netherlands, cuts Dutch into few tokens, may run on its own computers, and learned only from text it was allowed to use. Since 2019 Dutch has had models of its own, made in three ways, and the way a model was made shows in what it can do. This lesson is about those models: what each learned from, and what that tells you to expect from it.

    **An everyday example.** A guide to all of Europe gives Amsterdam two pages; a guide to the Netherlands gives it fifty, and knows the ferry to Texel too. Without a guide to the Netherlands, a European one with a Dutch chapter added is the next best thing.
    """)
    return


@app.cell(hide_code=True)
def guess_cell():
    guess_fietje = L.guess(
        "guess",
        question="Fietje is Phi-2, an English-centric model of 2.7 billion parameters, trained further on 28 billion tokens of Dutch text. RobBERT's tokenizer, made for Dutch, needs 1.31 tokens per word on the 1,997 Dutch sentences of NTREX-128. How many tokens per Dutch word does Fietje's tokenizer need?",
        options={
            "robbert": "about 1.3, like RobBERT: the 28 billion Dutch tokens taught it Dutch pieces",
            "between": "about 1.7, halfway between RobBERT and an English tokenizer",
            "right": "about 2.2, as many as GPT-2",
            "more": "about 3.5: learning a second language makes it cut both into more pieces",
        },
        reveal="**2.22**, to the last digit as many as GPT-2: 104,053 tokens for the 46,813 Dutch words. Training further changes the weights of a model, not its tokenizer: Fietje kept Phi-2's, and Phi-2's tokenizer is GPT-2's, with 38 tokens added for runs of spaces and tabs in code. GEITje kept Mistral's tokenizer in the same way: 2.01 tokens per Dutch word.",
    )
    guess_fietje
    return (guess_fietje,)


@app.cell(hide_code=True)
def text_idea():
    mo.md(r"""
    ## The idea

    **Three ways to a model for Dutch.** The models for Dutch so far were made in three ways. *From the start on Dutch*: a new model, with a tokenizer made from Dutch text, learns from Dutch text only; BERTje and RobBERT are encoders made this way. *Trained further*: a model that learned from mostly English goes on learning from Dutch text, with the tokenizer it had; GEITje and Fietje are decoders made this way. *From the start on many languages*: a new model, with a tokenizer made for all its languages, learns from Dutch, English and other languages at once; EuroLLM and GPT-NL are decoders made this way.

    **BERTje.** Wietse de Vries and five others at the University of Groningen trained a BERT the size of BERT-base, about 109 million parameters, on Dutch text: novels, the news corpus TwNC, SoNaR-500 of [Tools and corpora for Dutch](/course/open/nlp-dutch-tools?lang=en), four Dutch news sites and Wikipedia, "12GB of uncompressed text which amounts to about 2.4B tokens." Its vocabulary of 30,000 pieces was made from that text, and it is cased: it keeps capitals. Fine-tuned on the names in Dutch news of CoNLL-2002, BERTje reached a span F1 of 0.883, and mBERT, which learned Dutch only from Wikipedia, 0.807. The authors, in 2019: "This model consistently outperforms multilingual BERT on word-level NLP tasks."

    **RobBERT.** Pieter Delobelle, Thomas Winters and Bettina Berendt at KU Leuven took RoBERTa, the BERT that trained longer on more text, and trained it from the start on the Dutch part of OSCAR, web text from Common Crawl, a free collection of billions of web pages: "This Dutch corpus is 39GB large, with 6.6 billion words spread over 126 million lines of text". RobBERT has 117 million parameters. They made two versions: v1 kept RoBERTa's English tokenizer, and v2 got a new tokenizer of 40,000 tokens, made from the Dutch text the way GPT-2's was made from English. On *die* and *dat* that showed. Asked which of the two belongs at a [MASK], without any fine-tuning, v2 was right 98.75% of the time, v1 98.03%, BERTje 94.94% and mBERT 90.21%. And fine-tuned on the same task: "having a Dutch tokenizer strongly helps reduce the error rate for this task, halving the error rate when fine-tuned on the full dataset." RobBERT-2022 then added 2,774 tokens for newer words, such as words about corona, and RobBERT-2023 got a completely new tokenizer of 50,000 tokens.

    **GEITje: trained further, and taken offline.** In 2023 Edwin Rijgersberg took Mistral 7B, an open model of 7 billion parameters, and trained it further on Dutch: "GEITje was created by further training Mistral 7B on no less than 10 billion tokens of Dutch text from the Dutch Gigacorpus and the MADLAD-400 web crawling corpus." Then GEITje-chat learned from chat conversations translated into Dutch. GEITje kept Mistral's tokenizer of 32,000 tokens, and was open, under the license Apache 2.0. In January 2025 it was gone. Stichting BREIN, a Dutch foundation that acts for rights holders against illegal copies, said that GEITje had learned from tens of thousands of Dutch books from an illegal source, Library Genesis, and from text copied from news sites without permission. The README now opens with: "At the pressing request of Stichting BREIN GEITje is no longer available, starting immediately." What a model learned from decides whether it may be used at all.

    **Fietje: small, and trained further.** In 2024 Bram Vanroy did the same with a small model: Phi-2 of Microsoft, 2.7 billion parameters, which had learned from 1.4 trillion tokens, mostly English. Vanroy trained it further on 28 billion tokens of Dutch: Dutch Wikipedia and Dutch web text from CulturaX, filtered for quality. Fietje comes in three versions, under the license MIT: one that only continues a text, one that follows instructions, and one for conversations. It kept Phi-2's tokenizer: "Its tokenizer is a bit worse than other models in terms of fertility: on average 2.05 subword tokens are needed to encode a single Dutch word, highlighting that it was intended for English." Vanroy measured that on other Dutch text; on NTREX-128 it is 2.22, as the picture below shows. In Vanroy's tests, Fietje Chat, the version for conversations, beat GEITje Ultra, a version of GEITje for conversations and more than twice its size, on two of five tasks. And newer models that learned many languages beat both: "recent small models outperform older, larger models that were fine-tuned for Dutch."

    **EuroLLM: from the start on 35 languages.** EuroLLM, of Unbabel, the Instituto Superior Técnico in Lisbon, the University of Edinburgh and others, learned from the start on the 24 official languages of the European Union, Dutch among them, and 11 more, such as Arabic, Chinese and Ukrainian. Its models of 1.7, 9 and 22 billion parameters each learned from about 4 trillion tokens. In the main phase, half of them were English, 5% code and math, and the other 45% were shared over the other languages. Its tokenizer was made for all of them: "a vocabulary of 128,000 pieces provides the best trade-off", four times the 32,000 of Mistral's. The license is Apache 2.0.

    **GPT-NL: from the start on lawful data.** GPT-NL is a model of TNO, SURF and the Netherlands Forensic Institute, paid for by the Dutch government with €13.5 million. It learned from the start, from June to December 2025, only from data its makers obtained lawfully: text licensed by its owners, texts of Common Crawl under open licenses, synthetic text and code. Its site gives the mix: about 10% Dutch, 73% English and 17% other languages, and the model saw the Dutch texts more often than the others during training. An equal share of Dutch and English turned out not to be possible: with only lawful data, the makers drew on fewer sources than other developers. GPT-NL has 26 billion parameters and a tokenizer of 128,000 tokens, of which about 40% are Dutch. Its weights are not open: researchers can get them for a small fee, everyone else with a paid license, and in November 2026 only a few launching customers use the model, such as the KB, the national library. And OpenEuroLLM, 20 European partners with €37.4 million, partly from the EU, builds open models for all the official languages of the EU.

    **Encoder or decoder.** BERTje and RobBERT are encoders, like BERT: they read a whole text and give labels, after fine-tuning on labeled examples, as in [Fine-tuning an encoder for a task](/course/open/nlp-fine-tune-bert?lang=en). With about 110 million parameters they are small and fast. GEITje, Fietje, EuroLLM and GPT-NL are decoders: they write text, with billions of parameters.

    **What the tokenizer costs.** The picture compares the tokenizers of seven models on all 1,997 sentences of NTREX-128, in Dutch and in English, in tokens per word: the tokenizer fertility of [Languages with little data](/course/open/nlp-low-resource?lang=en). RobBERT's tokenizer needs 1.31 tokens per Dutch word, BERTje's 1.32, and both need more for English: 1.74 and 2.11. EuroLLM's, made for 35 languages, needs 1.47 for Dutch and 1.35 for English; Qwen3's 1.83 and 1.26. Mistral's needs 2.01 for Dutch, and Fietje's cuts both languages exactly as GPT-2's does. Training further teaches a model Dutch, but leaves the cost of its tokenizer as it was.
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
    | **BERTje** | a Dutch BERT of the University of Groningen (2019), trained from the start on 12 GB of Dutch books, news and Wikipedia |
    | **RobBERT** | a Dutch RoBERTa of KU Leuven (2020), trained on Dutch web text with a tokenizer made for Dutch; RobBERT-2023 is its newest version |
    | **GEITje** | Mistral 7B trained further on 10 billion tokens of Dutch (2023); taken offline in 2025 because of where part of its text came from |
    | **Fietje** | Phi-2, 2.7 billion parameters, trained further on 28 billion tokens of Dutch (2024), with Phi-2's tokenizer |
    | **GPT-NL** | a Dutch model of TNO, SURF and the NFI, trained from the start on data obtained lawfully, about 10% of it Dutch; its weights are not open |
    | **EuroLLM** | open models trained from the start on 35 languages, the 24 official languages of the EU among them, with a tokenizer of 128,000 tokens |
    """)
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recipe

    To choose a model for a task in Dutch:

    1. Decide what the task gives back. A label per text or per word: an encoder such as RobBERT is enough. New text: you need a decoder.
    2. Measure what its tokenizer makes of your Dutch text: tokens per word, and how much of the text fits in the context, the tokens the model reads at once.
    3. Read what it learned from: how much Dutch, from which sources, and under which license.
    4. Try it on examples of your own task, next to a newer model that learned many languages.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Worked example

    `C.OWN` holds the course's three sentences in Dutch, `C.OWN["nl"]`, and the same in English, `C.OWN["en"]`. `C.ENC` is GPT-2's tokenizer, which is also Fietje's, and `C.ROBBERT` RobBERT v2's. Both have `encode`, which gives the ids of the tokens of a text, and `decode`, which turns a list of ids back into text. `C.ROBBERT.pieces(text)` gives its tokens as they stand in its vocabulary, with Ġ for a space. `C.KEYS` names the seven tokenizers of the picture, and `C.counts()` gives the course's counts of all 1,997 sentences of NTREX-128 for each: the words and the tokens, in Dutch and in English. `C.per_word(key, code)` works out tokens per word from those counts, with `"nld"` for Dutch and `"eng"` for English:
    """)
    return


@app.cell
def example_cell():
    _sentence = C.OWN["nl"][0]
    _gpt2 = C.ENC.encode(_sentence)
    _robbert = C.ROBBERT.encode(_sentence)
    print(len(_gpt2), _gpt2[:6], C.ENC.decode(_gpt2))
    print(len(_robbert), _robbert[:6], C.ROBBERT.decode(_robbert))
    print(C.ROBBERT.pieces(_sentence))
    for _key in C.KEYS:
        print(_key, C.counts()[_key]["nld_tokens"], round(C.per_word(_key, "nld"), 2), round(C.per_word(_key, "eng"), 2))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    GPT-2 makes 29 tokens of the Dutch sentence about the cycle path, RobBERT 13: one for every word, and one for the period. The ids are not the same, `[5005, 16840, 68, ...]` against `[62, 512, 2733, ...]`, because every tokenizer numbers its own tokens, and `decode` gives back the sentence in both cases. RobBERT's pieces show why it needs so few: *Ġgemeente*, *Ġfietspad* and *Ġkanaal* are tokens of its own, each word with its space.

    Then NTREX-128. GPT-2 and Fietje make 104,053 tokens of the 46,813 Dutch words, 2.22 per word, and 1.24 per English word. RobBERT makes 61,429, 1.31 per word, and needs 1.74 per English word: it learned its pieces from Dutch, so English is now the language that costs more.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_pieces():
    mo.vstack([
        mo.md(r"""
        ## Your turn

        ### Exercise 1. pieces: how does a tokenizer cut a word?

        Write `pieces`, which gets `word`, a word, and `enc`, a tokenizer, and gives back the pieces of the word as a list of texts, with a space before the word, as it stands in a sentence. In the start code, `out` starts as an empty list, and `for token in enc.encode(" " + word)` goes through the ids of the tokens. `out.append(piece)` adds each piece to the list. Replace the `...` with the text of one token, `token`: `enc.decode` turns a list of ids into text. You wrote the same loop in [Tokens: how real LLMs chop up text](/course/open/lm-tokens-outside?lang=en) with one fixed tokenizer; here the tokenizer is an argument, `enc`, so one function cuts with every tokenizer.

        The check calls `pieces("fietsenstalling", C.ENC)`, with GPT-2's tokenizer, which gives `[' fi', 'ets', 'en', 'st', 'alling']`. Then cases of its own: Dutch and English words, with GPT-2's tokenizer and with RobBERT's, `C.ROBBERT`.
        """),
        L.task("Give `piece` the text of the one token `token`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pieces():
    box_pieces = L.code_box("pieces", lines=6)
    box_pieces
    return (box_pieces,)


@app.cell(hide_code=True)
def check_pieces(box_pieces):
    result_pieces = L.check("pieces", box_pieces.value)
    result_pieces
    return (result_pieces,)


@app.cell(hide_code=True)
def hints_pieces():
    L.hints(
        "pieces",
        [
            "`piece` is the text of one token. `enc.decode` turns a list of ids into text, so give it a list that holds only `token`.",
            "`C.ENC.decode([15496, 0])` gives `\"Hello!\"`, and `C.ENC.decode([0])` gives `\"!\"`: a list with one id gives the text of that one token.",
            "`        piece = enc.decode(___)`",
        ],
        why="`piece = enc.decode([token])`: `decode` takes a list, and `[token]` is a list of one id. Without the brackets, `enc.decode(token)` stops with a `TypeError`.",
    )
    return


@app.cell(hide_code=True)
def text_after_pieces():
    mo.md(r"""
    In the picture, your function cuts five Dutch words and two English ones with both tokenizers. GPT-2 cuts every Dutch word into three to five pieces, which mostly do not follow the parts of the word: *ziekenhuis*, hospital, becomes *z*, *ie*, *ken*, *hu* and *is*. RobBERT's tokenizer keeps *gemeenteraad*, *ziekenhuis*, *broodjes* and *uitzendbureau* whole, and cuts *fietsenstalling* into *fietsen* and *stalling*. With English it is the other way around: GPT-2 keeps *library* and *Saturday* whole, and RobBERT cuts *Saturday* into *Sat*, *ur* and *day*. A tokenizer learns its pieces from the text it was made from.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_words_in_context():
    mo.vstack([
        mo.md(r"""
        ### Exercise 2. words_in_context: how much Dutch fits?

        A model reads a fixed number of tokens at once, its context: Fietje 2,048, like Phi-2. Write `words_in_context`, which gets `context`, a number of tokens, `texts`, a list of sentences, and `enc`, a tokenizer, and gives back how many words of such text fit in `context` tokens. In the start code, `tokens` and `words` start at 0, and `for text in texts` goes through the sentences: `tokens = tokens + len(enc.encode(text))` counts the tokens, and `words = words + len(text.split())` the words. Replace the `...` after `return` with the number of words that fit: `context` times the part of a word that one token holds on average.

        The check calls `words_in_context(2048, ["De gemeente legt volgend jaar een nieuw fietspad aan langs het kanaal."], C.ENC)`, which gives `847.448275862069`: 29 tokens for 12 words. Then cases of its own: the course's sentences in Dutch and in English, with both tokenizers, and contexts of 512 and 8,192 tokens.
        """),
        L.task("Return the number of words that fit in `context` tokens."),
    ])
    return


@app.cell(hide_code=True)
def exercise_words_in_context():
    box_words_in_context = L.code_box("words_in_context", lines=8)
    box_words_in_context
    return (box_words_in_context,)


@app.cell(hide_code=True)
def check_words_in_context(box_words_in_context):
    result_words_in_context = L.check("words_in_context", box_words_in_context.value)
    result_words_in_context
    return (result_words_in_context,)


@app.cell(hide_code=True)
def hints_words_in_context():
    L.hints(
        "words_in_context",
        [
            "One token holds on average the words of the texts divided by their tokens. Multiply that by `context`.",
            "When 17 tokens hold 8 words, one token holds 8 / 17 of a word, and 100 tokens hold 100 × 8 / 17, about 47 words.",
            "`    return context * ___`",
        ],
        why="`return context * words / tokens`: `words / tokens` is the part of a word that one token holds, the inverse of the tokens per word of [Languages with little data](/course/open/nlp-low-resource?lang=en). `context / tokens * words` gives the same.",
    )
    return


@app.cell(hide_code=True)
def text_after_words_in_context():
    mo.md(r"""
    In the picture, your function works on the course's three sentences: in 2,048 tokens, GPT-2's tokenizer fits 970 words of their Dutch and 1,787 of their English, and RobBERT's 1,843 words of Dutch and 1,354 of English. Below that are the course's counts of NTREX-128 for all seven tokenizers: Fietje's 2,048 tokens hold 921 Dutch words, and with RobBERT's tokenizer they would hold 1,561. A model whose tokenizer cuts Dutch into more pieces reads less Dutch at once, writes Dutch more slowly, one token at a time, and costs more per Dutch word through a service that charges per token.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_share():
    mo.vstack([
        mo.md(r"""
        ### Exercise 3. share: how much Dutch did a model learn from?

        Write `share`, which gets `mix`, the training text of a model as a dictionary from a language to its billions of tokens, and `language`, one of its keys, and gives back the share of that language: its tokens divided by all the tokens. In the start code, `total` starts at 0, and `for name, tokens in mix.items()` adds the tokens of every language to it. Replace the `...` after `return` with the share of `language`.

        The check calls `share({"nl": 29, "en": 208, "other": 48}, "nl")`, the text of GPT-NL, which gives `0.10175438596491228`. Then cases of its own: other languages, and the training text of other models.
        """),
        L.task("Return the share of `language` in `mix`."),
    ])
    return


@app.cell(hide_code=True)
def exercise_share():
    box_share = L.code_box("share", lines=6)
    box_share
    return (box_share,)


@app.cell(hide_code=True)
def check_share(box_share):
    result_share = L.check("share", box_share.value)
    result_share
    return (result_share,)


@app.cell(hide_code=True)
def hints_share():
    L.hints(
        "share",
        [
            "The tokens of `language` are in `mix`, under the key `language`. Divide them by `total`.",
            "`{\"nl\": 29, \"en\": 208}[\"en\"]` gives `208`: square brackets after a dictionary give the value of a key.",
            "`    return ___ / total`",
        ],
        why="`return mix[language] / total`: the tokens of one language divided by the tokens of all of them, a number from 0 to 1.",
    )
    return


@app.cell(hide_code=True)
def text_after_share():
    mo.md(r"""
    In the picture, your function works on the training text of four models. BERTje learned from Dutch only: 100%. GPT-NL, a model for Dutch, learned from text that is 10.2% Dutch. Fietje's 28 billion Dutch tokens are 2.0% of all the tokens it learned from, the 1.4 trillion of Phi-2 included. OpenEuroLLM's first test model, Prelude, of 2026, gave Dutch 47.5 billion of the 7,815 billion tokens of its first stage: 0.6%, more Dutch tokens than GEITje and Fietje together, in a far larger whole. For GEITje no share can be given: Mistral has not published how much text Mistral 7B learned from.

    ### Try it: your text, seven tokenizers

    The demo needs the tokenizers of BERTje, Mistral 7B, Fietje, EuroLLM and Qwen3. Type a text in the box: the picture cuts it with all seven tokenizers, fewest tokens first.

    Start with the sentence in the box. Then try a long Dutch compound, such as *arbeidsongeschiktheidsverzekering*, alone and in a sentence, then a sentence in English, and *Eén café, twee cafés.*
    """)
    return


@app.cell(hide_code=True)
def get_button_cell():
    get_button = mo.ui.run_button(label=f"Download the five tokenizers, {C.download_size(C.FILES, L.lang)}")
    return (get_button,)


@app.cell(hide_code=True)
def tokenizers_cell(get_button):
    C.files_or_stop(get_button, {
        "ask": f"*The tokenizers of BERTje, Mistral 7B, Fietje, EuroLLM and Qwen3 are not on your computer yet. The button downloads them once from Hugging Face, {C.download_size(C.FILES, L.lang)}: Mistral 7B's, EuroLLM's and Qwen3's under the license Apache-2.0, Fietje's under MIT, and BERTje's vocabulary from the page of its makers. The rest of the lesson works without them.*",
        "busy": "**Downloading** {done} of {total} MB …",
        "no_net": "*The download did not work: Hugging Face could not be reached, or the connection broke off. Check your connection, and click again.*",
        "not_pinned": "*The download did not work: a file that came in is not the course's pinned file, so the course threw it away. Click again; if it happens again, report it on GitHub.*",
        "no_room": "*The download did not work: your disk has no room for it. Make room, and click again.*",
        "failed": "*The download did not work ({why}). Click again.*",
    })
    try_text = mo.ui.text_area(value=C.TRY_START, label="Your text", full_width=True, rows=3)
    try_text
    return (try_text,)


@app.cell(hide_code=True)
def try_look(try_text):
    mo.Html(C.try_picture(L.lang, try_text.value))
    return


@app.cell(hide_code=True)
def text_after_try():
    mo.md(r"""
    In the sentence about the town council, RobBERT and BERTje need 16 tokens for 13 words, and both keep *gemeenteraad*, *vanavond* and *binnenstad* whole. EuroLLM needs 21, Qwen3 26, Mistral's tokenizer, which GEITje kept, 32, and GPT-2 and Fietje 34 each, token for token the same. Typed alone, *arbeidsongeschiktheidsverzekering* costs GPT-2 12 tokens and RobBERT 6; in a sentence, after a space, it costs RobBERT 2, *arbeidsongeschiktheids* and *verzekering*. RobBERT learned its pieces from running text, where a word almost always stands after a space. In an English sentence the order turns around: RobBERT and BERTje need the most tokens. And in *Eén café, twee cafés.* GPT-2 cuts *Eén*, *twee* and *cafés* in two each, while BERTje and RobBERT keep every word whole, *Eén* with its capital too.

    ### Exercise 4. Encoder or decoder?

    You have six jobs with Dutch text. Sort each one: is an encoder such as RobBERT enough, or does the job need a decoder such as Fietje?
    """)
    return


@app.cell(hide_code=True)
def exercise_fits():
    sort_fits = L.sort("fits")
    sort_fits
    return (sort_fits,)


@app.cell(hide_code=True)
def check_fits(sort_fits):
    result_fits = L.check("fits", sort_fits)
    result_fits
    return (result_fits,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Wrap-up

    **How this connects.** BERTje and RobBERT are BERTs, as in [BERT: filling in the blank](/course/open/nlp-bert?lang=en), and they are fine-tuned for a task as in [Fine-tuning an encoder for a task](/course/open/nlp-fine-tune-bert?lang=en). BERTje learned partly from SoNaR, of [Tools and corpora for Dutch](/course/open/nlp-dutch-tools?lang=en). Tokens per word is the tokenizer fertility of [Languages with little data](/course/open/nlp-low-resource?lang=en): a tokenizer made for Dutch brings it down to about 1.3, while training further leaves the tokenizer, and its cost, as it was. This lesson ends the module about more languages, and with it the basis of the domain Language and search.

    **An everyday example.** A cookbook translated from English into Dutch reads in Dutch, but it often still measures in cups and ounces: the translator changed the words, not the measures. Training a model further on Dutch changes what it knows, and leaves its tokenizer, the measure it cuts text with, as it was.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_pieces, result_words_in_context, result_share, result_fits):
    L.wrap_up(result_pieces, result_words_in_context, result_share, result_fits)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
