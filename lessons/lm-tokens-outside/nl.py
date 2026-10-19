# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les lm-tokens-outside. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai.lesson import lesson

    L = lesson("lm-tokens-outside", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_count():
    quick_count = L.choice(
        "quick-count",
        question="Met één teken per token: hoeveel tokens heeft de tekst \"a cat\"?",
        options={
            "words": "2: één voor elk woord, \"a\" en \"cat\"",
            "letters": "4: één voor elke letter",
            "right": "5: de spatie is ook een token",
        },
        correct="right",
        why={
            "words": "In die les is een token één teken, geen woord. Zie [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl).",
            "letters": "De spatie is ook een teken, en dus een token. Zie [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl).",
            "right": "\"a\", spatie, \"c\", \"a\", \"t\": vijf tekens, vijf tokens. Echte modellen gebruiken grotere stukken, en vandaag zie je welke.",
        },
    )
    quick_count
    return (quick_count,)


@app.cell(hide_code=True)
def quick_check_examples():
    quick_examples = L.choice(
        "quick-examples",
        question="Hoeveel voorbeelden voor next-token prediction geeft een tekst van 10 tokens?",
        options={
            "right": "9: elk token na het eerste is het label van één voorbeeld",
            "ten": "10: één voor elk token",
            "one": "1: de hele tekst is één voorbeeld, met zijn laatste token als label",
        },
        correct="right",
        why={
            "ten": "Het eerste token heeft geen tekst ervoor, dus het is nooit een label. Zie [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl).",
            "one": "Elke plek geeft een voorbeeld: de tekst tot nu toe, en het token dat volgt. Zie [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl).",
            "right": "Een tekst van $n$ tokens geeft $n - 1$ voorbeelden. Hoe een tekst in tokens wordt geknipt, bepaalt dus hoeveel voorbeelden hij geeft, en hoe lang hij is voor het model.",
        },
    )
    quick_examples
    return (quick_examples,)


@app.cell(hide_code=True)
def text_why():
    mo.md(r"""
    ## Waarom

    Het language model uit [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl) las één teken tegelijk. Echte language models lezen tokens die hele woorden en stukken van woorden zijn, uit een vaste lijst van tienduizenden. Veel van wat je over LLM's hoort, wordt in die tokens geteld: hoeveel tekst een model in één keer kan lezen, wat het kost per miljoen tokens, hoe snel het schrijft. En hoeveel tokens een tekst kost, hangt af van hoe de tokenizer hem knipt.

    **Een voorbeeld uit het dagelijks leven.** Met alleen de kleinste bouwsteentjes, met één nop, kun je alles bouwen, maar dat kost heel veel steentjes. Met alleen kant-en-klare muren bouw je snel, maar alleen waar je muren voor hebt. Met een doos met grote stukken voor wat je vaak bouwt, en kleine voor de rest, bouw je alles, en wat je vaak bouwt met weinig stukken. Tekens zijn de steentjes met één nop, hele woorden de kant-en-klare muren, en echte tokens de gemengde doos.
    """)
    return


@app.cell(hide_code=True)
def predict_dutch():
    guess_dutch = L.guess(
        "dutch-tokens",
        question="GPT-2, een open language model uit 2019, knipt de Engelse zin \"Tokenization is surprisingly important.\" in 6 tokens. Ongeveer hoeveel tokens maakt het van dezelfde zin in het Nederlands, \"Tokenisatie is verrassend belangrijk.\"?",
        options={
            "same": "Ongeveer 6: dezelfde zin, evenveel woorden",
            "fewer": "Ongeveer 4: één token per woord",
            "right": "Ongeveer 15: meer dan twee keer zoveel",
        },
        reveal="**15.** \"Tokenization\" is 2 tokens, \"Token\" en \"ization\", en \" surprisingly\" en \" important\" zijn er elk één. De Nederlandse woorden vallen uit elkaar: \" verrassend\" wordt \" ver\", \"r\", \"ass\" en \"end\", en \" belangrijk\" vijf stukken. GPT-2 maakte zijn lijst tokens uit tekst die bijna helemaal Engels was, dus het heeft veel Engelse woorden heel, en van de meeste Nederlandse woorden alleen stukken.",
    )
    guess_dutch
    return (guess_dutch,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **tokenizer** knipt een tekst in tokens en geeft elk token zijn nummer, zijn **token ID**; hij maakt van token IDs ook weer tekst. Zijn **vocabulary** is de vaste lijst van alle tokens die hij kent: voor GPT-2 zijn dat er 50.257. Een veelgebruikt woord is één token, vaak met de spatie ervoor, en een zeldzamer woord valt uiteen in **subwords**: stukken van woorden die veel voorkomen, zoals "un", "bel", "iev" en "able". De kleinste tokens zijn de 256 bytes uit [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl), dus elke tekst past, ook een naam die de tokenizer nooit gezien heeft, of een emoji: in het ergste geval valt die uiteen in bytes.

        De vocabulary is één keer gemaakt, uit heel veel tekst: stukken die vaak samen voorkwamen, werden één token. Bij GPT-2 was die tekst bijna helemaal Engels, dus Engelse woorden zijn vaak heel, en Nederlandse woorden vallen in meer stukken uiteen. In een latere les bouw je zo'n tokenizer zelf.

        In Python heeft de library **tiktoken** de tokenizers van de modellen van OpenAI: `tiktoken.get_encoding("gpt2")` geeft die van GPT-2. `enc.encode(text)` geeft de lijst token IDs, `enc.decode(ids)` de tekst terug, en `enc.n_vocab` de grootte van de vocabulary. Oefening 3 gebruikt ook één nieuw stukje Python: `text.split()` knipt een tekst bij de spaties in een lijst woorden.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **tokenizer** | | knipt een tekst in tokens, geeft elk zijn token ID, en maakt van token IDs weer tekst | `enc.encode(text)`, `enc.decode(ids)` |
        | **token ID** | | het nummer van een token in de vocabulary | `enc.encode(" the")` geeft `[262]` |
        | **vocabulary** | | de vaste lijst van alle tokens die een tokenizer kent | `enc.n_vocab` |
        | **subword** | | een token dat een stuk van een woord is | `"un"`, `"able"` |
        | `split` | | de woorden van een tekst, geknipt bij de spaties | `"a cat".split()` geeft `['a', 'cat']` |
        """),
        L.watch_out("Een token is geen woord: \" the\", met de spatie, is een ander token dan \"the\", en \"The\" is een derde. De token IDs horen bij één tokenizer: 262 is \" the\" voor GPT-2, en kan iets anders zijn voor een model met een andere tokenizer. En een teken van meerdere bytes kan in meerdere tokens uiteenvallen: 🙂 is 4 bytes en 2 tokens. Decodeer er één los, en je krijgt �-tekens, het teken voor bytes die geen heel teken zijn."),
        mo.md(r"""
        De figuur: de twee zinnen uit de gok, geknipt in de tokens van GPT-2, met onder elk token zijn token ID. Een spatie zie je als ␣: in GPT-2 hoort hij bij het token erna.
        """),
        mo.Html(C.tokens_picture({"title": "Eén zin, twee talen", "en": "Engels: {n} tokens", "nl": "Nederlands: {n} tokens",
                                  "alt": "Engels, 6 tokens: Token, ization, spatie is, spatie surprisingly, spatie important, punt. Nederlands, 15 tokens: Token, is, at, ie, spatie is, spatie ver, r, ass, end, spatie bel, ang, ri, j, k, punt."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    De tokens van een tekst, met een kant-en-klare tokenizer:

    1. Laad de tokenizer: `enc = tiktoken.get_encoding("gpt2")`.
    2. Encodeer de tekst: `enc.encode(text)` geeft zijn token IDs, in een lijst.
    3. Tel de tokens: de lengte van die lijst.
    4. Decodeer om de stukken te zien: `enc.decode([i])` geeft het stuk tekst van token ID `i`, en `enc.decode(ids)` de hele tekst terug.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Een Engelse en een Nederlandse zin, door de tokenizer van GPT-2. Het commentaar met `#` zegt welke stap van het recept elk deel is. Voer de cel uit.
    """)
    return


@app.cell
def example_tokens():
    import tiktoken

    # 1. de tokenizer van GPT-2, en de grootte van zijn vocabulary
    _enc = tiktoken.get_encoding("gpt2")
    print(_enc.n_vocab)
    # 2. de token IDs van de twee zinnen
    _en = _enc.encode("The cat sat on the mat.")
    _nl = _enc.encode("De kat zat op de mat.")
    print(_en)
    print(_nl)
    # 3. het aantal tokens
    print(len(_en), len(_nl))
    # 4. de stukken van de Nederlandse zin, en de hele tekst terug
    print([_enc.decode([_i]) for _i in _nl])
    print(_enc.decode(_nl))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    GPT-2 kent 50.257 tokens. De Engelse zin is 7 tokens: één voor elk woord, behalve het eerste met de spatie ervoor, en één voor de punt. "The" aan het begin is 464, en " the" met zijn spatie is 262. De Nederlandse zin is 9 tokens: " kat" en " zat" staan niet in de vocabulary, dus die vallen uiteen in " k" en "at", en " z" en "at". " mat" is in beide zinnen hetzelfde token, 2603. En `decode` geeft de hele zin precies terug.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_space():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Twee IDs voor "the"
    """)
    return


@app.cell(hide_code=True)
def choice_space_cell():
    choice_space = L.choice(
        "space",
        question="`enc.encode(\"the\")` geeft `[1169]`, en `enc.encode(\" the\")` geeft `[262]`. Waarom twee verschillende token IDs?",
        options={
            "slip": "De tokenizer vergist zich: één woord hoort altijd één token ID te krijgen, wat er ook voor staat",
            "right": "De spatie hoort bij het token: \"the\" en \" the\" zijn twee tokens van de vocabulary",
            "random": "De token IDs worden bij elke aanroep van `encode` willekeurig gekozen",
        },
        correct="right",
        why={
            "slip": "Het is geen vergissing: de tokenizer knipt de hele tekst, spaties ook. Een woord midden in een zin heeft een spatie ervoor, dus \" the\" komt veel vaker voor dan \"the\" en kreeg een eigen token.",
            "random": "Roep `encode` nog eens aan en je krijgt dezelfde IDs: de vocabulary ligt vast, en elk token ID erin ook.",
            "right": "Midden in een zin heeft bijna elk woord een spatie ervoor, dus de vocabulary van GPT-2 heeft \" the\", met de spatie, als één token. \"the\" zonder spatie, aan het begin van een regel, is een ander token.",
        },
    )
    choice_space
    return (choice_space,)


@app.cell(hide_code=True)
def text_exercise_pieces():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. De stukken van elke tekst

        Laat `pieces(text)` de tokens van een tekst geven als stukken tekst, in een lijst. Het vak laadt de tokenizer als `enc`. Vervang de twee `...`: de token IDs van de tekst, volgens stap 2 van het recept, en voor elk token ID zijn stuk tekst, volgens stap 4. De check roept `pieces("Hello world")` aan, wat `['Hello', ' world']` moet geven, en probeert andere teksten, waaronder een lege.
        """),
        L.task("Laat `pieces` de tokens van een tekst geven als stukken tekst."),
    ])
    return


@app.cell(hide_code=True)
def exercise_pieces():
    box_pieces = L.code_box("pieces", lines=11)
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
            "Twee stappen van het recept: eerst de token IDs van de hele tekst, dan in de loop één stuk voor elk token ID.",
            "Stap 2 geeft de token IDs: `enc.encode` van de tekst. Stap 4 maakt van één token ID weer tekst, en `decode` wil een lijst, ook voor één ID.",
            "`ids = enc.encode(___)`, en in de loop `out.append(enc.decode([___]))`.",
        ],
        why="Voor \"Hello world\" geeft `enc.encode` de lijst `[15496, 995]`, en elk ID los decoderen geeft \"Hello\" en \" world\", met zijn spatie: de spatie hoort bij het token erna.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_per_word():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Tokens per woord

        Laat `tokens_per_word(text)` geven hoeveel tokens een tekst per woord kost: zijn aantal tokens gedeeld door zijn aantal woorden. Vervang de twee `...`: het aantal tokens, volgens stap 3 van het recept, en het aantal woorden, met `text.split()`. De check roept `tokens_per_word("The weather is nice today.")` aan, wat `1.2` moet geven, en probeert andere teksten, een paar ook in het Nederlands.
        """),
        L.task("Laat `tokens_per_word` het aantal tokens van een tekst per woord geven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_per_word():
    box_per_word = L.code_box("tokens-per-word", lines=9)
    box_per_word
    return (box_per_word,)


@app.cell(hide_code=True)
def check_per_word(box_per_word):
    result_per_word = L.check("tokens-per-word", box_per_word.value)
    result_per_word
    return (result_per_word,)


@app.cell(hide_code=True)
def hints_per_word():
    L.hints(
        "tokens-per-word",
        [
            "Twee keer tellen: de tokens van de tekst, en zijn woorden. Allebei zijn het de lengte van een lijst.",
            "De lijst tokens komt uit `enc.encode(text)`, en de lijst woorden uit `text.split()`. `len` geeft de lengte van een lijst.",
            "`n_tokens = len(enc.encode(___))` en `n_words = len(___.split())`.",
        ],
        why="\"The weather is nice today.\" is 6 tokens, vijf woorden en een punt, en `split` geeft 5 woorden, met de punt vast aan \"today.\": 6 / 5 = 1,2.",
    )
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De tokens uit [Een language model voorspelt het volgende token](/course/open/lm-what-is-a-language-model?lang=nl) waren tekens; een echte tokenizer knipt tekst in woorden en stukken van woorden, uit een vaste vocabulary, en zijn kleinste stukken zijn de bytes uit [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl). Een language model leest en schrijft alleen token IDs: het geeft een kans voor elk token van de vocabulary. In de volgende les, *Een bigram-model door te tellen*, maak je een eigen language model door te tellen, weer met tekens als tokens: een kleine vocabulary die makkelijk te tellen is.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_space, result_pieces, result_per_word):
    L.wrap_up(choice_space, result_pieces, result_per_word)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
