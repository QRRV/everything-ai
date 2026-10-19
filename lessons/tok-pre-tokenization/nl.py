# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les tok-pre-tokenization. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("tok-pre-tokenization", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Byte-level BPE](/course/open/tok-byte-level?lang=nl) en [Lijsten en dictionaries bouwen in één regel](/course/open/py-comprehensions?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_decode():
    quick_decode = L.choice(
        "quick-decode",
        question="Een byte-level BPE decodeert het losse id 195, de eerste byte van é, met `errors=\"replace\"`. Welke tekst komt eruit?",
        options={
            "e": "`é`: de tokenizer weet bij welk teken het hoort",
            "right": "`�`: één byte van é is geen heel teken",
            "a": "`Ã`: het teken met code point 195",
        },
        correct="right",
        why={
            "e": "Eén byte kan geen é zijn: é heeft zowel 195 als 169 nodig. Zie [Byte-level BPE](/course/open/tok-byte-level?lang=nl).",
            "a": "Zo leest Latin-1 de byte 195. In UTF-8 begint 195 alleen een teken van twee bytes. Zie [Byte-level BPE](/course/open/tok-byte-level?lang=nl).",
            "right": "Vandaag komen de bytes in stukken: woorden, leestekens en spaties, elk apart geëncodeerd.",
        },
    )
    quick_decode
    return (quick_decode,)


@app.cell(hide_code=True)
def quick_check_comprehension():
    quick_comprehension = L.choice(
        "quick-comprehension",
        question="Wat geeft `[w + \"!\" for w in [\"to\", \"be\"]]`?",
        options={
            "joined": "`\"to!be!\"`",
            "added": "`[\"to\", \"be\", \"!\"]`",
            "right": "`[\"to!\", \"be!\"]`",
        },
        correct="right",
        why={
            "joined": "Een comprehension geeft een lijst, met één item voor elk item waar hij langs gaat. Zie [Lijsten en dictionaries bouwen in één regel](/course/open/py-comprehensions?lang=nl).",
            "added": "De `!` komt achter elk item, niet achter de lijst. Zie [Lijsten en dictionaries bouwen in één regel](/course/open/py-comprehensions?lang=nl).",
            "right": "Vandaag doet een comprehension iets met elk stuk van een tekst.",
        },
    )
    quick_comprehension
    return (quick_comprehension,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je byte-level BPE uit [Byte-level BPE](/course/open/tok-byte-level?lang=nl) voegt samen over woorden heen: `e␣` en `,␣` werden tokens, elk met het eind van een woord en de spatie voor het volgende. De vocabulary loopt dan vol met zulke combinaties, `the␣`, `and␣`, `you␣`, en hetzelfde woord krijgt andere tokens, afhankelijk van wat erop volgt. GPT-2 knipt een tekst eerst in woorden, getallen en leestekens, elk met de spatie ervoor, en draait BPE alleen binnen die stukken. In deze projectstap voeg je dat knippen toe aan je tokenizer.
        """),
        draw.project_steps(["Letters en paren tellen", "Een bigram-model door te tellen", "Hetzelfde model, geleerd door een neural network",
                            "Je eigen tokenizer", "Je eigen GPT", "Een klein LLM pretrainen", "Het LLM instructies en voorkeuren laten volgen"],
                           3, here_label="← vandaag", title="Waar je project staat: je eigen LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_first():
    guess_first = L.guess(
        "first-merge",
        question="Zonder pre-tokenization was de eerste merge op 20.000 tekens van Tiny Shakespeare `e` en een spatie. Met pre-tokenization zijn de stukken woorden met de spatie ervoor, leestekens en spaties. Welk paar wordt nu volgens jou als eerste samengevoegd?",
        options={
            "e": "`e` en een spatie, net als eerst",
            "right": "Een spatie en `t`, zoals aan het begin van the, to en that",
            "th": "`t` en `h`, zoals in the, that en thou",
        },
        reveal="**Een spatie en `t`**: 404 keer binnen de stukken, vaker dan elk ander paar. `e` en een spatie kunnen elkaar niet meer tegenkomen: binnen een stuk komt een spatie nooit na een letter. `t` en `h` komen 402 keer voor, vlak erachter.",
    )
    guess_first
    return (guess_first,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Pre-tokenization** knipt een tekst in stukken vóór BPE, en BPE telt en voegt alleen binnen elk stuk samen, dus een merge loopt nooit van het ene stuk in het volgende. Om een nieuwe tekst te encoderen, wordt die op dezelfde manier geknipt, wordt elk stuk apart geëncodeerd, en komen de ids achter elkaar.

        De stukken komen uit een **reguliere expressie**: een kort patroon voor stukken tekst. `re.findall(pattern, text)`, uit Pythons module `re`, geeft elk stuk van `text` dat in het patroon past, van links naar rechts, zonder overlap. Het patroon van deze les is `r" ?\w+| ?[^\w\s]+|\s+"`; de `r` voor de aanhalingstekens laat elke backslash zoals hij is. De drie delen worden gescheiden door `|`, wat *of* betekent; op elke plek wint het eerste deel dat past:

        - `r" ?\w+"`: misschien een spatie (`?` maakt wat ervoor staat optioneel), dan één of meer (`+`) letters of cijfers (`\w`), zoveel als er zijn: een woord, met de spatie ervoor.
        - `r" ?[^\w\s]+"`: misschien een spatie, dan één of meer tekens die geen (`[^...]`) letter, cijfer of witruimte (`\s`) zijn: leestekens, of een emoji.
        - `r"\s+"`: witruimte die overblijft, zoals een nieuwe regel.

        Dus `re.findall(PATTERN, "Hi, you!")` geeft `["Hi", ",", " you", "!"]`. GPT-2 gebruikt een langer patroon van dezelfde soort, dat ook uitgangen zoals `'s` en `'re` apart houdt, en getallen apart van letters.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **pre-tokenization** | | een tekst in stukken knippen vóór BPE, zodat geen merge van het ene stuk in het volgende loopt | `re.findall(PATTERN, text)` |
        | **reguliere expressie** | | een kort patroon voor stukken tekst | `r" ?\w+"` |
        | `re.findall` | re find all | elk stuk van een tekst dat in een patroon past, van links naar rechts, zonder overlap | `re.findall(r"\w+", "Hi, you!")` |
        | `\w`, `\s` | | een letter, een cijfer of `_`; witruimte, zoals een spatie of een nieuwe regel | `\s+` |
        | `+`, `?` | | één of meer van wat ervoor staat, zoveel als er zijn; misschien één | `r" ?\w+"` |
        | `|` | of | elk deel mag passen; het eerste dat past, wint | `r"\w+|\s+"` |
        | `[^...]` | | één teken dat geen van de tekens erin is | `[^\w\s]` |
        | `r"..."` | | een raw string: elke backslash erin blijft zoals hij is | `r"\w+"` |
        """),
        L.watch_out("Train en encodeer met hetzelfde patroon: een tokenizer die bij het encoderen andere stukken knipt dan toen hij leerde, komt paren tegen die hij nooit telde. En knip eerst de tekst, en maak dan van elk stuk bytes: `re.findall` met dit patroon knipt een tekst, geen bytes."),
        mo.md(r"""
        De figuur: de regel uit Hamlet, geknipt in 12 stukken, en dan in tokens door 100 merge rules die de course leerde binnen de stukken van de eerste 20.000 tekens van Tiny Shakespeare. ␣ is een spatie. Geen token bevat het eind van het ene woord en het begin van een ander.
        """),
        C.pieces_picture({"title": "Eerst stukken, dan merges", "pieces": "{k} stukken van het patroon", "tokens": "{k} tokens na 100 merge rules binnen de stukken",
                          "alt": "To be, or not to be, that is the question in 12 stukken: To, spatie-be, komma, spatie-or, spatie-not, spatie-to, spatie-be, komma, spatie-that, spatie-is, spatie-the, spatie-question. Na 100 merge rules binnen de stukken 20 tokens, zoals spatie-be, spatie-not en spatie-the; question is nog in kleine stukken."}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Byte-level BPE met pre-tokenization, $n$ merge rules:

    1. Knip de tekst in stukken: `re.findall(PATTERN, text)`, woorden met de spatie ervoor, leestekens, en witruimte.
    2. Elk stuk als lijst van zijn bytes in UTF-8: een lijst van lijsten ids.
    3. Tel de paren binnen elk stuk, nooit over twee stukken heen; het vaakst voorkomende paar krijgt het volgende nieuwe id.
    4. Voeg dat paar samen binnen elk stuk, en ga door, $n$ rondes. Om een nieuwe tekst te encoderen: knip hem op dezelfde manier, encodeer elk stuk, en zet de ids achter elkaar.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    "be, or not to be", geknipt in stukken, en de paren erbinnen geteld. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 3 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_pieces():
    import re

    _PATTERN = r" ?\w+| ?[^\w\s]+|\s+"
    _text = "be, or not to be"
    # 1. de stukken: woorden met de spatie ervoor, leestekens, en witruimte
    _pieces = re.findall(_PATTERN, _text)
    print(_pieces)
    # 2. elk stuk als zijn bytes in UTF-8
    _chunks = [list(_p.encode("utf-8")) for _p in _pieces]
    print(_chunks)
    # 3. tel de paren binnen elk stuk
    _counts = {}
    for _ids in [list(_text.encode("utf-8"))]:  # maak af: loop in plaats daarvan over _chunks
        for _i in range(len(_ids) - 1):
            _pair = (_ids[_i], _ids[_i + 1])
            if _pair in _counts:
                _counts[_pair] = _counts[_pair] + 1
            else:
                _counts[_pair] = 1
    print(len(_counts), (101, 44) in _counts, _counts[(98, 101)])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De stukken zijn `['be', ',', ' or', ' not', ' to', ' be']`, en hun bytes `[[98, 101], [44], [32, 111, 114], [32, 110, 111, 116], [32, 116, 111], [32, 98, 101]]`. Zoals de cel nu is, telt de loop de bytes van de hele tekst als één stuk: 14 verschillende paren, waaronder (101, 44), `e` en een komma (`True`), en `be` (98, 101) twee keer. Afgemaakt telt hij binnen de stukken: 9 verschillende paren, (101, 44) is weg (`False`), en `be` komt nog steeds twee keer voor.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_findall():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Wat het patroon knipt
    """)
    return


@app.cell(hide_code=True)
def choice_findall_cell():
    choice_findall = L.choice(
        "findall",
        question="Met `PATTERN = r\" ?\\w+| ?[^\\w\\s]+|\\s+\"`, wat geeft `re.findall(PATTERN, \"Wait... what?!\")`?",
        options={
            "right": "`[\"Wait\", \"...\", \" what\", \"?!\"]`",
            "each": "`[\"Wait\", \".\", \".\", \".\", \" what\", \"?\", \"!\"]`",
            "stuck": "`[\"Wait...\", \" what?!\"]`",
        },
        correct="right",
        why={
            "each": "`+` neemt één of meer, zoveel als er zijn: de drie puntjes zijn één reeks leestekens, en `?!` ook.",
            "stuck": "`\\w` neemt alleen letters, cijfers en `_`, dus een woord stopt bij het eerste leesteken; de leestekens zijn een eigen stuk.",
            "right": "Een woord, een reeks leestekens, een woord met de spatie ervoor, en een reeks leestekens. BPE zal `Wait` nooit met zijn puntjes samenvoegen.",
        },
    )
    choice_findall
    return (choice_findall,)


@app.cell(hide_code=True)
def text_exercise_counts():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Tellen binnen de stukken

        Maak `chunk_pair_counts(chunks)` af: de functie geeft een dictionary met hoe vaak elk paar naburige ids voorkomt binnen de stukken van `chunks`, een lijst van lijsten ids, en nooit over twee stukken heen: stap 3 van het recept. De check roept `chunk_pair_counts([[104, 105], [32, 104, 105], [33]])` aan, de stukken "hi", " hi" en "!", wat `{(104, 105): 2, (32, 104): 1}` moet geven, en probeert andere stukken. Als hij goed is, bewaart de check hem in je project, als `work/projects/llm/chunk_pairs.py`.
        """),
        L.task("Laat `chunk_pair_counts` de paren binnen elk stuk tellen, en geen enkel paar over twee stukken heen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_counts():
    box_counts = L.code_box("chunk-pair-counts", lines=11)
    box_counts
    return (box_counts,)


@app.cell(hide_code=True)
def check_counts(box_counts):
    result_counts = L.check("chunk-pair-counts", box_counts.value)
    result_counts
    return (result_counts,)


@app.cell(hide_code=True)
def hints_counts():
    L.hints(
        "chunk-pair-counts",
        [
            "Binnen de loop over de stukken een tweede loop over de paren van één stuk: `for i in range(len(ids) - 1):`, zoals in je `pair_counts` uit [Byte pair encoding: bouw je eigen tokenizer](/course/open/tok-bpe?lang=nl).",
            "Stap 3 van het afgemaakte uitgewerkte voorbeeld, met `chunks`, `ids` en `counts` in plaats van `_chunks`, `_ids` en `_counts`.",
        ],
        why="De paren worden stuk voor stuk geteld, allemaal in dezelfde dictionary: het laatste id van het ene stuk en het eerste id van het volgende zijn in een loop over één stuk nooit buren.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_train(result_counts):
    result_counts  # opnieuw gemaakt na de check van oefening 2, die je chunk_pair_counts bewaart
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. BPE binnen de stukken

        Maak `train_pre_bpe(text, n)` af: je `train_byte_bpe` uit [Byte-level BPE](/course/open/tok-byte-level?lang=nl), op de stukken van `PATTERN`. De eerste regels importeren `merge_ids` uit je `byte_bpe.py` en je `chunk_pair_counts`; het zoeken naar het beste paar en het nieuwe id zijn gegeven. Vervang de twee `...` door stap 1, 2 en 4 van het recept. De check roept `train_pre_bpe("the cat, the hat", 3)` aan, wat `{(116, 104): 256, (256, 101): 257, (97, 116): 258}` moet geven, en probeert andere teksten. Als hij goed is, bewaart de check hem als `work/projects/llm/pre_bpe.py`.
        """),
        L.course_versions(),
        L.task("Laat `train_pre_bpe` merge rules leren binnen de stukken van een tekst."),
    ])
    return


@app.cell(hide_code=True)
def exercise_train():
    box_train = L.code_box("train-pre-bpe", lines=22)
    box_train
    return (box_train,)


@app.cell(hide_code=True)
def check_train(box_train, result_counts):
    result_counts  # check again when exercise 2 saves your chunk_pair_counts
    result_train = L.check("train-pre-bpe", box_train.value)
    result_train
    return (result_train,)


@app.cell(hide_code=True)
def hints_train():
    L.hints(
        "train-pre-bpe",
        [
            "De eerste `...` is stap 1 en 2 van het uitgewerkte voorbeeld in één comprehension: voor elk stuk van `re.findall(PATTERN, text)` de lijst van zijn bytes.",
            "De tweede `...` is ook een comprehension: `merge_ids(ids, best, new_id)` voor elke `ids` in `chunks`.",
        ],
        why="Elk stuk wordt apart samengevoegd, dus een paar van het eind van het ene stuk en het begin van het volgende wordt nooit samengevoegd, omdat het nooit geteld is.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel hieronder gebruikt je project. Hij traint je BPE met pre-tokenization op de eerste 20.000 tekens van Tiny Shakespeare, 100 merge rules, en encodeert daarmee de regel uit Hamlet. Dan telt hij hoeveel van de 100 nieuwe tokens van het ene woord in het volgende lopen, en vergelijkt dat met de byte-level BPE van de course zonder pre-tokenization, op dezelfde tekst. Het werkt zodra oefening 2 en 3 af zijn, en duurt ongeveer een seconde.
    """)
    return


@app.cell(hide_code=True)
def demo_pieces(result_counts, result_train):
    result_counts, result_train  # run again when exercise 2 or 3 saves your module
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Maak eerst oefening 2 en 3 af: dan staan je `chunk_pair_counts` en je `train_pre_bpe` in je project, en gebruikt deze cel ze.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*Je BPE duurt hier te lang: meer dan een minuut. Doet je `train_pre_bpe` meer werk dan nodig is, zoals een loop die te vaak draait?*",
            "bpe": "*`train_pre_bpe`, `chunk_pair_counts` of `merge_ids` in `work/projects/llm/` stopt met een fout. Heb je het bestand veranderd? Verwijder het, en check oefening 2 of 3 opnieuw: de check bewaart je code dan opnieuw.*",
        }[_result])
    else:
        _first = ", ".join(f"`{C.shown(t)}`" for t in _result["first"])
        _line = " ".join(f"`{C.shown(t)}`" for t in _result["line"])
        _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de melding bij oefening 3 zegt."
        _out = mo.md(f"""Je eerste 12 nieuwe tokens: {_first}. ␣ is een spatie.

De regel wordt {len(_result['line'])} tokens: {_line}. Van je 100 nieuwe tokens lopen er **{_result['crossing']}** van het ene woord in het volgende; zonder pre-tokenization waren dat er {_result['plain_crossing']} van de 100, zoals `e␣` en `,␣`. Op de volgende 5.000 tekens bevat een id gemiddeld {C.number(_result['per_id'], 2, L.lang)} bytes, ongeveer evenveel als zonder pre-tokenization: evenveel tokens, maar tokens die woorden of delen van woorden zijn.{_who}""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** Je byte-level BPE uit [Byte-level BPE](/course/open/tok-byte-level?lang=nl), nu binnen de stukken van een reguliere expressie: dezelfde stappen als de tokenizer van GPT-2 uit [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl), waar " surprisingly" één token was met zijn spatie ervoor. Je project `llm` heeft nu `chunk_pairs.py` en `pre_bpe.py`. Hierna: *Pre-tokenization en special tokens*, tokens zoals end-of-text die nooit uit gewone tekst komen, en aangeven waar het ene document stopt en het volgende begint.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_findall, result_counts, result_train):
    L.wrap_up(choice_findall, result_counts, result_train)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
