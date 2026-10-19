# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les tok-special-tokens. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("tok-special-tokens", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Pre-tokenization](/course/open/tok-pre-tokenization?lang=nl) en [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_cross():
    quick_cross = L.choice(
        "quick-cross",
        question="Kan BPE met pre-tokenization een merge rule leren voor een komma gevolgd door een spatie?",
        options={
            "right": "Nee: de spatie gaat mee met het volgende stuk",
            "often": "Ja: zoals elk paar, als het vaak genoeg voorkomt",
            "lines": "Alleen in een tekst zonder nieuwe regels",
        },
        correct="right",
        why={
            "often": "BPE telt alleen paren binnen een stuk, en de komma en de spatie zitten in twee stukken: `,` en ` woord`. Zie [Pre-tokenization](/course/open/tok-pre-tokenization?lang=nl).",
            "lines": "Nieuwe regels doen er hier niet toe: in elke tekst gaat de spatie mee met het volgende stuk. Zie [Pre-tokenization](/course/open/tok-pre-tokenization?lang=nl).",
            "right": "Vandaag gaat het knippen een stap verder: bij elke end-of-text, vóór de stukken.",
        },
    )
    quick_cross
    return (quick_cross,)


@app.cell(hide_code=True)
def quick_check_join():
    quick_join = L.choice(
        "quick-join",
        question="Wat geeft `\"-\".join([\"a\", \"b\", \"c\"])`?",
        options={
            "around": "`\"-a-b-c-\"`",
            "right": "`\"a-b-c\"`",
            "end": "`\"abc-\"`",
        },
        correct="right",
        why={
            "around": "De tekst voor `.join` komt tussen de delen, niet voor het eerste of na het laatste. Zie [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl).",
            "end": "De tekst voor `.join` komt tussen elke twee delen. Zie [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl).",
            "right": "Vandaag zet `.join` weer een end-of-text tussen documenten, als ids tekst worden.",
        },
    )
    quick_join
    return (quick_join,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Een language model traint op veel documenten, achter elkaar gezet in één lange rij tokens: toneelstukken, brieven, webpagina's. Zonder teken ertussen leert het model dat een nieuw document het vorige voortzet, en leert het nooit waar een tekst eindigt. GPT-2 zet een **special token** tussen twee opeenvolgende documenten: end-of-text, geschreven als `<|endoftext|>`. Het is één token met een eigen id, en gewone tekst kan het nooit maken. Als het model later end-of-text schrijft, is zijn tekst af. In deze projectstap krijgt je tokenizer end-of-text, en daarmee is hij af.
        """),
        draw.project_steps(["Letters en paren tellen", "Een bigram-model door te tellen", "Hetzelfde model, geleerd door een neural network",
                            "Je eigen tokenizer", "Je eigen GPT", "Een klein LLM pretrainen", "Het LLM instructies en voorkeuren laten volgen"],
                           3, here_label="← vandaag", title="Waar je project staat: je eigen LLM"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_plain():
    guess_plain = L.guess(
        "as-text",
        question="Je tokenizer met 100 merge rules uit Shakespeare krijgt de 13 tekens `<|endoftext|>` als gewone tekst, zonder speciale behandeling. Hoeveel tokens worden ze volgens jou?",
        options={
            "one": "1: het is één woord",
            "three": "3: `<|`, `endoftext` en `|>`",
            "right": "12: bijna één per teken",
        },
        reveal="**12.** Pre-tokenization knipt het in `<|`, `endoftext` en `|>`, en daarin worden alleen `e` en `n` samengevoegd, tot `en`. Dus 13 tekens zijn 12 tokens, en geen ervan betekent: hier eindigt een tekst. Daarom heeft end-of-text een eigen id nodig, en mag het niet uit de letters komen.",
    )
    guess_plain
    return (guess_plain,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        Een **special token** is een token met een eigen id dat gewone tekst nooit wordt. End-of-text krijgt het eerste id dat nog vrij is, na de 256 bytes en de merge rules: `256 + len(merges)`. In GPT-2, met zijn 50.000 regels, is dat 50.256, het laatste van zijn 50.257 ids.

        Om te encoderen knipt de tokenizer de tekst eerst bij elke end-of-text. `text.split(END)` geeft de delen ertussen, als lijst: `"a<|endoftext|>b".split(END)` geeft `["a", "b"]`. Elk deel wordt geëncodeerd zoals eerder, met pre-tokenization en de merge rules, en het id van end-of-text komt tussen de delen. Decoderen gaat andersom: de ids tussen twee ids van end-of-text worden tekst via hun bytes, en `END.join(texts)` zet end-of-text weer tussen de teksten.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent | In code |
        |---|---|---|
        | **special token** | een token met een eigen id dat gewone tekst nooit wordt, zoals end-of-text | `END` |
        | end-of-text | het special token tussen twee documenten; als een model het schrijft, is zijn tekst af | `end_id` |
        | `.split(sep)` | de delen van een tekst tussen elke `sep`, als lijst | `"a-b".split("-")` |
        """),
        L.watch_out("End-of-text is geen byte en geen merge rule, dus het heeft geen bytes: decode moet zijn id afhandelen voordat het bytes opzoekt. En een tokenizer die elke `<|endoftext|>` in een tekst het special token maakt, laat iedereen die die 13 tekens typt een document beëindigen. Daarom stopt tiktoken uit [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl) met een fout bij een tekst met `<|endoftext|>` erin, tenzij het programma dat special token zelf toestaat."),
        mo.md(r"""
        De figuur: twee documenten zoals ze de training in gaan, elk in stukken geknipt, en end-of-text ertussen als één eigen token, id 356 na 100 merge rules. ␣ is een spatie.
        """),
        C.special_picture({"title": "Twee documenten, achter elkaar", "first": "document 1, in stukken", "end": "end-of-text: één token, id 356", "second": "document 2, in stukken",
                           "alt": "Document 1, To be, or not to be, in stukken; dan end-of-text als één token met id 356; dan document 2, Één ijsje met een emoji van een ijsje, in stukken."}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Een tokenizer met end-of-text, voor merge rules `merges`:

    1. End-of-text krijgt het eerste vrije id: `end_id = 256 + len(merges)`.
    2. Om te encoderen: knip de tekst bij elke end-of-text met `text.split(END)`; elk deel in stukken, elk stuk als bytes met de merge rules op volgorde toegepast; het id van end-of-text tussen de delen.
    3. Om te decoderen: knip de ids bij elke `end_id`; elk deel terug naar tekst via zijn bytes; `END.join` tussen de teksten.
    4. In de training data staat end-of-text tussen twee opeenvolgende documenten.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Twee korte documenten met end-of-text ertussen, in delen geknipt, zonder merge rules, zodat elk deel gewoon zijn bytes is. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 2 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_end():
    _END = "<|endoftext|>"
    _merges = {}
    # 1. end-of-text krijgt het eerste vrije id: zonder merge rules 256
    _end_id = 256 + len(_merges)
    # 2. de delen tussen de end-of-text-tokens, elk als zijn bytes, en het id van end-of-text tussen twee delen
    _parts = ("Hi." + _END + "Bye.").split(_END)
    print(_parts)
    _ids = []
    for _k in range(len(_parts)):
        # maak af: als _k > 0, voeg dan eerst _end_id toe aan _ids
        _ids = _ids + list(_parts[_k].encode("utf-8"))
    print(_ids)
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    De delen zijn `['Hi.', 'Bye.']`. Zoals de cel nu is, zijn de ids `[72, 105, 46, 66, 121, 101, 46]`: de bytes van beide documenten achter elkaar, en niemand kan zien dat het er twee waren. Afgemaakt zijn ze `[72, 105, 46, 256, 66, 121, 101, 46]`: end-of-text is id 256, het eerste na de bytes, omdat er hier geen merge rules zijn.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_end():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. Een vrij id
    """)
    return


@app.cell(hide_code=True)
def choice_end_cell():
    choice_end = L.choice(
        "end-id",
        question="Je tokenizer heeft de 256 bytes en 100 merge rules, ids 0 tot en met 355, en end-of-text krijgt 356. Waarom niet 33, dat ook geen merge rule gebruikt?",
        options={
            "free": "Dat kan: elk getal dat geen merge rule gebruikt, is goed",
            "order": "Ids moeten uitgedeeld worden in de volgorde waarin de tokens gemaakt zijn",
            "right": "33 is al een byte, `!`: decode kan ze dan niet uit elkaar houden",
        },
        correct="right",
        why={
            "free": "Elk id van 0 tot en met 255 is een byte, gebruikt of niet; end-of-text heeft een id nodig dat niets anders betekent.",
            "order": "De volgorde alleen doet er niet toe; waar het om gaat, is dat elk id precies één ding betekent.",
            "right": "Elk id moet precies één ding betekenen. Met 33 voor end-of-text zou elke `!` in een tekst die tekst beëindigen.",
        },
    )
    choice_end
    return (choice_end,)


@app.cell(hide_code=True)
def text_exercise_encode():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Encoderen

        Maak `encode(text, merges)` af: de functie geeft de ids van een tekst, met het id van end-of-text overal waar `<|endoftext|>` staat, stap 1 en 2 van het recept. `encode_part(part, merges)` is gegeven: de ids van één deel, in stukken geknipt en samengevoegd zoals in je `train_pre_bpe`, met `merge_ids` uit je `byte_bpe.py` en `PATTERN` uit je `pre_bpe.py`. Vervang de twee `...`: het id van end-of-text, en de delen met hun ids. De check roept `encode("hi!<|endoftext|>hi", {(104, 105): 256})` aan, wat `[256, 33, 257, 256]` moet geven, en probeert andere teksten en regels. Als hij goed is, bewaart de check hem in je project, als `work/projects/llm/encode.py`.
        """),
        L.task("Laat `encode` het id van end-of-text tussen de ids van de delen van een tekst zetten."),
    ])
    return


@app.cell(hide_code=True)
def exercise_encode():
    box_encode = L.code_box("encode", lines=24)
    box_encode
    return (box_encode,)


@app.cell(hide_code=True)
def check_encode(box_encode):
    result_encode = L.check("encode", box_encode.value)
    result_encode
    return (result_encode,)


@app.cell(hide_code=True)
def hints_encode():
    L.hints(
        "encode",
        [
            "De eerste `...` is stap 1 van het uitgewerkte voorbeeld, met `merges`.",
            "De tweede `...` is stap 2 van het afgemaakte uitgewerkte voorbeeld, met `text` en `END`, en `encode_part(parts[k], merges)` voor de ids van een deel.",
        ],
        why="`split` geeft één deel meer dan er end-of-text-tokens in de tekst staan, dus het id komt vóór elk deel behalve het eerste. Twee daarvan achter elkaar geven een leeg deel ertussen, dat geen ids heeft.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_decode():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Decoderen

        Maak `decode(ids, merges)` af: de functie geeft de tekst van een lijst ids, met `<|endoftext|>` overal waar het id van end-of-text staat, stap 3 van het recept. De eerste regel importeert `decode` uit je `byte_decode.py`, als `decode_bytes`; `import ... as ...` geeft wat het importeert een andere naam. De loop over de ids en het laatste deel zijn gegeven. De check roept `decode([256, 33, 257, 256], {(104, 105): 256})` aan, wat `"hi!<|endoftext|>hi"` moet geven, en probeert andere ids. Als hij goed is, bewaart de check hem als `work/projects/llm/decode.py`.
        """),
        L.task("Laat `decode` de ids tussen de end-of-text-tokens weer tekst maken, met end-of-text ertussen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_decode():
    box_decode = L.code_box("decode", lines=16)
    box_decode
    return (box_decode,)


@app.cell(hide_code=True)
def check_decode(box_decode):
    result_decode = L.check("decode", box_decode.value)
    result_decode
    return (result_decode,)


@app.cell(hide_code=True)
def hints_decode():
    L.hints(
        "decode",
        [
            "Een `if` op `i == end_id`: dan is het deel tot nu toe af; anders (`else`) gaat `i` in het deel met `part.append(i)`.",
            "Een afgerond deel gaat naar `texts` op dezelfde manier als het laatste deel, in de regel na de loop; daarna begint `part` weer als lege lijst.",
        ],
        why="End-of-text heeft geen bytes, dus zijn id gaat nooit in een deel: het sluit er alleen een af. `END.join` zet het terug als tekst, tussen de delen.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel hieronder gebruikt je hele tokenizer: je `train_pre_bpe` leert 100 merge rules uit de eerste 20.000 tekens van Tiny Shakespeare, je `encode` maakt van twee korte documenten met end-of-text ertussen ids, en je `decode` maakt van die ids weer tekst. Het werkt zodra oefening 2 en 3 af zijn, en duurt ongeveer een seconde.
    """)
    return


@app.cell(hide_code=True)
def demo_tokenizer(result_encode, result_decode):
    result_encode, result_decode  # run again when exercise 2 or 3 saves your module
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Maak eerst oefening 2 en 3 af: dan staan je `encode` en je `decode` in je project, en gebruikt deze cel ze.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*Je tokenizer duurt hier te lang: meer dan een minuut. Doet je `train_pre_bpe` of je `encode` meer werk dan nodig is, zoals een loop die te vaak draait?*",
            "tokenizer": "*`train_pre_bpe`, `encode` of `decode` in `work/projects/llm/` stopt met een fout. Heb je het bestand veranderd? Verwijder het, en check zijn oefening opnieuw: de check bewaart je code dan opnieuw.*",
        }[_result])
    else:
        _back = "Je `decode` geeft de twee documenten terug, met end-of-text ertussen." if _result["back"] else "Maar je `decode` geeft de twee documenten niet terug."
        _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de melding bij *Waar je project staat* zegt."
        _out = mo.md(f"""\"{C.DOCS[0]}\" en \"{C.DOCS[1]}\" worden {len(_result['ids'])} ids: `{_result['ids']}`. End-of-text is het ene id {_result['end_id']}, tussen de twee documenten. {_back}

Als gewone tekst zouden de 13 tekens van `<|endoftext|>` {_result['plain']} ids zijn. Je tokenizer is af: hij leert merge rules, knipt tekst in stukken, kan elke tekst in elke taal aan, en geeft aan waar documenten eindigen.{_who}""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De stukken uit [Pre-tokenization](/course/open/tok-pre-tokenization?lang=nl), de bytes en merge rules uit [Byte-level BPE](/course/open/tok-byte-level?lang=nl), en nu end-of-text tussen documenten: de onderdelen van de tokenizer van GPT-2, die [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl) kant-en-klaar gebruikte. Je project `llm` heeft nu `encode.py` en `decode.py`. Hierna: *Je GPT in elkaar zetten*, dat embeddings, een stapel van je transformer blocks en een output head samenvoegt tot je eigen GPT.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_end, result_encode, result_decode):
    L.wrap_up(choice_end, result_encode, result_decode)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
