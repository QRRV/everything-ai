# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les tok-byte-level. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.lesson import lesson

    L = lesson("tok-byte-level", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Byte pair encoding: bouw je eigen tokenizer](/course/open/tok-bpe?lang=nl) en [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_rules():
    quick_rules = L.choice(
        "quick-rules",
        question="Een BPE leerde de merge rules (`l`, `o`) en daarna (`lo`, `w`). In hoeveel tokens knipt hij \"lowly\"?",
        options={
            "two": "2: `low` en `ly`",
            "right": "3: `low`, `l` en `y`",
            "four": "4: `lo`, `w`, `l` en `y`",
        },
        correct="right",
        why={
            "two": "Er is geen regel (`l`, `y`), dus `l` en `y` blijven los. Zie [Byte pair encoding: bouw je eigen tokenizer](/course/open/tok-bpe?lang=nl).",
            "four": "De regels gaan op volgorde: nadat (`l`, `o`) `lo` maakt, maakt de regel (`lo`, `w`) er `low` van. Zie [Byte pair encoding: bouw je eigen tokenizer](/course/open/tok-bpe?lang=nl).",
            "right": "Vandaag werken dezelfde regels op bytes in plaats van op tekens.",
        },
    )
    quick_rules
    return (quick_rules,)


@app.cell(hide_code=True)
def quick_check_byte():
    quick_byte = L.choice(
        "quick-byte",
        question="Welke gehele getallen kan een byte zijn?",
        options={
            "ascii": "0 tot en met 127: één voor elk code point dat één byte neemt",
            "code": "Elk code point, zoals 8364 voor €",
            "right": "0 tot en met 255: het zijn er 256",
        },
        correct="right",
        why={
            "ascii": "Een code point onder 128 neemt één byte met hetzelfde getal, maar een teken zoals é gebruikt bytes boven 127. Zie [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl).",
            "code": "Een code point kan ver boven 255 liggen; UTF-8 schrijft het dan als twee, drie of vier bytes. Zie [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl).",
            "right": "Maar 256 stuks, voor elke tekst in elke taal: daarop bouwt de tokenizer van vandaag.",
        },
    )
    quick_byte
    return (quick_byte,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Je BPE uit [Byte pair encoding: bouw je eigen tokenizer](/course/open/tok-bpe?lang=nl) begint bij de tekens die hij zag. Een tekst met een teken dat hij nooit zag, een emoji, een Chinees teken of een letter met een accent die Shakespeare nooit gebruikte, heeft daar geen token voor. Unicode heeft meer dan 150.000 tekens, te veel voor een vocabulary. GPT-2 werkt daarom op de bytes van UTF-8: elke tekst is bytes, en daarvan zijn er maar 256. Zet alle 256 vanaf het begin in de vocabulary, en elke tekst kan getokeniseerd worden. In deze projectstap gaat je tokenizer over op bytes.
        """),
        draw.project_steps(["Letters en paren tellen", "Een bigram-model door te tellen", "Hetzelfde model, geleerd door een neural network",
                            "Je eigen tokenizer", "Je eigen GPT", "Een klein LLM pretrainen", "Het LLM instructies en voorkeuren laten volgen"],
                           3, here_label="← vandaag", title="Waar je project staat: je eigen LLM"),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def predict_bytes():
    guess_bytes = L.guess(
        "bytes",
        question="De tekst \"café 🙂\" heeft 6 tekens. Hoeveel tokens is deze tekst voor een byte-level BPE, vóór de eerste merge?",
        options={
            "six": "6: één voor elk teken",
            "seven": "7: é heeft er één extra nodig",
            "right": "10: é neemt twee bytes, en de emoji vier",
        },
        reveal="**10.** In UTF-8 nemen `c`, `a`, `f` en de spatie elk één byte; `é` neemt er twee, 195 en 169, en 🙂 vier, 240, 159, 153 en 130. Een byte-level BPE begint met één id voor elke byte, en voegt van daaruit samen.",
    )
    guess_bytes
    return (guess_bytes,)


@app.cell(hide_code=True)
def text_idea():
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Byte-level BPE** is BPE op de bytes van een tekst in UTF-8, in plaats van op zijn tekens. De eerste tokens zijn de bytes, ids 0 tot en met 255, en alle 256 staan vanaf het begin in de vocabulary, of de trainingstekst ze nu heeft of niet. Elke merge rule krijgt een nieuw id: 256 voor de eerste, 257 voor de volgende, enzovoort. De merges zijn dus een dictionary, van een paar ids naar zijn nieuwe id: `{(99, 97): 256}` betekent dat `c` gevolgd door `a` id 256 wordt. GPT-2 heeft de 256 bytes, 50.000 merges en één special token: 50.257 tokens.

        Om ids weer tekst te maken, heeft elk id zijn bytes nodig. Een id onder 256 is die byte: `bytes([i])` maakt een bytes-waarde van een lijst getallen. Een nieuw id is de bytes van de twee helften van zijn paar, na elkaar, en `+` zet twee bytes-waarden achter elkaar. Daarna worden alle bytes op een rij gedecodeerd als UTF-8. Eén token is niet altijd een heel teken: een id kan de losse eerste byte van é zijn. `.decode("utf-8", errors="replace")` zet daar dan � neer, in plaats van te stoppen met een fout.

        ### Nieuwe woorden en tekens

        | Woord of teken | Wat het betekent |
        |---|---|
        | **byte-level BPE** | BPE op de bytes van UTF-8; de vocabulary begint met alle 256 bytes, dus elke tekst kan getokeniseerd worden |
        | `bytes([...])` | een bytes-waarde gemaakt van een lijst getallen van 0 tot en met 255: `bytes([104, 105])` |
        | `b""` | een lege bytes-waarde; `+` zet bytes-waarden achter elkaar: `data + vocab[i]` |
        | `errors="replace"` | in `.decode`: bytes die geen heel teken zijn, worden �, in plaats van een fout |
        """),
        L.watch_out("Een nieuw id staat voor een paar ids, geen paar bytes: (256, 102) betekent de bytes van 256, dan die van 102. Bouw de bytes op in de volgorde waarin de regels geleerd zijn, en tel nooit twee ids op: 256 + 102 is gewoon een ander getal."),
        mo.md(r"""
        De figuur: "café 🙂" als 6 tekens en als 10 bytes. `c`, `a`, `f` en de spatie nemen elk één byte, hetzelfde getal als hun code point; `é` neemt twee bytes, en de emoji vier.
        """),
        C.bytes_picture({"title": "Zes tekens, tien bytes", "chars": "{k} tekens", "word": "café en de spatie: {k} bytes", "emoji": "de emoji: {k} bytes",
                         "alt": "café, een spatie en een emoji: 6 tekens; café en de spatie zijn de 6 bytes 99, 97, 102, 195, 169 en 32 in UTF-8, en de emoji de 4 bytes 240, 159, 153 en 130."}),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Byte-level BPE met $n$ merge rules:

    1. De tekst als bytes in UTF-8, als getallen: `list(text.encode("utf-8"))`, de eerste ids.
    2. Elke ronde krijgt het vaakst voorkomende paar ids het volgende nieuwe id, `256 + step`: een merge rule in een dictionary, paar → nieuw id. Voeg het overal samen, en ga door, $n$ rondes.
    3. Om een nieuwe tekst te encoderen: zijn bytes, met elke merge rule toegepast in de volgorde waarin hij geleerd is.
    4. Om te decoderen: de bytes van elk id, de byte zelf onder 256 en de bytes van beide helften van zijn paar voor een nieuw id; allemaal op een rij, terug naar tekst in UTF-8 met `errors="replace"`.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    "café café" als bytes, één merge rule, en terug. `_merge_ids` is je `merge` uit de vorige les, met een nieuw id in plaats van twee strings aan elkaar. Het commentaar met `#` zegt welke stap van het recept elk deel is. Stap 4 is niet af: voer de cel uit zoals hij is, maak hem dan af waar het commentaar het zegt, en voer hem opnieuw uit.
    """)
    return


@app.cell
def example_bytes():
    def _merge_ids(ids, pair, new_id):
        out = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i = i + 2
            else:
                out.append(ids[i])
                i = i + 1
        return out

    # 1. de bytes van de tekst in UTF-8: de eerste ids
    _ids = list("café café".encode("utf-8"))
    print(_ids)
    # 2. het vaakst voorkomende paar, (99, 97) voor "ca", krijgt het nieuwe id 256
    _merges = {(99, 97): 256}
    _ids = _merge_ids(_ids, (99, 97), _merges[(99, 97)])
    print(_ids)
    # 4. terug naar tekst: de bytes van elk id, allemaal op een rij
    _vocab = {}
    for _b in range(256):
        _vocab[_b] = bytes([_b])
    _vocab[256] = b""  # maak af: _vocab[99] + _vocab[97]
    _data = b""
    for _id in _ids:
        _data = _data + _vocab[_id]
    print(_data.decode("utf-8", errors="replace"))
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    9 tekens zijn 11 bytes, `[99, 97, 102, 195, 169, 32, 99, 97, 102, 195, 169]`: é is 195 en 169. Na de merge zijn er 9 ids, `[256, 102, 195, 169, 32, 256, 102, 195, 169]`. Zoals de cel nu is, heeft id 256 geen bytes, dus de tekst komt terug als `fé fé`. Afgemaakt is 256 de bytes van 99 en 97, en is de tekst weer `café café`.
    """)
    return


@app.cell(hide_code=True)
def text_exercise_vocab():
    mo.md(r"""
    ## Zelf doen

    ### Oefening 1. De grootte van de vocabulary
    """)
    return


@app.cell(hide_code=True)
def choice_vocab_cell():
    choice_vocab = L.choice(
        "vocab-size",
        question="Een byte-level BPE leerde 1.000 merge rules uit Shakespeare, die 65 verschillende tekens gebruikt. Hoeveel ids heeft zijn vocabulary?",
        options={
            "right": "1.256: de 256 bytes en één per regel",
            "rules": "1.000: één voor elke merge rule",
            "seen": "1.065: de 65 tekens van Shakespeare en één per regel",
        },
        correct="right",
        why={
            "rules": "De bytes zijn ook tokens: elke tekst begint als bytes, en een byte die in geen merge zit, blijft een eigen id.",
            "seen": "Een byte-level BPE houdt alle 256 bytes, ook die Shakespeare nooit gebruikte, dus hij kan elke tekst aan.",
            "right": "Ids 0 tot en met 255 voor de bytes, 256 tot en met 1.255 voor de regels. Zo komt GPT-2 op 50.257: 256 bytes, 50.000 regels, en één special token.",
        },
    )
    choice_vocab
    return (choice_vocab,)


@app.cell(hide_code=True)
def text_exercise_train():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. BPE op bytes

        Maak `train_byte_bpe(text, n)` af: je `train_bpe` uit [Byte pair encoding: bouw je eigen tokenizer](/course/open/tok-bpe?lang=nl), op bytes. De ids beginnen als de bytes van de tekst, en elke merge rule krijgt een nieuw id vanaf 256; de functie geeft de regels als dictionary, paar → nieuw id. `merge_ids` en het zoeken naar het beste paar zijn gegeven; de eerste regel importeert `pair_counts` uit je `bpe.py`. De check roept `train_byte_bpe("naïve naïve", 4)` aan, wat `{(110, 97): 256, (256, 195): 257, (257, 175): 258, (258, 118): 259}` moet geven, en probeert andere teksten. Als hij goed is, bewaart de check hem in je project, als `work/projects/llm/byte_bpe.py`.
        """),
        L.task("Laat `train_byte_bpe` merge rules leren op de bytes van een tekst, met nieuwe ids vanaf 256."),
    ])
    return


@app.cell(hide_code=True)
def exercise_train():
    box_train = L.code_box("train-byte-bpe", lines=30)
    box_train
    return (box_train,)


@app.cell(hide_code=True)
def check_train(box_train):
    result_train = L.check("train-byte-bpe", box_train.value)
    result_train
    return (result_train,)


@app.cell(hide_code=True)
def hints_train():
    L.hints(
        "train-byte-bpe",
        [
            "De eerste `...` is stap 1 van het uitgewerkte voorbeeld, met `text`.",
            "Drie regels in de loop: `new_id = ___ + step`, dan `merges[___] = new_id`, dan `ids = merge_ids(ids, ___, ___)`, zodat de volgende ronde de samengevoegde ids telt.",
        ],
        why="Ids 0 tot en met 255 zijn bezet door de bytes, dus het eerste nieuwe id is 256. Met het paar als key kan een tokenizer elk paar opzoeken dat hij ziet, en vinden wat het wordt.",
    )
    return


@app.cell(hide_code=True)
def text_exercise_decode():
    mo.vstack([
        mo.md(r"""
        ### Oefening 3. Terug naar tekst

        Maak `decode(ids, merges)` af: de functie geeft de tekst van een lijst ids, stap 4 van het recept. De 256 bytes in `vocab` en de loop die alle bytes op een rij zet, zijn gegeven. De check roept `decode([257, 256, 111], {(108, 108): 256, (104, 101): 257})` aan, wat `"hello"` moet geven, en probeert andere ids, ook een byte die maar een deel van een teken is. Als hij goed is, bewaart de check hem als `work/projects/llm/byte_decode.py`.
        """),
        L.task("Laat `decode` de bytes van elk nieuw id geven, en alle bytes weer tekst maken."),
    ])
    return


@app.cell(hide_code=True)
def exercise_decode():
    box_decode = L.code_box("decode", lines=14)
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
            "Stap 4 van het recept: de bytes van een nieuw id zijn die van beide helften van zijn paar, na elkaar, en `vocab` heeft de bytes van elke helft al.",
            "`vocab[___] = vocab[___] + vocab[___]`, zoals de regel die je in stap 4 van het uitgewerkte voorbeeld afmaakte; aan het eind `return data.decode(___)`, met de encoding en de `errors=` van het recept.",
        ],
        why="De merges komen in de volgorde waarin ze geleerd zijn, dus beide helften van een paar hebben hun bytes al in `vocab` als het paar langskomt.",
    )
    return


@app.cell(hide_code=True)
def text_demo():
    mo.md(r"""
    ### Wat je project nu kan

    De cel hieronder gebruikt je project. Hij traint je byte-level BPE op de eerste 20.000 tekens van Tiny Shakespeare, 100 merge rules, en stuurt twee regels door je `merge_ids` en terug door je `decode`: de regel uit Hamlet, en een Nederlandse regel met een emoji, die Shakespeare nooit schreef. Het werkt zodra oefening 2 en 3 af zijn, en duurt ongeveer een seconde.
    """)
    return


@app.cell(hide_code=True)
def demo_bytes(result_train, result_decode):
    result_train, result_decode  # run again when exercise 2 or 3 saves your module
    _result = C.demo()
    if _result is None:
        _out = mo.md("*Maak eerst oefening 2 en 3 af: dan staan je `train_byte_bpe` en je `decode` in je project, en gebruikt deze cel ze.*")
    elif isinstance(_result, str):
        _out = mo.md({
            "slow": "*Je byte-level BPE duurt hier te lang: meer dan een minuut. Doet je `train_byte_bpe` meer werk dan nodig is, zoals een loop die te vaak draait?*",
            "bpe": "*`train_byte_bpe`, `merge_ids` of `decode` in `work/projects/llm/` stopt met een fout. Heb je het bestand veranderd? Verwijder het, en check oefening 2 of 3 opnieuw: de check bewaart je code dan opnieuw.*",
        }[_result])
    else:
        _rows = []
        for _line, _r in zip(C.LINES, _result["lines"]):
            _back = "en je `decode` geeft de regel terug" if _r["back"] else "maar je `decode` geeft de regel niet terug"
            _rows.append(f"- \"{_line}\": {len(_line)} tekens, {_r['bytes']} bytes, **{len(_r['ids'])} ids**, {_back}: `{_r['ids']}`")
        _who = "" if not _result["stand_ins"] else " De versie van de course sprong in voor " + ", ".join(f"`{m}`" for m in _result["stand_ins"]) + ", zoals de melding bij *Waar je project staat* zegt."
        _out = mo.md("\n".join(_rows) + f"""

De Nederlandse regel heeft letters en een emoji die niet in Shakespeare staan. Die blijven losse bytes, zoals 240, 159, 141 en 166 voor het ijsje, maar ze kunnen getokeniseerd worden, en komen heel terug. Op 5.000 tekens waar hij niet van leerde, bevat een id gemiddeld {C.number(_result['per_id'], 2, L.lang)} bytes.{_who}""")
    _out
    return


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De bytes van UTF-8 uit [Tekst is een rij getallen](/course/open/data-text-as-numbers?lang=nl), met de BPE uit [Byte pair encoding: bouw je eigen tokenizer](/course/open/tok-bpe?lang=nl): een tokenizer die elke tekst aankan. De tokenizer van GPT-2 uit [Tokens: hoe echte LLM's tekst opknippen](/course/open/lm-tokens-outside?lang=nl) is byte-level BPE met 50.000 merge rules. Je project `llm` heeft nu `byte_bpe.py` en `byte_decode.py`. Hierna: *Pre-tokenization*, die een tekst eerst in woorden, getallen en leestekens knipt, zodat merges binnen een woord blijven.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(choice_vocab, result_train, result_decode):
    L.wrap_up(choice_vocab, result_train, result_decode)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
