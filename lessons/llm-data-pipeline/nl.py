# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
# Les llm-data-pipeline. Route: course/route/lessen/taal.yaml. Checks: checks.py.

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup(hide_code=True):
    import marimo as mo
    from everything_ai import draw
    from everything_ai.check import run_code
    from everything_ai.lesson import lesson

    L = lesson("llm-data-pipeline", "nl")
    C = L.checks


@app.cell(hide_code=True)
def header():
    L.header()
    return


@app.cell(hide_code=True)
def text_quick_check():
    mo.md(r"""
    ### Instapcheck

    Twee korte vragen over [Pre-tokenization en special tokens](/course/open/tok-special-tokens?lang=nl) en [Datasets en DataLoaders](/course/open/torch-dataloader?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def quick_check_end():
    quick_end = L.choice(
        "quick-end",
        question="Duizenden verhalen gaan in één lange lijst token-ids, met de end-of-text-id tussen elke twee. Wat leert een GPT van die id?",
        options={
            "skip": "Niets: de trainer slaat hem over, want het is geen tekst",
            "right": "Dat een verhaal daar kan eindigen, en dat het volgende opnieuw begint",
            "join": "Dat het volgende verhaal verdergaat waar het vorige ophield, als één lang verhaal",
        },
        correct="right",
        why={
            "skip": "De end-of-text-id is een token zoals elk ander in de training data: de GPT leert hem te voorspellen, en wat erna komt. Zie [Pre-tokenization en special tokens](/course/open/tok-special-tokens?lang=nl).",
            "join": "Zonder de id zou een venster over twee verhalen eruitzien als één verhaal. Met de id ziet de GPT waar er een eindigt. Zie [Pre-tokenization en special tokens](/course/open/tok-special-tokens?lang=nl).",
            "right": "Daarom gaan de verhalen van deze les als één lijst ids de bestanden in, met end-of-text ertussen.",
        },
    )
    quick_end
    return (quick_end,)


@app.cell(hide_code=True)
def quick_check_len():
    quick_len = L.choice(
        "quick-len",
        question="De `__len__` van een Dataset geeft 1.000. Hoeveel mini-batches geeft `DataLoader(ds, batch_size=64)` in één epoch?",
        options={
            "right": "16: 15 van 64 en één van 40",
            "fifteen": "15: de laatste 40 voorbeelden vallen weg",
            "thousand": "1.000: één voor elk voorbeeld",
        },
        correct="right",
        why={
            "fifteen": "Een DataLoader houdt de laatste, kleinere mini-batch, tenzij je vraagt hem weg te laten. Zie [Datasets en DataLoaders](/course/open/torch-dataloader?lang=nl).",
            "thousand": "`__getitem__` geeft één voorbeeld; de DataLoader zet er 64 in een mini-batch. Zie [Datasets en DataLoaders](/course/open/torch-dataloader?lang=nl).",
            "right": "In deze les is `__len__` het aantal vensters in de bestanden, en leest `__getitem__` één venster.",
        },
    )
    quick_len
    return (quick_len,)


@app.cell(hide_code=True)
def text_why():
    mo.vstack([
        mo.md(r"""
        ## Waarom

        Tot nu toe codeerde elke run van [Project: je GPT op je eigen tokens](/course/open/gpt-train-bpe?lang=nl) alle verhalen opnieuw, en hield hij al hun ids in het geheugen, als een lijst van Python. Voor 5 MB gaat dat. nanoGPT, een bekende kleine codebase om GPT-2 te trainen, bereidt OpenWebText voor, 8 miljoen webpagina's: hun token-ids voor GPT-2, zo'n 9 miljard, nemen als getallen van 2 bytes ongeveer 17 GB in. Met de snelheid van de tokenizer van de course, zo'n 400.000 ids per seconde, zou het coderen ongeveer 6 uur duren, en ze passen niet in het geheugen van de meeste computers. Daarom codeert een pipeline voor pretraining het corpus één keer, schrijft hij de ids naar bestanden, en leest hij tijdens het trainen alleen de vensters die hij nodig heeft, rechtstreeks van de schijf.

        **Een voorbeeld uit het dagelijks leven.** Een bibliotheek kopieert geen boek voor elke lezer die één bladzijde wil: de boeken staan genummerd in de kasten, en je pakt de bladzijde die je nodig hebt.
        """),
        draw.project_steps(["Letters en paren tellen", "Een bigram-model door te tellen", "Hetzelfde model, geleerd door een neural network",
                            "Je eigen tokenizer", "Je eigen GPT", "Een klein LLM pretrainen", "Het LLM instructies en voorkeuren laten volgen"],
                           5, here_label="← vandaag", title="Waar je project staat: je eigen LLM"),
    ])
    return


@app.cell(hide_code=True)
def predict_batch():
    _m = C.MEASURED
    guess_batch = L.guess(
        "batch",
        question=f"Op de computer van de course duurt het coderen van de {C.number(_m['chars'] / 1e6, 1, L.lang)} miljoen tekens van de verhalen ongeveer {round(_m['encode'])} seconden. Hoe lang duurt het volgens jou, als de ids eenmaal in bestanden staan, om daar één mini-batch van {C.BATCH} vensters van {C.BLOCK} ids uit te lezen?",
        options={
            "same": f"Ongeveer even lang: {round(_m['encode'])} seconden",
            "tenth": "Ongeveer een tiende daarvan",
            "right": "Een paar duizendsten van een seconde",
        },
        reveal=f"**Een paar duizendsten van een seconde: {C.number(1000 * _m['batch'], 1, L.lang)} ms op de computer van de course.** Voor een mini-batch worden alleen zijn {C.BATCH} vensters van {C.BLOCK + 1} ids gelezen, hoe groot de bestanden ook zijn, en er wordt niets opnieuw gecodeerd. De {_m['shards']} bestanden openen duurde {C.number(1000 * _m['open'], 1, L.lang)} ms: er wordt niets gelezen tot er om een venster gevraagd wordt.",
    )
    guess_batch
    return (guess_batch,)


@app.cell(hide_code=True)
def text_idea():
    _m = C.MEASURED
    mo.vstack([
        mo.md(r"""
        ## Het idee

        **Eén keer coderen.** Codeer het hele corpus één keer, en schrijf de ids naar bestanden: vanaf dan leest elke run, elke epoch en elke computer de ids in plaats van de tekst.

        **Het kleinste type.** Een id van een vocabulary van $V$ ids is een getal van 0 tot en met $V - 1$. Een getal van 16 bits zonder teken, `np.uint16`, bevat 0 tot en met 65.535, in 2 bytes: genoeg voor elke vocabulary tot en met $2^{16}$ = 65.536 ids, zoals de 1.257 van je tokenizer en de 50.257 van GPT-2. Llama 3, met 128.256, heeft `np.uint32` nodig, 4 bytes. Het standaardtype van NumPy, `int64`, neemt 8 bytes per id in, en een lijst ints van Python nog veel meer.

        **Token shards.** Een groot corpus wordt geknipt in bestanden van een vast aantal ids elk: token shards. Veel processen kunnen tegelijk hun eigen shards coderen, schrijven en lezen; een kapot of kwijtgeraakt bestand kost één shard; en een run kan na elke shard stoppen en later verdergaan. Een shard bevat de ids als kale bytes, het ene getal na het andere, zonder header: `ids.tofile(path)` schrijft ze, en alleen het type zegt hoe je ze terugleest.

        **Lezen van de schijf.** `np.memmap(path, dtype=np.uint16, mode="r")` opent een shard als een array die op de schijf blijft: openen kost niets, en `data[s:s + 129]` leest alleen die ids. Een Dataset kan dan elke shard in vensters knippen, zoals `gpt_examples` uit [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl) dat met een lijst deed, en een DataLoader shufflet ze.

        ### Nieuwe woorden en tekens

        | Woord of teken | Uitspraak | Wat het betekent | In code |
        |---|---|---|---|
        | **token shard** | | één bestand met een vast aantal token-ids, opgeslagen als kale getallen van één type; een corpus bestaat uit veel ervan | `shard_0.bin` |
        | **memory-mapped file** | | een array waarvan de getallen in een bestand op de schijf blijven; indexeren leest alleen die getallen | `np.memmap(path, dtype=np.uint16, mode="r")` |
        | `uint16` | joe-int zestien | een heel getal van 16 bits zonder teken: 0 tot en met 65.535, in 2 bytes | `np.uint16` |
        | **naar een bestand** | | de getallen van een array als kale bytes wegschrijven, het ene na het andere; `np.fromfile` leest ze in hun geheel terug | `arr.tofile(path)` |
        | **een tijdelijke map** | | een nieuwe, lege map voor bestanden die je maar even nodig hebt; je krijgt het pad van de map terug | `tempfile.mkdtemp()` |
        """),
        mo.md(r"""
        Het beeld: hoeveel ruimte de verhalen van TinyStories innemen, als tekst en als ids, met de tokenizer van de course.
        """),
        mo.Html(C.sizes_picture([("de tekst", _m["chars"] / 1e6), ("de ids, als int64", 8 * _m["ids"] / 1e6), ("de ids, als uint16", 2 * _m["ids"] / 1e6)],
                                {"title": "Eén corpus, drie groottes", "lang": L.lang,
                                 "note": f"{C.number(_m['ids'], 0, L.lang)} ids, in {_m['shards']} shards van hooguit {C.number(C.SHARD, 0, L.lang)}",
                                 "alt": f"Balken met megabytes: de tekst {C.number(_m['chars'] / 1e6, 1, L.lang)}; de ids als int64 {C.number(8 * _m['ids'] / 1e6, 1, L.lang)}; de ids als uint16 {C.number(2 * _m['ids'] / 1e6, 1, L.lang)}."})),
    ])
    return


@app.cell(hide_code=True)
def text_recipe():
    mo.md(r"""
    ### Recept

    Om een GPT te voeden uit token shards:

    1. Codeer het corpus één keer, met end-of-text tussen de teksten.
    2. Kies het kleinste type dat elke id bevat: `np.uint16` tot en met 65.536 ids, anders `np.uint32`.
    3. Schrijf elk vast aantal ids naar een eigen shard.
    4. Open bij het trainen elke shard met `np.memmap`, alleen om te lezen.
    5. Maak een lijst van het begin van elk venster in elke shard, en lees een venster en de id erna als erom gevraagd wordt.
    6. Laat een DataLoader de vensters shufflen en in mini-batches zetten.
    """)
    return


@app.cell(hide_code=True)
def text_worked_example():
    mo.md(r"""
    ### Uitgewerkt voorbeeld

    Stap 2 tot en met 4 met tien verzonnen ids. `tempfile.mkdtemp()` maakt een nieuwe, lege map voor tijdelijke bestanden, en geeft het pad ervan terug.
    """)
    return


@app.cell
def example_file():
    import tempfile

    import numpy as np

    _ids = [3, 1256, 7, 42, 0, 999, 12, 5, 64, 300]
    _path = f"{tempfile.mkdtemp()}/example.bin"
    # 2. en 3. als getallen van 2 bytes, het ene na het andere
    np.array(_ids, dtype=np.uint16).tofile(_path)
    # 4. open het bestand zonder het te lezen, en lees drie ids
    _data = np.memmap(_path, dtype=np.uint16, mode="r")
    print(len(_data), "ids,", _data.nbytes, "bytes:", _data[4:7])
    # dezelfde bytes, gelezen als een ander type, geven andere getallen
    print(np.fromfile(_path, dtype=np.uint32)[:2])
    return


@app.cell(hide_code=True)
def text_after_example():
    mo.md(r"""
    Tien ids van 2 bytes nemen 20 bytes in op de schijf, en `_data[4:7]` geeft `[  0 999  12]`, de ids op plek 4, 5 en 6. Gelezen als `uint32` geven dezelfde bytes andere getallen: 3 en 1256 worden samen $3 + 1256 \cdot 65.536 = 82.313.219$. Een shard zegt niets over zijn type, dus wie hem leest, moet het weten: deze les bewaart het in een klein bestand naast de shards.
    """)
    return


@app.cell(hide_code=True)
def text_torch_rung():
    mo.md(r"""
    ### Een trede hoger: hetzelfde in PyTorch

    Een GPT heeft zijn vensters nodig als tensors van `int64`, die een embedding layer opzoekt. Een stuk van een memory map is een array van `uint16`; `.tolist()` maakt er een lijst ints van Python van, en `torch.tensor` maakt die `int64`, zoals in [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl). Alleen dat stuk wordt van de schijf gelezen.
    """)
    return


@app.cell
def example_tensor():
    import tempfile as _tempfile

    import numpy as _np
    import torch

    _path = f"{_tempfile.mkdtemp()}/example.bin"
    _np.array([3, 1256, 7, 42, 0, 999, 12, 5, 64, 300], dtype=_np.uint16).tofile(_path)
    _data = _np.memmap(_path, dtype=_np.uint16, mode="r")
    # een venster van 4 ids en de id erna, als tensor
    _ids = torch.tensor(_data[2:7].tolist())
    print(_ids[:-1], _ids[1:], _ids.dtype)
    return


@app.cell(hide_code=True)
def text_after_torch():
    mo.md(r"""
    | NumPy | PyTorch |
    |---|---|
    | `data[s:s + block + 1]`, een array van `uint16` | `torch.tensor(data[s:s + block + 1].tolist())`, van `int64` |
    | het venster en zijn target: `w[:-1]`, `w[1:]` | hetzelfde: `ids[:-1]`, `ids[1:]` |

    Het venster `[7, 42, 0, 999]` heeft als target `[42, 0, 999, 12]`: elke id die erna komt, zoals in [Je GPT in elkaar zetten](/course/open/gpt-assemble?lang=nl).
    """)
    return


@app.cell(hide_code=True)
def text_exercise_write():
    mo.vstack([
        mo.md(r"""
        ## Zelf doen

        ### Oefening 1. Schrijf de shards

        Maak `write_shards(ids, vocab, folder, size)` af: stap 2 en 3 van het recept. Kies `np.uint16` als `vocab`, het aantal verschillende ids, hooguit `2 ** 16` is, en anders `np.uint32`. Schrijf dan elke `size` ids naar een eigen bestand in `folder`, met de namen `shard_0.bin`, `shard_1.bin`, enzovoort; de laatste shard bevat wat overblijft. Hij geeft de lijst met de paden van de bestanden terug. De check schrijft een paar shards in een map die daarna weg mag, en leest ze terug.
        """),
        L.task("Laat `write_shards` elke `size` ids, in het kleinste type, naar een genummerde shard schrijven."),
    ])
    return


@app.cell(hide_code=True)
def exercise_write():
    box_write = L.code_box("write-shards", lines=16)
    box_write
    return (box_write,)


@app.cell(hide_code=True)
def check_write(box_write):
    result_write = L.check("write-shards", box_write.value)
    result_write
    return (result_write,)


@app.cell(hide_code=True)
def hints_write():
    L.hints(
        "write-shards",
        [
            "`range(0, len(ids), size)` geeft de plek waar elke shard begint; `ids[start:start + size]` zijn ids, ook voor de laatste, kortere.",
            "De naam van de volgende shard is zijn nummer, `len(paths)`, in een f-string. Het uitgewerkte voorbeeld laat zien hoe je één shard schrijft.",
        ],
        why="Twee bytes per id en geen kop maken een shard zo klein als kan, en elke id staat op een plek die je kunt uitrekenen: id i van een shard begint bij byte 2i. Daardoor kan `np.memmap` één venster lezen zonder de rest.",
    )
    return


@app.cell(hide_code=True)
def text_demo_write():
    mo.vstack([
        mo.md(f"""
        ### Je verhalen als shards

        Als oefening 1 goed is, codeert de course alle {C.number(C.MEASURED['chars'], 0, L.lang)} tekens van TinyStories één keer, met je `encode_fast` uit [Project: je GPT op je eigen tokens](/course/open/gpt-train-bpe?lang=nl) en de merge rules van je GPT op tokens (of die van de course, als je er zelf geen hebt getraind), en schrijft je `write_shards` ze in shards van {C.number(C.SHARD, 0, L.lang)} ids naar `work/projects/llm/{C.FOLDER}/`. Latere lessen trainen daaruit.
        """),
        L.course_versions(),
    ])
    return


@app.cell(hide_code=True)
def demo_write(box_write, result_write):
    if result_write.state != "correct":
        written = None
        _out = mo.md("*Maak eerst oefening 1 af: de demo gebruikt je `write_shards`.*")
    else:
        _r = C.write_demo(run_code(box_write.value).names["write_shards"])
        written = _r if isinstance(_r, dict) else None
        if _r == "run":
            _out = mo.md("*Je `encode_fast` in `work/projects/llm/fast_encode.py` stopt met een fout. Kijk de oefening ervan nog eens na, of verwijder het bestand om de versie van de course te gebruiken.*")
        else:
            _whose = "de merge rules van je GPT" if _r["source"] == "yours" else "de merge rules van de GPT van de course"
            _out = mo.md(f"""Met {_whose} worden de {C.number(_r['chars'] / 1e6, 1, L.lang)} miljoen tekens {C.number(_r['ids'], 0, L.lang)} ids, in {C.number(_r['seconds'], 1, L.lang)} seconden. Je `write_shards` schreef ze in {_r['shards']} shards, samen {C.number(_r['bytes'] / 1e6, 1, L.lang)} MB: {C.number(_r['bytes'] / _r['ids'], 0, L.lang)} bytes per id, tegen {C.number(_r['text_bytes'] / _r['ids'], 1, L.lang)} bytes tekst per id. De grootste id is {_r['top']}, de end-of-text-id.""")
    _out
    return (written,)


@app.cell(hide_code=True)
def text_exercise_windows():
    mo.vstack([
        mo.md(r"""
        ### Oefening 2. Vensters van de schijf

        Maak de class `TokenWindows(paths, block, dtype)` af: stap 4 en 5. Zijn `__init__` opent elk bestand van `paths` met `np.memmap`, alleen om te lezen, en maakt een lijst met de plek van elk venster als `(k, s)`: shard `k`, begin `s`. De vensters van een shard liggen naast elkaar, om de `block` ids, vanaf 0, zolang er nog één id na het venster komt: een venster loopt nooit door in de volgende shard. `__len__` geeft het aantal vensters, en `__getitem__(i)` leest venster `i` en de id erna, als tensor van `int64`, en geeft het venster en zijn target terug. De check schrijft twee kleine shards en vraagt je class om zijn vensters.
        """),
        L.task("Laat `TokenWindows` elk venster van elke shard, en zijn target, van de schijf lezen."),
    ])
    return


@app.cell(hide_code=True)
def exercise_windows():
    box_windows = L.code_box("token-windows", lines=22)
    box_windows
    return (box_windows,)


@app.cell(hide_code=True)
def check_windows(box_windows):
    result_windows = L.check("token-windows", box_windows.value)
    result_windows
    return (result_windows,)


@app.cell(hide_code=True)
def hints_windows():
    L.hints(
        "token-windows",
        [
            "Een venster dat bij `s` begint, heeft de ids `s` tot en met `s + block` nodig, dus `s + block` moet een plek in de shard zijn: `s < len(shard) - block`. `range` met een stap van `block` geeft zulke beginplekken.",
            "Snijd in `__getitem__` de shard `self.shards[k]` vanaf `s`, `self.block + 1` ids lang, zoals in de trede hierboven.",
        ],
        why="Alleen `(k, s)` bijhouden kost een paar bytes per venster; de ids zelf blijven op de schijf tot een DataLoader erom vraagt. Zo traint een corpus van vele gigabytes op een computer met minder geheugen.",
    )
    return


@app.cell(hide_code=True)
def demo_read(box_windows, result_windows, written):
    mo.stop(result_windows.state != "correct", mo.md("*Maak eerst oefening 2 af: de demo gebruikt je `TokenWindows`.*"))
    _r = None if written is None else C.read_demo(run_code(box_windows.value).names["TokenWindows"])
    if _r is None:
        _out = mo.md("*Deze demo leest de shards die de demo van oefening 1 schrijft: hij draait zodra die demo ze heeft geschreven.*")
    else:
        _same = ("Het zijn precies de vensters die `gpt_examples` van elke shard maakt." if _r["same"] else
                 "Het zijn niet de vensters die `gpt_examples` van elke shard maakt: kijk oefening 2 nog eens na.")
        _text = "" if _r["text"] is None else f"\n\nHet eerste venster van die mini-batch, gedecodeerd:\n\n> {_r['text'][:400].replace(chr(10), ' ')}"
        _out = mo.md(f"""Je `TokenWindows` opent de shards in {C.number(1000 * _r['opened'], 1, L.lang)} ms en vindt {C.number(_r['windows'], 0, L.lang)} vensters van {C.BLOCK} ids. Een DataLoader die ze shufflet, leest één mini-batch met shape {_r['shape']} in {C.number(1000 * _r['batch'], 1, L.lang)} ms. {_same}{_text}""")
    _out
    return


@app.cell(hide_code=True)
def text_exercise_shards():
    mo.md(r"""
    ### Oefening 3. Waarom shards?
    """)
    return


@app.cell(hide_code=True)
def choice_shards_cell():
    choice_shards = L.choice(
        "why-shards",
        question="Een corpus van 2 biljoen token-ids staat in duizenden shards, niet in één bestand. Waarom?",
        options={
            "memmap": "Omdat `np.memmap` geen bestand van meer dan een paar gigabyte kan openen, dus moet een corpus gesplitst worden",
            "right": "Zodat veel processen elk aan hun eigen shards kunnen werken, en een kapot bestand één shard kost",
            "small": "Omdat een shard met minder ids minder bytes per id nodig heeft, zodat het corpus als geheel kleiner wordt",
        },
        correct="right",
        why={
            "memmap": "Op een 64-bits computer opent `np.memmap` bestanden die veel groter zijn dan het geheugen van de computer: daar is hij voor. De reden is het werk rond de bestanden.",
            "small": "Het aantal bytes per id hangt alleen af van het type, 2 voor `uint16`, hoeveel ids een shard ook bevat.",
            "right": "Coderen, schrijven en lezen gaan allemaal tegelijk, shard voor shard, en een run kan bij elke shard stoppen en verdergaan.",
        },
    )
    choice_shards
    return (choice_shards,)


@app.cell(hide_code=True)
def text_connections():
    mo.md(r"""
    ## Afronding

    **Samenhang.** De verhalen van [Project: je GPT op je eigen tokens](/course/open/gpt-train-bpe?lang=nl), nu één keer gecodeerd, met de end-of-text uit [Pre-tokenization en special tokens](/course/open/tok-special-tokens?lang=nl) ertussen; en de Dataset uit [Datasets en DataLoaders](/course/open/torch-dataloader?lang=nl), die nu van de schijf leest. Een echte pipeline maakt zijn corpus vóór deze stap schoon, zoals [Testdata uit de training data houden](/course/open/llm-decontamination?lang=nl) dat met testvragen doet. Latere lessen trainen uit deze shards.
    """)
    return


@app.cell(hide_code=True)
def wrap_up(result_write, result_windows, choice_shards):
    L.wrap_up(result_write, result_windows, choice_shards)
    return


@app.cell(hide_code=True)
def footer():
    L.footer()
    return


if __name__ == "__main__":
    app.run()
